"""손글씨 단어 인식 (오프라인, TrOCR).

Tesseract는 인쇄체 전용이라 손글씨 노트는 표시 영역만 잘라서 읽어도 못 읽음(실측:
`investigate`→`[investiga`, `submit`→`Gakmit`). 손글씨로 학습된 TrOCR을 형광펜이 칠해진
영역에만 적용하고, 모델 출력을 영어 어휘 사전으로 보정해 인식률을 높임.
"""
import math
import os
import re
import threading
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from spellchecker import SpellChecker

# small을 기본으로 씀 (같은 15개 손글씨 크롭 실측): base 9/15 · 사진당 ~37초 vs small 8/15 · ~4.5초.
# base는 디코더가 24층이라 CPU에서 너무 느림. 빔을 늘리거나 여러 크롭 변형을 합쳐도(앙상블)
# small 정확도는 그대로(8/15)이고 시간만 3~6배 늘어서 빔 4 + 단일 크롭으로 고정.
MODEL_NAME = os.environ.get("HANDWRITING_MODEL", "microsoft/trocr-small-handwritten")
# 일반 빔서치는 빔 4개가 전부 같은 단어의 변형(`investigator`의 대소문자/공백 차이)이라 후보가
# 다양하지 않았음. 다양성 빔서치(그룹별로 서로 다른 단어를 내도록 페널티)로 바꾸면 1순위 정확도는
# 그대로(8/15)지만 후보 칩으로 복구 가능한 정답이 3개→6개로 늘어남(사진당 4.5초→6.3초).
NUM_BEAMS = int(os.environ.get("HANDWRITING_BEAMS", "8"))
BEAM_GROUPS = 4
DIVERSITY_PENALTY = 1.0
# 8 vCPU VM에서 스레드를 전부(8) 쓰면 오히려 느림(실측 58~63초 vs 4스레드 37초)
NUM_THREADS = int(os.environ.get("HANDWRITING_THREADS", "4"))
# 단어 하나는 토큰 3~6개라 상한을 낮게 잡아도 되고, 빔이 끝나지 않고 길어지는 낭비를 막음
MAX_NEW_TOKENS = 16

# 형광펜 bbox 주변 여백(px). 가로를 넓게 잡으면 이웃 단어의 획(하이픈/괄호)이 글자로 읽힘
CROP_PAD_X = 6
CROP_PAD_Y = 6

# 이 가로세로비를 넘는 형광펜은 여러 단어에 걸친 것으로 봄 (실측: 단어 1개는 2.1~3.9)
MULTIWORD_ASPECT = 5.0

# 어휘 재순위 파라미터 (로그확률 단위)
EDGE_STRIP_PENALTY = 1.0   # 가장자리 잡음 글자 1개를 떼어낼 때의 페널티
EDIT_PENALTY = 1.0         # 편집거리 1 교정의 페널티
WORD_BONUS = 1.5           # 사전에 있는 단어에 주는 가산점
FREQ_WEIGHT = 0.05         # 빈도가 높은 단어를 살짝 선호
MAX_EDGE_STRIP = 2
MAX_PHRASE_WORDS = 3       # 형광펜 하나에서 띄어 읽힌 단어를 구로 인정하는 최대 개수
PHRASE_SPLIT_PENALTY = 1.0 # 단어를 더 쪼갤 때마다의 페널티

# 후보 단어의 최소 빈도: 인식된 단어 빈도 대비 비율 (너무 희귀한 단어를 후보에서 제외)
_MIN_ALTERNATIVE_FREQ_RATIO = 0.05

_spell = SpellChecker()
_freq = _spell.word_frequency
_TOTAL = sum(_freq.dictionary.values()) or 1

_model_lock = threading.Lock()
_model = None
_processor = None


def is_english_word(word):
    word = word.lower()
    return len(word) >= 2 and _freq.dictionary.get(word, 0) > 0


def _log_freq(word):
    return math.log((_freq.dictionary.get(word, 0) + 1) / _TOTAL)


def _load_model():
    global _model, _processor
    if _model is None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        torch.set_num_threads(min(NUM_THREADS, os.cpu_count() or NUM_THREADS))
        _processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
        _model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).eval()
    return _model, _processor


def _preprocess(image_bgr, bbox):
    # 형광펜은 색이 진해서 그대로 넣으면 배경이 어두워 인식이 나빠짐(실측 color 입력 3/15 vs 아래 방식
    # 4~6/15). HSV의 V(밝기) 채널은 형광펜 칠을 종이만큼 밝게, 잉크는 어둡게 남기므로 이를 쓰고
    # 종이 밝기를 흰색으로 끌어올려 배경을 정리함.
    x, y, w, h = bbox
    height, width = image_bgr.shape[:2]
    crop = image_bgr[max(0, y - CROP_PAD_Y):min(height, y + h + CROP_PAD_Y),
                     max(0, x - CROP_PAD_X):min(width, x + w + CROP_PAD_X)]
    v = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    hi, lo = np.percentile(v, 60), np.percentile(v, 1)
    gray = np.clip((v - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(gray).convert("RGB")


def _n_best(images):
    import torch

    with _model_lock:
        model, processor = _load_model()
        with torch.no_grad():
            pixel_values = processor(images=images, return_tensors="pt").pixel_values
            out = model.generate(
                pixel_values,
                num_beams=NUM_BEAMS,
                num_beam_groups=BEAM_GROUPS,
                diversity_penalty=DIVERSITY_PENALTY,
                num_return_sequences=NUM_BEAMS,
                max_new_tokens=MAX_NEW_TOKENS,
                output_scores=True,
                return_dict_in_generate=True,
            )
    texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
    scores = out.sequences_scores.tolist()
    return [
        [(texts[i * NUM_BEAMS + j].strip(), scores[i * NUM_BEAMS + j]) for j in range(NUM_BEAMS)]
        for i in range(len(images))
    ]


def rerank_candidates(candidates):
    """모델 n-best를 영어 어휘로 재순위해 [(단어, 점수, 확률비중)] 를 점수 내림차순으로 반환.

    - 같은 문자열이 여러 빔에서 나오면 확률을 합산(빔 합의)
    - 크롭 가장자리에 걸린 이웃 획이 글자로 읽힌 경우(예: `claunching`, `issubmit`)를 위해,
      앞/뒤 1~2글자를 떼어낸 변형이 사전 단어면 원본 후보의 확률을 페널티를 곱해 그 변형에 누적
    - 사전에 없는 후보는 편집거리 1의 사전 단어로도 확률을 전이 (`caunching` → `launching`)
    - 사전에 있는 단어에는 가산점
    """
    mass = defaultdict(float)
    phrase_mass = defaultdict(float)   # 띄어쓴 가설 (`test out`)
    for text, score in candidates:
        word = re.sub(r"[^a-z]", "", text.lower())
        if word:
            mass[word] += math.exp(score)
        tokens = re.sub(r"[^a-z]+", " ", text.lower()).split()
        if 2 <= len(tokens) <= MAX_PHRASE_WORDS and all(is_english_word(t) for t in tokens):
            phrase_mass[" ".join(tokens)] += math.exp(score)

    accumulated = defaultdict(float)
    for word, m in mass.items():
        for head in range(MAX_EDGE_STRIP + 1):
            for tail in range(MAX_EDGE_STRIP + 1):
                variant = word[head:len(word) - tail] if tail else word[head:]
                if len(variant) < 3:
                    continue
                if (head or tail) and not is_english_word(variant):
                    continue
                accumulated[variant] += m * math.exp(-EDGE_STRIP_PENALTY * (head + tail))

    # 사전에 없는 후보(예: `caunching`)는 편집거리 1의 사전 단어(`launching`)로도 질량을 전이함.
    # 여러 이웃이 있으면 단어 빈도에 비례해 나눔
    for word, m in mass.items():
        if len(word) < 4 or is_english_word(word):
            continue
        neighbours = [n for n in _spell.edit_distance_1(word) if len(n) >= 3 and is_english_word(n)]
        total_freq = sum(_freq.dictionary[n] for n in neighbours)
        for n in neighbours:
            accumulated[n] += m * math.exp(-EDIT_PENALTY) * _freq.dictionary[n] / total_freq

    total_mass = sum(mass.values()) or 1.0
    scored = []
    for variant, m in accumulated.items():
        score = math.log(m)
        if is_english_word(variant):
            score += WORD_BONUS + FREQ_WEIGHT * _log_freq(variant)
        scored.append((variant, score, min(1.0, m / total_mass)))

    # 모델이 형광펜 하나에서 단어 두세 개를 띄어 읽은 경우(`test out`). 글자를 이어붙인 가설
    # (`testout`)은 사전에 없어 버려지므로 띄어쓴 가설을 따로 세움. 다만 필기체 한 단어가 띄어
    # 읽히는 경우(`launch in` ← `launching`)가 있어 단어 수만큼 분할 페널티를 줌.
    for phrase, m in phrase_mass.items():
        n_words = phrase.count(" ") + 1
        score = math.log(m) + WORD_BONUS - PHRASE_SPLIT_PENALTY * (n_words - 1)
        scored.append((phrase, score, min(1.0, m / total_mass)))

    scored.sort(key=lambda item: -item[1])
    return scored


def _correct_token(token):
    # 여러 단어 형광펜의 개별 토큰 보정: 사전에 없으면 편집거리 1의 가장 흔한 단어로
    token = re.sub(r"[^a-z']", "", token.lower())
    if not token or is_english_word(token):
        return token
    neighbours = _spell.edit_distance_1(token) & set(_freq.dictionary)
    if not neighbours:
        return token
    return max(neighbours, key=lambda w: _freq.dictionary[w])


def recognize_marks(image_bgr, bboxes):
    """각 형광펜 bbox의 손글씨를 읽어 [{"words": [...], "alternatives": [...]}] 반환 (bbox 순서 유지)."""
    if not bboxes:
        return []

    images = [_preprocess(image_bgr, bbox) for bbox in bboxes]
    all_candidates = _n_best(images)

    results = []
    for bbox, candidates in zip(bboxes, all_candidates):
        # 결과가 영어 단어가 아니면 버림: 배경 오검출(책상 무늬, 한글 텍스트 근처 색번짐)을 억지로 읽으면
        # `abcdefghijklhij` 같은 의미 없는 문자열이 나옴(실측 6개 중 5개). 반면 진짜 손글씨 15개는
        # 전부 영어 단어였음. (모델 확신도 컷오프는 진짜도 0.16~1.0으로 넓게 퍼져 쓸 수 없었음)
        aspect = bbox[2] / max(bbox[3], 1)
        if aspect > MULTIWORD_ASPECT:
            tokens = [_correct_token(t) for t in candidates[0][0].split()]
            results.append({"words": [t for t in tokens if is_english_word(t)], "alternatives": []})
            continue

        ranked = rerank_candidates(candidates)
        if not ranked or not all(is_english_word(t) for t in ranked[0][0].split()):
            results.append({"words": [], "alternatives": []})
            continue
        best = ranked[0][0]
        if " " in best:   # 띄어쓴 여러 단어 (`test out`) — 후보 칩은 제공하지 않음
            results.append({"words": best.split(), "alternatives": []})
            continue
        alternatives = _alternatives(best, [w for w, _, _ in ranked[1:] if " " not in w])
        results.append({"words": [best], "alternatives": alternatives})
    return results


def _alternatives(best, model_ranked, limit=4):
    # 필기체는 모델도 구분 못 하는 글자쌍이 있음(실측: 필기체 r이 l로 보여 `order`가 `older`로 읽힘).
    # 억지로 하나를 확정하는 대신 사용자가 고를 수 있게 후보를 함께 제공.
    # 글자가 한 개만 다른 흔한 단어를 먼저(실제로 헷갈리는 쌍이 대부분 여기 속함),
    # 그다음에 모델이 함께 내놓은 흔한 단어를 채움. 빈도가 매우 낮은 단어(`olde` 같은 고어)는 제외.
    min_freq = _MIN_ALTERNATIVE_FREQ_RATIO * _freq.dictionary.get(best, 0)

    def usable(word):
        return word != best and len(word) >= 3 and is_english_word(word) and _freq.dictionary[word] >= min_freq

    alternatives = []
    neighbours = sorted((n for n in _spell.edit_distance_1(best) if usable(n)), key=lambda n: -_freq.dictionary[n])
    for word in neighbours + [w for w in model_ranked if usable(w)]:
        if word not in alternatives:
            alternatives.append(word)
    return alternatives[:limit]
