from statistics import median

import cv2
import pytesseract
from PIL import Image

# 예문으로 쓸 수 있는 줄의 최소 신뢰도(줄 내 단어 신뢰도 중앙값). 인쇄체 줄은 대체로 90 안팎
MIN_LINE_CONFIDENCE = 70

# 마킹 테두리/얼룩 등이 글자처럼 오인식되는 잡음은 대체로 신뢰도가 낮고 알파벳이
# 하나도 없는 경우("@", "-", "1" 등)가 많아 이 둘로 걸러냄. 사전 조회 대상도
# 어차피 알파벳 단어뿐이라 알파벳이 없는 토큰은 애초에 의미가 없음.
MIN_CONFIDENCE = 40


def extract_words(image_bgr):
    # CLAHE(지역 대비 보정)를 시도했으나 실측(TOEIC 지문 사진)에서 오히려 단어 인식이
    # 줄어드는 역효과가 확인돼(61 vs CLAHE 없이 88단어, 신뢰 줄 10 vs 14) 적용 안 함.
    # 조명이 고르지 않은 사진에서 도움이 될 수 있다는 가설은 검증되지 않았고, 이미 괜찮은
    # 사진에서는 명백히 손해라 원본 색상 그대로 Tesseract에 넘김
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(Image.fromarray(rgb), lang="eng", output_type=pytesseract.Output.DICT)

    words = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        if data["conf"][i] < MIN_CONFIDENCE:
            continue
        if not any(ch.isalpha() for ch in text):
            continue
        words.append(
            {
                "text": text,
                "bbox": [data["left"][i], data["top"][i], data["width"][i], data["height"][i]],
                "conf": data["conf"][i],
                # 같은 줄(block/paragraph/line)에 속한 단어를 묶어 예문용 문장을
                # 재구성하는 데 씀 (build_line_text 참고)
                "line": (data["block_num"][i], data["par_num"][i], data["line_num"][i]),
            }
        )

    return words


def build_line_text(words):
    # tesseract가 인식한 "한 줄" 단위로 단어를 묶어 원문 그대로의 문장(조각)을 복원함.
    # 단, 문장이 여러 줄에 걸쳐 줄바꿈된 경우는 이어붙이지 않음 — 한 줄만으로도
    # 예문으로 쓰기엔 충분한 경우가 많고, 여러 줄을 하나의 문장으로 합치려면
    # 문장부호 기반 판단이 추가로 필요해서 범위를 좁혀둠.
    lines = {}
    for word in words:
        lines.setdefault(word["line"], []).append(word)

    line_text = {}
    for key, line_words in lines.items():
        # 손글씨 줄을 Tesseract가 억지로 읽으면 `Pat in Gh Submit an fom` 같은 쓰레기 문장이 되는데,
        # 이걸 "사진 속 문장" 예문으로 보여주면 오해를 줌 → 줄 전체 신뢰도가 낮으면 예문 후보에서 제외
        if median(w["conf"] for w in line_words) < MIN_LINE_CONFIDENCE:
            continue
        line_words.sort(key=lambda w: w["bbox"][0])
        line_text[key] = " ".join(w["text"] for w in line_words)
    return line_text
