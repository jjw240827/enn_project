import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

DICTIONARY_API_URL = "https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
REQUEST_TIMEOUT = 25

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "dictionary_cache.json"

_cache_lock = threading.Lock()


def _load_cache():
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


_cache = _load_cache()

# API가 404로 "이 표제어는 없음"을 확정해준 경우의 캐시 표식. "planning"처럼 활용형은
# 표제어가 없어서 매번 404를 받는데, 이걸 캐싱하지 않으면 원형("plan")을 찾기 전에
# 느린 API를 매번 다시 기다리게 됨(실측 ~40초).
_MISSING = {"missing": True}


def _fetch_word(word):
    # 원본 응답을 통째로 캐싱해두고, 어떤 뜻을 고를지는 조회 시점(_select_definition)에
    # 결정함 — 품사 힌트(동사/명사 등)에 따라 같은 캐시라도 다른 뜻을 고를 수 있어야 해서.
    # 반환: 표제어 dict / _MISSING(404 확정) / None(타임아웃·5xx 등 일시 오류 — 재시도 대상)
    try:
        response = requests.get(DICTIONARY_API_URL.format(word=word), timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        return None

    if response.status_code == 404:
        return _MISSING
    if response.status_code != 200:
        return None

    # 이 API는 불안정해서 200인데 본문이 비어 있거나 JSON이 아닌 경우가 있음 → 일시 오류로 취급
    # (예외가 전파되면 /analyze 전체가 500이 되어 다른 단어 결과까지 다 잃음)
    try:
        entries = response.json()
    except ValueError:
        return None
    if not isinstance(entries, list) or not entries:
        return None
    entry = entries[0]

    meanings = []
    for meaning in entry.get("meanings", []):
        definitions = [d for d in meaning.get("definitions", []) if d.get("definition")]
        if not definitions:
            continue
        meanings.append({"part_of_speech": meaning.get("partOfSpeech"), "definitions": definitions})

    if not meanings:
        return _MISSING

    return {"word": entry.get("word", word), "phonetic": entry.get("phonetic"), "meanings": meanings}


def _pick_from(meanings):
    # 품사별로 여러 뜻을 다 보여주면 오히려 헷갈려함(예: "plan"의 건축 도면 뜻처럼
    # 흔치 않은 의미가 먼저 나오는 경우). 예문이 달려 있는 정의가 대체로 더 흔히
    # 쓰이는 뜻이라, 예문 있는 첫 정의를 우선 고르고 없으면 첫 정의를 씀.
    chosen = None
    for meaning in meanings:
        for definition in meaning["definitions"]:
            if chosen is None:
                chosen = (meaning, definition)
            if definition.get("example"):
                return (meaning, definition)
    return chosen


def _select_definition(entry, prefer_pos=None):
    meanings = entry["meanings"]

    result = None
    if prefer_pos:
        preferred = [m for m in meanings if m["part_of_speech"] == prefer_pos]
        result = _pick_from(preferred)
    if result is None:
        result = _pick_from(meanings)
    if result is None:
        return None

    meaning, definition = result
    return {
        "word": entry["word"],
        "phonetic": entry["phonetic"],
        "part_of_speech": meaning["part_of_speech"],
        "definition": definition["definition"],
        "example": definition.get("example"),
    }


def _clean_word(word):
    # OCR 결과에 붙은 문장부호("leader.")나 표시(동그라미)가 글자를 침범해
    # 생긴 잡음을 제거. 아포스트로피(곧은/굽은 모두)는 didn't 같은 축약형에
    # 필요하므로 남겨둠
    normalized = word.replace("’", "'")
    return re.sub(r"[^a-zA-Z']", "", normalized)


def _stem_candidates(word):
    # OCR로 뽑힌 단어가 사전에 없는 원형이 아니라 활용형(복수형/진행형/과거형)인
    # 경우가 흔함("planning" 등). 정식 형태소 분석기 없이 흔한 영어 접미사 규칙만
    # 적용해 원형 후보를 몇 개 만들고, 원래 단어부터 순서대로 시도함.
    #
    # 각 후보에는 품사 힌트도 같이 붙임 — "-ing"/"-ed"로 끝났던 원래 단어는 거의
    # 항상 동사로 쓰인 것이고 "-s"/"-es"/"-ies"로 끝났던 단어는 거의 항상 명사라서,
    # 원형을 찾은 뒤 그 품사의 뜻을 우선 보여주면 "plan"이 명사(도면)로 잘못
    # 나오는 대신 동사(계획하다) 뜻으로 나오는 식으로 문맥에 더 맞는 뜻을 고를 수 있음.
    w = word.lower()
    candidates = [(w, None)]

    if w.endswith("ies") and len(w) > 4:
        candidates.append((w[:-3] + "y", "noun"))  # studies -> study
    if w.endswith("es") and len(w) > 3:
        candidates.append((w[:-2], "noun"))  # boxes -> box
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        candidates.append((w[:-1], "noun"))  # cats -> cat

    if w.endswith("ied") and len(w) > 4:
        candidates.append((w[:-3] + "y", "verb"))  # tried -> try
    elif w.endswith("ed") and len(w) > 3:
        stem = w[:-2]
        candidates.append((stem, "verb"))  # asked -> ask
        candidates.append((stem + "e", "verb"))  # liked -> like
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            candidates.append((stem[:-1], "verb"))  # stopped -> stop

    # 비교급/최상급 ("older" -> old, "bigger" -> big, "happiest" -> happy)
    for suffix in ("er", "est"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            stem = w[:-len(suffix)]
            if stem.endswith("i"):
                candidates.append((stem[:-1] + "y", "adjective"))
            candidates.append((stem, "adjective"))
            candidates.append((stem + "e", "adjective"))
            if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                candidates.append((stem[:-1], "adjective"))

    if w.endswith("ing") and len(w) > 5:
        stem = w[:-3]
        candidates.append((stem + "e", "verb"))  # making -> make
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            candidates.append((stem[:-1], "verb"))  # planning -> plan, running -> run
        else:
            candidates.append((stem, "verb"))  # going -> go

    seen = set()
    ordered = []
    for candidate, pos in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append((candidate, pos))
    return ordered


def _lookup_exact(word):
    key = word.lower()

    with _cache_lock:
        if key in _cache:
            cached = _cache[key]
            return None if cached.get("missing") else cached

    result = _fetch_word(word)

    # 일시적 오류(None)는 캐싱하지 않음 - 다음 조회 때 재시도할 수 있게 둠.
    # 정상 응답과 404 확정(_MISSING)만 캐싱
    if result is not None:
        with _cache_lock:
            _cache[key] = result
            _save_cache(_cache)

    return None if result is _MISSING else result


def lookup_word(word):
    clean_word = _clean_word(word)
    if not clean_word:
        return None

    candidates = _stem_candidates(clean_word)

    # 후보를 순차로 조회하면 API 지연(요청당 ~20초)이 후보 수만큼 누적되므로 동시에 조회하고,
    # 결과는 원래 우선순위(원래 단어 → 원형 후보 순)대로 첫 성공을 채택함
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        entries = list(pool.map(lambda c: _lookup_exact(c[0]), candidates))

    for (candidate, prefer_pos), entry in zip(candidates, entries):
        if entry is not None:
            return _select_definition(entry, prefer_pos)

    return None
