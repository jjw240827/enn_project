import json
import os
import threading
import time
import uuid
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
VOCAB_FILE = DATA_DIR / "vocab.json"

# 로그인/계정이 아직 없어서 서버 전체가 단어장 하나를 공유함 (데모 단계 단순화).
# 폰/PC 어느 쪽에서 접속해도 같은 단어장이 보이는 게 목적.
_lock = threading.Lock()

_FIELDS = ("lemma", "phonetic", "part_of_speech", "definition", "definition_en", "example", "example_ko")


def _load():
    if not VOCAB_FILE.exists():
        return []
    try:
        data = json.loads(VOCAB_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _save(items):
    # 쓰는 도중 서버가 죽어도 기존 파일이 깨지지 않도록 임시 파일에 쓴 뒤 교체
    tmp = VOCAB_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, VOCAB_FILE)


def _key(word):
    return word.strip().lower()


def list_words():
    with _lock:
        return _load()


def add_words(new_words):
    """이미 단어장에 있는 단어(대소문자 무시)는 건너뜀. 반환: (추가된 항목들, 중복으로 건너뛴 개수)"""
    with _lock:
        items = _load()
        seen = {_key(i["word"]) for i in items}
        added = []
        for w in new_words:
            key = _key(w["word"])
            if not key or key in seen:
                continue
            seen.add(key)  # 같은 요청 안의 중복도 걸러냄
            added.append({
                "id": uuid.uuid4().hex,
                "word": w["word"].strip(),
                **{f: w.get(f) for f in _FIELDS},
                "added_at": int(time.time()),
            })
        if added:
            _save(items + added)
        return added, len(new_words) - len(added)


def delete_word(word_id):
    with _lock:
        items = _load()
        kept = [i for i in items if i["id"] != word_id]
        if len(kept) == len(items):
            return False
        _save(kept)
        return True
