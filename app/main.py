import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import vocab
from app.handwriting import recognize_marks
from app.llm import LLMConfigError, lookup_word, lookup_words
from app.marks import detect_marks
from app.matching import match_marks_to_words
from app.ocr import build_line_text, extract_words
from app.orientation import correct_orientation

app = FastAPI(title="enn")

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

# Tesseract 단어 신뢰도가 이 값 이상이면 인쇄체로 제대로 읽은 것으로 봄. 실측: 인쇄체 정상 인식
# 83~96, 손글씨를 억지로 읽은 쓰레기 49~65 (어휘 사전 기준은 활용형/굽은 따옴표에서 오판해 폐기)
PRINTED_MIN_CONFIDENCE = 75


@app.get("/", include_in_schema=False)
def root():
    # 모바일에서 IP:8000 만 입력해도 바로 앱 화면이 열리도록
    return RedirectResponse(url="/ui/")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_image(file: UploadFile):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="jpg/png 이미지만 업로드 가능합니다.")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="이미지 용량이 너무 큽니다. (최대 15MB)")

    extension = Path(file.filename or "").suffix or ".jpg"
    file_id = uuid.uuid4().hex
    dest = UPLOAD_DIR / f"{file_id}{extension}"
    dest.write_bytes(data)

    return {"file_id": file_id, "filename": dest.name}


def _find_upload(file_id: str) -> Path:
    matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="해당 file_id의 이미지를 찾을 수 없습니다.")
    return matches[0]


def _load_oriented_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다.")
    return correct_orientation(image)


@app.get("/image/{file_id}")
def oriented_image(file_id: str):
    # /marks, /analyze 의 bbox 는 회전 보정된 이미지 기준 좌표이므로,
    # 프론트가 같은 좌표계의 이미지를 그릴 수 있도록 보정본을 내려줌
    image = _load_oriented_image(_find_upload(file_id))
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise HTTPException(status_code=500, detail="이미지 인코딩에 실패했습니다.")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/ocr/{file_id}")
def run_ocr(file_id: str):
    path = _find_upload(file_id)
    image = _load_oriented_image(path)
    words = extract_words(image)
    return {"file_id": file_id, "words": words}


@app.get("/marks/{file_id}")
def detect_marks_endpoint(file_id: str):
    path = _find_upload(file_id)
    image = _load_oriented_image(path)

    marks = detect_marks(image)
    return {"file_id": file_id, "marks": marks}


@app.get("/dictionary/{word}")
def dictionary_lookup(word: str):
    try:
        result = lookup_word(word)
    except LLMConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail="사전에서 단어를 찾을 수 없습니다.")
    return result


class VocabWord(BaseModel):
    word: str = Field(min_length=1, max_length=100)
    lemma: str | None = Field(default=None, max_length=100)
    phonetic: str | None = Field(default=None, max_length=100)
    part_of_speech: str | None = Field(default=None, max_length=50)
    definition: str | None = Field(default=None, max_length=1000)
    definition_en: str | None = Field(default=None, max_length=1000)
    example: str | None = Field(default=None, max_length=1000)
    example_ko: str | None = Field(default=None, max_length=1000)


class VocabAddRequest(BaseModel):
    words: list[VocabWord] = Field(min_length=1, max_length=200)


@app.get("/vocab")
def vocab_list():
    items = vocab.list_words()
    return {"words": sorted(items, key=lambda i: i["added_at"], reverse=True)}


@app.post("/vocab")
def vocab_add(body: VocabAddRequest):
    added, duplicates = vocab.add_words([w.model_dump() for w in body.words])
    return {"added": len(added), "duplicates": duplicates, "total": len(vocab.list_words())}


@app.delete("/vocab/{word_id}")
def vocab_delete(word_id: str):
    if not vocab.delete_word(word_id):
        raise HTTPException(status_code=404, detail="단어장에서 해당 단어를 찾을 수 없습니다.")
    return {"deleted": word_id}


# 형광펜 하나에 걸린 단어가 이 개수 이하면 "test out" 같은 숙어/구동사로 보고 한 항목으로 묶음.
# 그보다 길면 숙어가 아니라 문장 조각일 가능성이 커서 단어별로 나눔
MAX_PHRASE_WORDS = 4


def _printed_words(mark_words, line_text):
    if 2 <= len(mark_words) <= MAX_PHRASE_WORDS and len({w["line"] for w in mark_words}) == 1:
        text = " ".join(w["text"] for w in mark_words)
        return [{"text": text, "example": line_text.get(mark_words[0]["line"])}]
    return [{"text": w["text"], "example": line_text.get(w["line"])} for w in mark_words]


def _handwritten_words(rec):
    if 2 <= len(rec["words"]) <= MAX_PHRASE_WORDS:
        return [{"text": " ".join(rec["words"]), "example": None, "alternatives": rec["alternatives"]}]
    return [{"text": w, "example": None, "alternatives": rec["alternatives"]} for w in rec["words"]]


@app.get("/analyze/{file_id}")
def analyze_image(file_id: str):
    path = _find_upload(file_id)
    cv_image = _load_oriented_image(path)

    marks = detect_marks(cv_image)
    words = extract_words(cv_image)
    line_text = build_line_text(words)

    # Tesseract가 인쇄체로 확신하며 읽은 표시는 그대로 쓰고, 매칭이 없거나 신뢰도가 낮은 표시
    # (손글씨 등)는 손글씨 인식기로 다시 읽음
    printed = [
        m for m in match_marks_to_words(marks, words)
        if all(w["conf"] >= PRINTED_MIN_CONFIDENCE for w in m["words"])
    ]
    printed_boxes = {tuple(m["bbox"]) for m in printed}
    handwritten = [m for m in marks if tuple(m["bbox"]) not in printed_boxes]
    recognized = recognize_marks(cv_image, [m["bbox"] for m in handwritten])

    entries = [{**m, "words": _printed_words(m["words"], line_text)} for m in printed]
    for mark, rec in zip(handwritten, recognized):
        if not rec["words"]:
            continue
        entries.append({**mark, "words": _handwritten_words(rec)})
    entries.sort(key=lambda e: (e["bbox"][1] // 40, e["bbox"][0]))  # 위→아래, 왼→오 읽기 순서

    # 표시된 모든 단어를 사진 속 문장과 함께 LLM에 한 번에 보내 문맥에 맞는 뜻을 받음
    # 사진 속 문장이 없는 손글씨 단어도, 한 표시에 여러 단어가 걸렸다면("test out") 그 구를 문맥으로 줌
    flat, contexts = [], []
    for e in entries:
        phrase = " ".join(w["text"] for w in e["words"]) if len(e["words"]) > 1 else None
        for w in e["words"]:
            flat.append(w)
            contexts.append(w["example"] or phrase)
    try:
        definitions = lookup_words([{"word": w["text"], "sentence": c} for w, c in zip(flat, contexts)])
    except LLMConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    definition_of = {id(w): d for w, d in zip(flat, definitions)}

    result = []
    for entry in entries:
        word_entries = []
        for word in entry["words"]:
            word_entries.append({
                **word,
                "definition": definition_of[id(word)],
                "from_photo": bool(word["example"]),  # 프론트가 "사진 속 문장" 라벨을 붙이는 용도
            })
        result.append({"type": entry["type"], "color": entry.get("color"), "bbox": entry["bbox"], "words": word_entries})

    return {"file_id": file_id, "marked_words": result}
