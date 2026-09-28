import re

import cv2
import numpy as np
import pytesseract
from PIL import Image

from app.ocr import MIN_CONFIDENCE

# 카카오톡 등으로 전달되며 EXIF 방향 정보가 사라진 사진은 실제로 옆으로/거꾸로
# 찍힌 채 저장되는 경우가 있음. 이 상태로 마킹 검출/OCR을 돌리면 좌표계가
# 어긋나 전혀 매칭이 안 되므로, Tesseract OSD로 회전 각도를 추정해 먼저 바로잡음.
_ROTATE_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

# 디스큐(미세 기울기 보정) 대상으로 삼는 각도 범위. 이보다 작으면(손떨림 수준) 굳이
# 안 건드리고, 이보다 크면 90도 단위 오분류 등 다른 문제일 가능성이 커서 손대지 않음
_DESKEW_MIN_ANGLE = 0.3
_DESKEW_MAX_ANGLE = 10.0


def correct_orientation(image_bgr):
    # 90도 단위(옆으로/거꾸로) 보정 → 그 결과를 기준으로 미세한(몇 도 이내) 기울기까지 보정.
    # 순서가 중요함: 사진이 아예 옆으로 누워 있으면 디스큐용 가로선 검출 자체가 안 먹힘
    image_bgr = _correct_right_angle(image_bgr)
    return _deskew(image_bgr)


def _correct_right_angle(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    try:
        osd = pytesseract.image_to_osd(Image.fromarray(rgb))
    except pytesseract.TesseractError:
        # OSD 실패는 두 가지 경우가 섞여 있음: (a) 글자가 거의 없는 사진(빈 페이지 등)
        # (b) 반대로 글자가 너무 심하게(90도 등) 돌아가 있어서 그 상태로는 Tesseract가
        # 글자를 아예 못 읽어 OSD가 요구하는 최소 문자 수를 못 채우는 경우. 실측 사례
        # (토익 지문 사진이 90도로 저장됨)로 (b)가 실제로 발생함을 확인 — "OSD 실패 =
        # 그대로 두기"로는 (b)를 구제 못 해서, 0/90/180/270도로 직접 돌려보고 실제로
        # 글자가 제일 잘 읽히는 각도를 고르는 방식으로 대체함
        return _best_rotation_by_ocr(image_bgr)

    match = re.search(r"Rotate: (\d+)", osd)
    if not match:
        return image_bgr

    rotate_code = _ROTATE_MAP.get(int(match.group(1)))
    if rotate_code is None:
        return image_bgr

    return cv2.rotate(image_bgr, rotate_code)


# 어느 각도로 돌려도 인쇄된 글자가 거의/전혀 없는 사진(손글씨 노트 등)에서는 점수가 전부 0에
# 가까움. 이때 배경 잡음(책상 무늬, 종이 테두리 등)이 짧은 단어 하나로 잘못 인식돼 우연히 낮은
# 양수 점수가 나오면, 그게 진짜 0점(=텍스트 없음, 올바른 판단)을 이겨버려 엉뚱한 각도로 뒤집는
# 문제가 실측으로 확인됨(tessdata_best 기준 노이즈 138점). 실제로 신뢰할 만한 각도는 단어가
# 여럿 맞아떨어져 훨씬 큰 점수(TOEIC 지문 실측 8144)가 나오므로, 이 정도 노이즈는 못 넘는
# 최소 점수를 넘어야만 회전을 채택하도록 함
_MIN_ROTATION_SCORE = 400


def _best_rotation_by_ocr(image_bgr):
    best_image = image_bgr
    best_score = _ocr_score(image_bgr)

    for rotate_code in _ROTATE_MAP.values():
        candidate = cv2.rotate(image_bgr, rotate_code)
        score = _ocr_score(candidate)
        if score > best_score and score >= _MIN_ROTATION_SCORE:
            best_image, best_score = candidate, score

    return best_image


def _ocr_score(image_bgr):
    # 글자가 실제로 읽히는 각도일수록 신뢰도 있는 알파벳 단어가 많이/진하게 잡힘.
    # image_to_osd처럼 최소 문자 수 미달 시 예외를 던지지 않고 조용히 빈 결과를
    # 주는 image_to_data를 대신 써서, 어느 각도에서든 점수(0 포함)를 항상 얻을 수 있게 함.
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(Image.fromarray(rgb), lang="eng", output_type=pytesseract.Output.DICT)

    score = 0
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        if not text or conf < MIN_CONFIDENCE:
            continue
        if not any(ch.isalpha() for ch in text):
            continue
        score += conf
    return score


def _deskew(image_bgr):
    # 문서 사진은 보통 손으로 들고 찍어서 몇 도씩 삐뚤어짐. 인쇄된 텍스트 줄/표 테두리처럼
    # 페이지 폭에 가깝게 이어지는 가로선의 각도를 Hough 변환으로 추정해 그만큼 되돌림.
    # (minAreaRect로 잉크 픽셀 전체의 기울기를 재는 방식도 있지만, OpenCV 버전마다 각도
    # 기준이 달라 부호가 뒤집히는 문제가 있어 선 각도 기반이 더 안정적)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    h, w = gray.shape[:2]
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 720, threshold=150,
        minLineLength=w // 3, maxLineGap=10,
    )
    if lines is None:
        return image_bgr

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # 세로선(표/사진 테두리 등)은 텍스트 줄과 무관하니 제외, 거의 가로에 가까운 선만
        if abs(angle) <= _DESKEW_MAX_ANGLE:
            angles.append(angle)

    # 선 몇 개만으로 판단하면 노이즈에 취약해서(형광펜 테두리 등) 표본이 충분할 때만 적용
    if len(angles) < 8:
        return image_bgr

    angle = float(np.median(angles))
    if abs(angle) < _DESKEW_MIN_ANGLE:
        return image_bgr

    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image_bgr, matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
