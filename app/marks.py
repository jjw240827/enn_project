import cv2
import numpy as np

# 실제 촬영 사진은 조명/카메라/JPEG 압축 때문에 합성 이미지보다 채도가 훨씬 낮게 나옴.
# 빨강/분홍 계열은 OpenCV HSV의 Hue가 0/180 경계에서 wrap-around 되므로 두 구간으로 나눠서 처리.
#
# 노랑/연두의 밝기(V) 하한은 낮게 잡음: 같은 형광펜이라도 촬영 밝기/그림자에 따라 V가 41~135(어두운 사진)
# ~110~214(밝은 사진)로 크게 달라짐(실측). 하한이 높으면(120) 어두운 사진에서 형광펜 픽셀 절반이
# 탈락해 밀도가 떨어지고 검출을 놓침. 나무 책상(H 8~18, V 21~145)은 색조가 노랑 범위(H≥20)와
# 겹치지 않아 이 완화로 딸려 들어오지 않음. 주황(H 5~19)은 나무와 겹치므로 하한을 유지함.
HIGHLIGHTER_COLOR_RANGES = {
    "yellow": ((20, 40, 60), (35, 255, 255)),
    "green": ((36, 30, 60), (85, 255, 255)),
    "orange": ((5, 40, 120), (19, 255, 255)),
}
PINK_RED_RANGES = (
    ((0, 15, 120), (10, 60, 255)),
    ((170, 15, 120), (179, 60, 255)),
)

MIN_HIGHLIGHT_AREA = 500

# 형광펜 한 덩어리가 이미지에서 차지하는 최대 비율. 실측 진짜 형광펜은 최대 0.5%였고,
# 사진 가장자리의 나무 책상 덩어리는 2.1%였음
MAX_HIGHLIGHT_AREA_RATIO = 0.015

# 형광펜은 글자 뒤를 넓게 칠하므로 박스 대비 채워진 비율(밀도)이 높음(실측 0.84~0.87).
# JPEG 압축으로 생기는 글자 테두리의 색 번짐은 듬성듬성 흩어져 있다가 morphology로
# 우연히 하나의 큰 박스로 뭉쳐지는데, 이 경우 밀도가 낮음(실측 0.38~0.75)
MIN_HIGHLIGHT_DENSITY = 0.75

# 연두/노랑/주황 형광펜은 기준을 완화: 노트에 손으로 칠한 형광펜은 모양이 둥글고 그 위에
# 진한 손글씨 획이 덮여서 밀도가 0.61~0.75로 나옴(실측, 5개 전부 기존 기준 0.75 미달).
# 반면 이 색 계열의 후보 덩어리는 잡음 없이 형광펜뿐이었음. 분홍/빨강은 빨간 펜 동그라미/
# 색번짐(한글 텍스트 근처)이 형광펜으로 오검출되므로 기존의 엄격한 기준을 유지함.
#
# 회전 사각형 기준 밀도(실측, 사진 7장): 진짜 형광펜 전부 ≥0.63, 가짜(나무 조각) 0.39·0.55 → 0.58
MIN_WARM_HIGHLIGHT_DENSITY = 0.58


def _color_masks(hsv):
    masks = {color: cv2.inRange(hsv, np.array(lower), np.array(upper))
             for color, (lower, upper) in HIGHLIGHTER_COLOR_RANGES.items()}
    pink_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in PINK_RED_RANGES:
        pink_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    masks["pink"] = pink_mask
    return masks


def _dominant_color(color_masks, bbox):
    x, y, w, h = bbox
    best_color, best_count = "unknown", 0
    for color, mask in color_masks.items():
        count = cv2.countNonZero(mask[y:y + h, x:x + w])
        if count > best_count:
            best_color, best_count = color, count
    return best_color


def _highlights_from_mask(mask, color_masks, min_density, rotated=False):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = MAX_HIGHLIGHT_AREA_RATIO * mask.shape[0] * mask.shape[1]
    results = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_HIGHLIGHT_AREA or area > max_area:
            continue
        bbox = cv2.boundingRect(contour)
        x, y, w, h = bbox
        # 노트가 기울어 찍히면 형광펜도 기울어서 축 정렬 박스가 부풀어 밀도가 낮게 나옴(실측: 같은
        # 형광펜이 0.55로 나와 기준에 0.001 차이로 탈락). rotated=True면 회전 사각형 기준으로 잼.
        if rotated:
            (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
            box_area = max(rw * rh, 1)
        else:
            box_area = w * h
        if area / box_area <= min_density:
            continue
        results.append({"type": "highlight", "color": _dominant_color(color_masks, bbox), "bbox": [x, y, w, h]})
    return results


def detect_highlights(image_bgr):
    # 형광펜 위에 놓인 글자가 진하고 빽빽할수록(예: 긴 단어) 특정 색상 하나의
    # 범위만으로는 박스 안에서 그 색이 차지하는 비율(밀도)이 낮게 측정될 수 있음
    # (실측: "convention"은 노랑 단독 0.59, 초록 단독 0.25로 각각 기준 미달이었으나
    # 실제로는 두 범위에 걸쳐 칠해진 하나의 형광펜이었음). 노랑/초록/주황은 색상별로
    # 따로 판정하지 않고 합친 마스크로 덩어리를 찾은 뒤, 어느 색이 우세한지만 나중에 판별.
    #
    # 핑크/빨강은 따로 처리함: 이 범위(특히 순수 빨강 H~0-10)가 빨간 밑줄펜과 겹쳐서,
    # 합치면 옆에 붙은 밑줄 마킹까지 하나의 덩어리로 잘못 병합되는 문제가 있었음(실측:
    # "convention" 형광펜+빨간 밑줄이 인접해 있던 사진에서 density가 0.78→0.74로 떨어져
    # 기준 미달이 됨).
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    color_masks = _color_masks(hsv)

    warm_mask = color_masks["yellow"] | color_masks["green"] | color_masks["orange"]
    return (_highlights_from_mask(warm_mask, color_masks, MIN_WARM_HIGHLIGHT_DENSITY, rotated=True)
            + _highlights_from_mask(color_masks["pink"], color_masks, MIN_HIGHLIGHT_DENSITY))


def detect_marks(image_bgr):
    return detect_highlights(image_bgr)
