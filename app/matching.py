HIGHLIGHT_MATCH_THRESHOLD = 0.4


def _intersection_area(bbox_a, bbox_b):
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)

    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _word_coverage(mark_bbox, word_bbox):
    # 표준 IoU 대신 "단어 영역이 표시 영역에 덮인 비율"을 사용.
    # 형광펜은 보통 단어보다 여유 있게 칠해져서 union이 커지고 표준 IoU는
    # 낮게 나오는 반면, 단어 기준 커버리지는 이런 경우에도 안정적으로 잡힘.
    word_area = word_bbox[2] * word_bbox[3]
    if word_area == 0:
        return 0.0
    return _intersection_area(mark_bbox, word_bbox) / word_area


def match_marks_to_words(marks, words):
    matched = []
    for mark in marks:
        covered_words = [w for w in words if _word_coverage(mark["bbox"], w["bbox"]) >= HIGHLIGHT_MATCH_THRESHOLD]
        if not covered_words:
            # 실제 단어와 겹치지 않는 표시는 페이지 밖 배경(책상, 인쇄 디자인 요소 등)을
            # 잘못 검출한 경우가 대부분이라 결과에서 제외
            continue
        # 텍스트뿐 아니라 bbox/line 정보까지 그대로 넘겨야 호출부에서 사전 조회,
        # 사진 속 문장 재구성(예문용) 등에 활용할 수 있음
        matched.append({**mark, "words": covered_words})
    return matched
