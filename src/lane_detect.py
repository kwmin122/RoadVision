"""
차선 검출 모듈 — 슬라이스 2: HoughLinesP로 원시 선분 검출 + 드로잉.

이 슬라이스에서는 raw 선분만 추출한다.
좌우 분리·폴리핏·시간평활은 이후 슬라이스에서 추가.

규칙: magic number 없음. 모든 파라미터는 config에서 가져옴.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from src import config


def raw_segments(edges: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Canny 엣지 이미지(ROI 적용 후)에서 HoughLinesP로 선분을 검출해 반환.

    파라미터: config.HOUGH (rho, theta_deg, threshold, min_line_len, max_line_gap).
    theta_deg는 라디안으로 변환해서 사용.

    반환: [(x1, y1, x2, y2), ...] — 검출 없으면 빈 리스트.
    """
    theta_rad = math.radians(config.HOUGH["theta_deg"])
    lines = cv2.HoughLinesP(
        edges,
        rho=config.HOUGH["rho"],
        theta=theta_rad,
        threshold=config.HOUGH["threshold"],
        minLineLength=config.HOUGH["min_line_len"],
        maxLineGap=config.HOUGH["max_line_gap"],
    )
    if lines is None:
        return []
    return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in lines[:, 0]]


def draw_segments(
    frame: np.ndarray,
    segments: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> None:
    """
    검출된 선분들을 frame에 직접 그린다 (in-place).

    기본 색상: RED (BGR = 0, 0, 255). 두께=2.
    """
    for x1, y1, x2, y2 in segments:
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
