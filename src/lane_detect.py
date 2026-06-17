"""
차선 검출 모듈 — 슬라이스 2+3: HoughLinesP 원시 선분 검출,
좌우 분리, 가중 폴리핏, 단일 차선선 드로잉.

규칙: magic number 없음. 모든 파라미터는 config에서 가져옴.
드로잉 색상/두께는 시각 상수이므로 이 모듈에 둬도 무방(PLAN §7 규칙 없음).
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


def split_segments(
    segments: list[tuple[int, int, int, int]],
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """
    Hough 선분 목록을 좌/우 차선 선분으로 분리.

    기준:
      - 기울기 |slope| = |dy/dx| 가 config.SLOPE_RANGE 범위 내인 선분만 통과 (수평·수직 제거).
      - dx == 0 인 선분은 기울기 계산 불가 → 무조건 제외.
      - 이미지 좌표계(좌상단 원점)에서 기울기 부호:
          왼쪽 차선: 왼쪽 아래 → 오른쪽 위 방향 → dy/dx < 0 (음수)
          오른쪽 차선: 왼쪽 위 → 오른쪽 아래 방향 → dy/dx > 0 (양수)

    반환: (left_segments, right_segments)
    """
    slope_min, slope_max = config.SLOPE_RANGE
    left: list[tuple[int, int, int, int]] = []
    right: list[tuple[int, int, int, int]] = []

    for x1, y1, x2, y2 in segments:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue  # 수직선 — 기울기 무한대, SLOPE_RANGE 밖
        slope = dy / dx
        abs_slope = abs(slope)
        if abs_slope < slope_min or abs_slope > slope_max:
            continue  # 수평·수직에 너무 가까운 잡선 제거
        if slope < 0:
            left.append((x1, y1, x2, y2))
        else:
            right.append((x1, y1, x2, y2))

    return left, right


def fit_lane(
    segments: list[tuple[int, int, int, int]],
    y_bottom: int,
    y_top: int,
) -> tuple[int, int] | None:
    """
    선분 목록에서 가중 1차 폴리핏으로 단일 차선 직선을 추정.

    방법:
      - 각 선분의 두 끝점(x, y)을 수집.
      - 가중치 = 선분 길이 (긴 선분일수록 신뢰도 높음).
        끝점 하나당 가중치 = 선분 길이.
      - np.polyfit(ys, xs, 1) 로 x = m*y + b 피팅 (차선은 거의 수직이므로 y→x 피팅).
      - y_bottom, y_top에서 x좌표 산출 → (x_bottom, x_top) 반환.

    반환: (x_bottom, x_top) — 선분이 너무 적으면(끝점 < 4개) None.
    최소 끝점 수 4 = 2선분 × 2끝점. 수학적 최소(2)보다 보수적으로 설정해
    노이즈 단일 선분에서의 가짜 피팅 방지.
    """
    if not segments:
        return None

    xs: list[float] = []
    ys: list[float] = []
    weights: list[float] = []

    for x1, y1, x2, y2 in segments:
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length == 0:
            continue
        xs.extend([x1, x2])
        ys.extend([y1, y2])
        weights.extend([length, length])

    # 최소 점 수 미달 시 None 반환 (너무 적으면 피팅 불안정)
    if len(xs) < config.FIT_MIN_POINTS:
        return None

    coeffs = np.polyfit(ys, xs, 1, w=weights)
    poly = np.poly1d(coeffs)

    x_bottom = int(round(poly(y_bottom)))
    x_top = int(round(poly(y_top)))
    return x_bottom, x_top


def draw_lane_lines(
    frame: np.ndarray,
    left_line: tuple[int, int] | None,
    right_line: tuple[int, int] | None,
    y_bottom: int,
    y_top: int,
) -> None:
    """
    피팅된 좌/우 차선 직선을 frame에 직접 그린다 (in-place).

    left_line, right_line: fit_lane()의 반환값 (x_bottom, x_top) 또는 None.
    None인 경우 해당 차선은 그리지 않음.

    색상: 초록 (BGR = 0, 255, 0). 두께: 8px.
    """
    color = (0, 255, 0)  # 초록
    thickness = 8

    if left_line is not None:
        x_bottom, x_top = left_line
        cv2.line(frame, (x_bottom, y_bottom), (x_top, y_top), color, thickness, lineType=cv2.LINE_AA)

    if right_line is not None:
        x_bottom, x_top = right_line
        cv2.line(frame, (x_bottom, y_bottom), (x_top, y_top), color, thickness, lineType=cv2.LINE_AA)
