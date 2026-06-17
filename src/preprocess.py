"""
전처리 모듈 — 슬라이스 2+3: grayscale → GaussianBlur → Canny 엣지 추출,
색상 마스크(흰색+노란색), 차선 마스크(색상 ∩ Canny 또는 ∪).

규칙: magic number 없음. 모든 파라미터는 config에서 가져옴.
"""
from __future__ import annotations

import cv2
import numpy as np

from src import config


def to_edges(frame_bgr: np.ndarray) -> np.ndarray:
    """
    BGR 프레임 → 단채널 Canny 엣지 이미지.

    파이프라인:
      1. grayscale 변환
      2. GaussianBlur (config.GAUSSIAN_KSIZE)
      3. Canny (config.CANNY: low, high, aperture)

    반환: uint8 단채널 이미지 (0 또는 255), 입력과 동일한 H×W.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_KSIZE, 0)
    edges = cv2.Canny(
        blurred,
        config.CANNY["low"],
        config.CANNY["high"],
        apertureSize=config.CANNY["aperture"],
    )
    return edges


def color_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """
    BGR 프레임 → 흰색+노란색 차선 후보 이진 마스크.

    파이프라인:
      1. BGR → HSV 변환
      2. cv2.inRange로 흰색 범위(config.HSV_WHITE) 마스크 추출
      3. cv2.inRange로 노란색 범위(config.HSV_YELLOW) 마스크 추출
      4. bitwise_or로 합집합 반환

    반환: uint8 단채널 이미지 (0 또는 255), 입력과 동일한 H×W.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    white_lower = np.array(config.HSV_WHITE["lower"], dtype=np.uint8)
    white_upper = np.array(config.HSV_WHITE["upper"], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, white_lower, white_upper)

    yellow_lower = np.array(config.HSV_YELLOW["lower"], dtype=np.uint8)
    yellow_upper = np.array(config.HSV_YELLOW["upper"], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    return cv2.bitwise_or(mask_white, mask_yellow)


def lane_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """
    BGR 프레임 → Hough 입력용 차선 후보 마스크.

    color_mask(흰+노랑)과 to_edges(Canny)를 config.MASK_COMBINE에 따라 결합:
      "intersection" → bitwise_and (교집합, 기본값, PLAN §5-4)
      "union"        → bitwise_or  (합집합, 폴백)

    반환: uint8 단채널 이미지 (0 또는 255), 입력과 동일한 H×W.
    """
    cmask = color_mask(frame_bgr)
    edges = to_edges(frame_bgr)

    if config.MASK_COMBINE == "intersection":
        return cv2.bitwise_and(cmask, edges)
    else:
        return cv2.bitwise_or(cmask, edges)
