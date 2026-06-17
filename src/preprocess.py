"""
전처리 모듈 — 슬라이스 2: grayscale → GaussianBlur → Canny 엣지 추출.

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
