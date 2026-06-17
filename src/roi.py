"""
ROI 마스킹 모듈 — 슬라이스 2: 사다리꼴 다각형 마스크 적용.

규칙: magic number 없음. 모든 비율은 config.ROI_TRAPEZOID_RATIO에서 가져옴.
좌표계: 이미지 좌상단 원점, (x, y) 픽셀.
"""
from __future__ import annotations

import cv2
import numpy as np

from src import config


def polygon(W: int, H: int, clip_name: str) -> np.ndarray:
    """
    클립 해상도 키에 맞는 ROI 사다리꼴 다각형을 픽셀 좌표로 반환.

    반환: shape (4, 2) int32 ndarray, 순서=(좌하, 좌상, 우상, 우하).
    """
    key = config.res_key(clip_name)
    ratios = config.ROI_TRAPEZOID_RATIO[key]
    pts = np.array(
        [(int(rx * W), int(ry * H)) for rx, ry in ratios],
        dtype=np.int32,
    )
    return pts


def apply(img: np.ndarray, W: int, H: int, clip_name: str) -> np.ndarray:
    """
    단채널(엣지) 이미지에 ROI 사다리꼴 마스크를 씌워 반환.

    파이프라인:
      1. 검정 마스크 생성
      2. fillPoly로 사다리꼴 내부를 255로 채움
      3. bitwise_and로 마스킹

    반환: img와 동일 shape, ROI 외부는 0.
    """
    pts = polygon(W, H, clip_name)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(img, mask)
    return masked
