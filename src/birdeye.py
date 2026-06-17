"""
Bird-eye view 변환 모듈 — M5 (9강 homography).

외부에서 사용하는 함수:
  matrix(clip)      → (M, Minv): 정변환 + 역변환 행렬 (클립당 1회 계산, 이후 캐시)
  warp(img, clip)   → warped  : 탑다운 워프 이미지
  unwarp(img, clip) → unwarped: 역워프 (원근 복원)

규칙:
- 모든 4점 좌표는 config.BIRDEYE에서만 읽는다. 여기에 magic number 없음.
- warpPerspective dsize = (W, H) 순 (cv2 규약 = width-first).
- getPerspectiveTransform 입력은 반드시 np.float32.
"""
from __future__ import annotations

import cv2
import numpy as np

from src import config

# 클립별 (M, Minv) 캐시 — process 생애주기 동안 재계산 없이 재사용
_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def matrix(clip: str) -> tuple[np.ndarray, np.ndarray]:
    """
    clip에 해당하는 homography 행렬 (M, Minv)을 반환한다.

    최초 호출 시 config.BIRDEYE[clip]["src"] / ["dst"]에서 읽어 계산하고 캐시.
    이후 호출은 캐시에서 즉시 반환.

    Returns:
        M    — 원근 → 탑다운 변환 행렬 (3×3 float64)
        Minv — 탑다운 → 원근 역변환 행렬 (3×3 float64)

    Raises:
        KeyError: 클립이 config.BIRDEYE에 없는 경우
        ValueError: src/dst 포인트가 None인 경우
    """
    if clip in _cache:
        return _cache[clip]

    cfg = config.BIRDEYE[clip]
    if cfg["src"] is None or cfg["dst"] is None:
        raise ValueError(
            f"config.BIRDEYE['{clip}']의 src/dst가 None입니다. "
            "PLAN.md §5-11의 4점을 먼저 채워주세요."
        )

    src = np.float32(cfg["src"])  # shape (4, 2)
    dst = np.float32(cfg["dst"])

    M    = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    _cache[clip] = (M, Minv)
    return M, Minv


def _frame_size(clip: str) -> tuple[int, int]:
    """config.CLIPS에서 (W, H) 반환."""
    c = config.CLIPS[clip]
    return c["width"], c["height"]


def warp(img: np.ndarray, clip: str) -> np.ndarray:
    """
    img를 탑다운(bird-eye) 뷰로 변환한다.

    Args:
        img : 입력 이미지 (BGR 또는 grayscale). 크기는 clip의 W×H여야 함.
        clip: config.CLIPS의 키.

    Returns:
        warped: 동일 크기(W×H)의 탑다운 뷰 이미지.
    """
    M, _ = matrix(clip)
    W, H = _frame_size(clip)
    return cv2.warpPerspective(img, M, (W, H))


def unwarp(img: np.ndarray, clip: str) -> np.ndarray:
    """
    탑다운 이미지를 원래 원근 뷰로 역변환한다.

    Args:
        img : 탑다운 이미지 (warp 출력과 같은 크기).
        clip: config.CLIPS의 키.

    Returns:
        unwarped: 원근 복원된 이미지.
    """
    _, Minv = matrix(clip)
    W, H = _frame_size(clip)
    return cv2.warpPerspective(img, Minv, (W, H))
