"""
한글 텍스트 렌더링 헬퍼 — Pillow(PIL) 기반.

cv2.putText는 비-ASCII 문자(한글 포함)를 렌더할 수 없음.
이 모듈은 BGR numpy 프레임에 PIL.ImageFont + ImageDraw로 한글 텍스트를 합성한다.

API:
    put_kr(frame, text, org, font_size, color, outline_color=None)
        frame        : BGR numpy 배열 (in-place 수정)
        text         : 그릴 문자열 (한글·영어·숫자 모두 가능)
        org          : (x, y) — 텍스트 좌상단 기준점 (cv2.putText와 동일 규약)
        font_size    : 폰트 크기 (pt)
        color        : (B, G, R) BGR 색상
        outline_color: (B, G, R) 외곽선 색상, None이면 외곽선 없음
        outline_px   : 외곽선 두께 (픽셀, 기본 2)

    get_font(size) → PIL.ImageFont.FreeTypeFont
        크기별 폰트 인스턴스를 캐싱해 반환.
        최초 호출 시 KOREAN_FONT_PATH → KOREAN_FONT_FALLBACK 순서로 로드.

주의:
    - PIL은 RGB 순서이므로 내부에서 BGR↔RGB 변환을 수행함.
    - 매 put_kr 호출마다 PIL 이미지로 변환·합성하므로 대량 호출 시 속도 저하 있음.
      성능 최적화 필요 시 배치 드로잉 고려.
"""
from __future__ import annotations

import os
import numpy as np
from typing import Tuple

from src import config

# PIL 임포트 (requirements.txt에 Pillow 포함)
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# 폰트 캐시: {(path, size): PIL.ImageFont.FreeTypeFont}
_font_cache: dict[tuple[str, int], "ImageFont.FreeTypeFont"] = {}

# 로드할 폰트 경로 (최초 로드 시 결정, 이후 고정)
_font_path: str | None = None


def _resolve_font_path() -> str:
    """시스템에서 사용 가능한 한글 폰트 경로를 반환."""
    global _font_path
    if _font_path is not None:
        return _font_path
    for candidate in (config.KOREAN_FONT_PATH, config.KOREAN_FONT_FALLBACK):
        if os.path.isfile(candidate):
            _font_path = candidate
            return _font_path
    raise RuntimeError(
        f"한글 폰트를 찾을 수 없습니다. "
        f"경로 확인: {config.KOREAN_FONT_PATH}, {config.KOREAN_FONT_FALLBACK}"
    )


def get_font(size: int) -> "ImageFont.FreeTypeFont":
    """크기별 폰트 인스턴스를 캐싱해 반환."""
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow가 설치되지 않았습니다: pip install Pillow")
    path = _resolve_font_path()
    key = (path, size)
    if key not in _font_cache:
        # .ttc 파일은 index=0 (기본값) 사용
        _font_cache[key] = ImageFont.truetype(path, size, encoding="unic")
    return _font_cache[key]


def put_kr(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_size: int,
    color: tuple[int, int, int],
    outline_color: tuple[int, int, int] | None = None,
    outline_px: int = 2,
) -> None:
    """
    BGR numpy 프레임에 한글 텍스트를 in-place로 렌더링한다.

    Args:
        frame        : BGR numpy 배열 (H, W, 3), uint8. in-place 수정.
        text         : 렌더링할 문자열.
        org          : (x, y) 텍스트 좌상단 기준점.
        font_size    : 폰트 크기 (pt).
        color        : (B, G, R) 전경색.
        outline_color: (B, G, R) 외곽선색. None이면 외곽선 없음.
        outline_px   : 외곽선 두께 (px).
    """
    if not text:
        return

    font = get_font(font_size)

    # BGR→RGB 변환 (PIL은 RGB)
    rgb_frame = frame[:, :, ::-1].copy()
    pil_img = Image.fromarray(rgb_frame, mode="RGB")
    draw = ImageDraw.Draw(pil_img)

    x, y = org

    # BGR→RGB 색상 변환
    def _bgr2rgb(c: tuple[int, int, int]) -> tuple[int, int, int]:
        return (c[2], c[1], c[0])

    # 외곽선 (가독성 향상)
    if outline_color is not None:
        oc = _bgr2rgb(outline_color)
        for dx in range(-outline_px, outline_px + 1):
            for dy in range(-outline_px, outline_px + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, font=font, fill=oc)

    # 전경 텍스트
    fc = _bgr2rgb(color)
    draw.text((x, y), text, font=font, fill=fc)

    # RGB→BGR 다시 변환 후 frame에 in-place 적용
    result = np.array(pil_img)
    frame[:, :, :] = result[:, :, ::-1]
