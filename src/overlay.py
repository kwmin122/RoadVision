"""
오버레이 렌더링 모듈 — Slice 5 (M4): 주행영역 폴리곤 + 차선선 + LDW 경고 + HUD.

구성:
  draw_drivable_area()  : 좌우 차선 사이 초록 반투명 폴리곤
  draw_lane_lines_ldw() : 차선선 (경고 ON 시 이탈 측 빨강으로 강조)
  draw_ldw_banner()     : 경고 ON 시 상단 빨강 배너 + 텍스트
  draw_hud()            : 오프셋 수치 + 경고 상태 텍스트 HUD
  render_frame()        : 위 4가지를 순서대로 합성 (외부 호출 진입점)

색상 상수 (BGR):
  GREEN_FILL  = (0, 200, 0)  — 주행영역 폴리곤 채움색
  GREEN_LINE  = (0, 255, 0)  — 정상 차선선
  RED_LINE    = (0, 0, 255)  — 경고 시 이탈 차선선
  RED_BANNER  = (0, 0, 200)  — 경고 배너 배경
  WHITE_TEXT  = (255, 255, 255) — 텍스트 전경
  YELLOW_HUD  = (0, 220, 255)  — HUD 수치 강조

규칙: 드로잉 색상/두께 외 임계·크기 상수는 config.LDW에서 가져옴.
"""
from __future__ import annotations

import cv2
import numpy as np

from src import config

# ---------- 색상 상수 ----------
_GREEN_FILL = (0, 200, 0)
_GREEN_LINE = (0, 255, 0)
_RED_LINE   = (0, 0, 255)
_RED_BANNER = (0, 0, 180)
_WHITE      = (255, 255, 255)
_YELLOW_HUD = (0, 220, 255)
_BLACK      = (0, 0, 0)

_LINE_THICKNESS = 8


def draw_drivable_area(
    frame: np.ndarray,
    left_fit: tuple[int, int],
    right_fit: tuple[int, int],
    y_bottom: int,
    y_top: int,
) -> None:
    """
    좌우 차선 피팅 사이의 주행가능 영역을 반투명 초록 폴리곤으로 채운다 (in-place).

    폴리곤 꼭짓점 순서 (시계 방향):
      left_bottom → left_top → right_top → right_bottom

    fill_alpha: config.LDW["fill_alpha"] (0.0~1.0).
    """
    alpha = config.LDW["fill_alpha"]

    lx_bot, lx_top = left_fit
    rx_bot, rx_top = right_fit

    pts = np.array([
        [lx_bot, y_bottom],
        [lx_top, y_top],
        [rx_top, y_top],
        [rx_bot, y_bottom],
    ], dtype=np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], _GREEN_FILL)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_lane_lines_ldw(
    frame: np.ndarray,
    left_fit: tuple[int, int] | None,
    right_fit: tuple[int, int] | None,
    y_bottom: int,
    y_top: int,
    warning: bool,
    side: str | None,
) -> None:
    """
    좌/우 차선선을 그린다. 경고 ON 시 이탈 방향 차선을 빨간색으로 강조.

    warning=True, side="LEFT"  → 왼쪽 차선선 빨강
    warning=True, side="RIGHT" → 오른쪽 차선선 빨강
    그 외 → 양쪽 모두 초록
    """
    def _color(lane_side: str) -> tuple[int, int, int]:
        if warning and side == lane_side:
            return _RED_LINE
        return _GREEN_LINE

    if left_fit is not None:
        lx_bot, lx_top = left_fit
        cv2.line(frame,
                 (lx_bot, y_bottom), (lx_top, y_top),
                 _color("LEFT"), _LINE_THICKNESS, lineType=cv2.LINE_AA)

    if right_fit is not None:
        rx_bot, rx_top = right_fit
        cv2.line(frame,
                 (rx_bot, y_bottom), (rx_top, y_top),
                 _color("RIGHT"), _LINE_THICKNESS, lineType=cv2.LINE_AA)


def draw_ldw_banner(
    frame: np.ndarray,
    side: str,
) -> None:
    """
    경고 ON 시 화면 상단에 빨강 배너와 텍스트를 그린다 (in-place).

    배너 높이: config.LDW["banner_height"] (px).
    표시 텍스트: "LANE DEPARTURE → LEFT" 또는 "LANE DEPARTURE → RIGHT".
    """
    H, W = frame.shape[:2]
    bh = config.LDW["banner_height"]

    # 배너 배경 (반투명) — 불투명도: config.LDW["banner_alpha"]
    banner_alpha = config.LDW["banner_alpha"]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, bh), _RED_BANNER, -1)
    cv2.addWeighted(overlay, banner_alpha, frame, 1 - banner_alpha, 0, frame)

    text = f"LANE DEPARTURE  ->  {side}"
    font      = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0
    thickness  = 2

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = (W - tw) // 2
    ty = (bh + th) // 2

    # 윤곽선 (가독성)
    cv2.putText(frame, text, (tx, ty), font, font_scale, _BLACK,  thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty), font, font_scale, _WHITE,  thickness,     cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    off: float | None,
    warning: bool,
    side: str | None,
    frame_no: int,
    total_frames: int,
    n_segs: int,
    left_state: str,
    right_state: str,
) -> None:
    """
    화면 좌하단 HUD: 오프셋 수치 + 경고 상태 + 프레임 정보.

    표시 항목:
      - frame N/total
      - L:<state_abbr>  R:<state_abbr>  segs:<n>
      - offset: -0.12  (또는 N/A)
      - WARN: OFF | LEFT | RIGHT
    """
    H, W = frame.shape[:2]

    _abbr = {
        "raw_detected":        "raw",
        "rejected_as_outlier": "rej",
        "held_from_previous":  "hld",
        "consecutive_missing": "mis",
    }

    off_str = f"{off:+.3f}" if off is not None else "N/A"
    warn_str = f"{side}" if (warning and side) else "OFF"
    warn_color = _RED_LINE if warning else _GREEN_LINE

    lines = [
        (f"frame {frame_no}/{total_frames}  "
         f"L:{_abbr.get(left_state, left_state)}  "
         f"R:{_abbr.get(right_state, right_state)}  "
         f"segs:{n_segs}",
         _WHITE),
        (f"offset: {off_str}", _YELLOW_HUD),
        (f"WARN: {warn_str}", warn_color),
    ]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 2
    line_h     = 24
    x0         = 10
    y0         = H - line_h * len(lines) - 8

    for i, (text, color) in enumerate(lines):
        y = y0 + i * line_h
        # 검정 외곽선으로 가독성 확보
        cv2.putText(frame, text, (x0, y), font, font_scale, _BLACK, thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, text, (x0, y), font, font_scale, color,  thickness,     cv2.LINE_AA)


def draw_birdeye_pip(
    frame: np.ndarray,
    warped: np.ndarray,
) -> None:
    """
    Bird-eye 탑다운 뷰를 프레임 우상단에 PiP(Picture-in-Picture)로 합성한다 (in-place).

    레이아웃:
      - 폭: config.BIRDEYE_PIP["width_ratio"] × frame.W
      - 위치: 우상단, LDW 배너 아래 (top_offset)
      - 테두리: border_px 두께의 흰 사각형
      - 레이블: "BIRD-EYE" (소형 텍스트, 패널 좌하단)

    Args:
        frame  : 출력 프레임 (BGR, in-place 수정).
        warped : birdeye.warp()로 생성된 탑다운 BGR 이미지.
    """
    pip_cfg = config.BIRDEYE_PIP
    H, W = frame.shape[:2]

    # PiP 축소 크기 계산 (종횡비 유지)
    pip_w = int(W * pip_cfg["width_ratio"])
    pip_h = int(pip_w * H / W)  # 원본과 동일 종횡비

    top_offset = pip_cfg["top_offset"]
    border     = pip_cfg["border_px"]

    # 우상단 좌표 (테두리 포함)
    x1 = W - pip_w - border
    y1 = top_offset
    x2 = W - border
    y2 = y1 + pip_h

    # 범위 체크 — 프레임 바깥으로 나가면 클램핑
    y2 = min(y2, H)
    pip_h_actual = y2 - y1
    pip_w_actual = x2 - x1

    if pip_w_actual <= 0 or pip_h_actual <= 0:
        return

    # warped 축소
    small = cv2.resize(warped, (pip_w_actual, pip_h_actual))
    frame[y1:y2, x1:x2] = small

    # 테두리 (흰색 사각형)
    bx1 = x1 - border
    by1 = y1 - border
    bx2 = x2 + border
    by2 = y2 + border
    cv2.rectangle(
        frame,
        (max(bx1, 0), max(by1, 0)),
        (min(bx2, W - 1), min(by2, H - 1)),
        pip_cfg["border_color"],
        border,
    )

    # "BIRD-EYE" 레이블 (패널 좌하단 안쪽)
    label      = "BIRD-EYE"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness  = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    lx = x1 + 4
    ly = y2 - 6
    # 검정 외곽선으로 가독성 확보
    cv2.putText(frame, label, (lx, ly), font, font_scale, _BLACK,              thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, label, (lx, ly), font, font_scale, pip_cfg["label_color"], thickness,     cv2.LINE_AA)


def render_frame(
    frame: np.ndarray,
    left_fit: tuple[int, int] | None,
    right_fit: tuple[int, int] | None,
    y_bottom: int,
    y_top: int,
    off: float | None,
    warning: bool,
    side: str | None,
    frame_no: int,
    total_frames: int,
    n_segs: int,
    left_state: str,
    right_state: str,
) -> None:
    """
    한 프레임에 모든 오버레이를 합성한다 (in-place).

    순서:
      1. 주행영역 반투명 폴리곤 (양쪽 차선이 있을 때만)
      2. 차선선 (경고 방향은 빨강)
      3. 경고 배너 (warning ON 시만)
      4. HUD
    """
    # 1. 주행영역 폴리곤
    if left_fit is not None and right_fit is not None:
        draw_drivable_area(frame, left_fit, right_fit, y_bottom, y_top)

    # 2. 차선선 (폴리곤 위에 그려야 선이 보임)
    draw_lane_lines_ldw(frame, left_fit, right_fit, y_bottom, y_top, warning, side)

    # 3. 경고 배너
    if warning and side is not None:
        draw_ldw_banner(frame, side)

    # 4. HUD
    draw_hud(frame, off, warning, side,
             frame_no, total_frames, n_segs, left_state, right_state)
