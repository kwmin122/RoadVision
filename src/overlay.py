"""
오버레이 렌더링 모듈 — 3-state LDW 위험 시각화 (SAFE/CAUTION/DANGER) 포함.

구성:
  draw_drivable_area()  : 좌우 차선 사이 반투명 폴리곤 (상태 색상 반영)
  draw_lane_lines_ldw() : 차선선 (경고 ON 시 이탈 측 빨강·두꺼운 선 강조)
  draw_ldw_banner()     : 경고 ON 시 상단 빨강 배너 + 텍스트  ← SAFE에서 기존 동작 유지
  draw_hud()            : 오프셋 수치 + 경고 상태 텍스트 HUD
  draw_danger_border()  : DANGER 시 프레임 전체 두꺼운 플래시 테두리
  draw_caution_strip()  : CAUTION 시 상단 황색 좁은 띠 + 텍스트
  draw_danger_banner()  : DANGER 시 대형 중앙 배너 ("⚠ 차선 이탈 / LANE DEPARTURE")
  draw_drift_arrow()    : DANGER 시 이탈 방향 화살표
  draw_offset_gauge()   : 공통 하단 오프셋 게이지 (색상은 상태별)
  render_frame()        : 위를 순서대로 합성 (외부 호출 진입점, 하위 호환)

색상 상수 (BGR):
  GREEN_FILL  = (0, 200, 0)   — SAFE 주행영역 채움 (기존 동일)
  GREEN_LINE  = (0, 255, 0)   — 정상 차선선
  RED_LINE    = (0, 0, 255)   — DANGER 이탈 차선선
  RED_BANNER  = (0, 0, 200)   — 경고 배너 배경
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
    lane_st: str = "SAFE",
) -> None:
    """
    좌우 차선 피팅 사이의 주행가능 영역을 반투명 폴리곤으로 채운다 (in-place).

    폴리곤 꼭짓점 순서 (시계 방향):
      left_bottom → left_top → right_top → right_bottom

    lane_st: "SAFE"(초록) | "CAUTION"(황색) | "DANGER"(적색)
    fill_alpha: config.LDW["fill_alpha"] (0.0~1.0).
    """
    alpha = config.LDW["fill_alpha"]

    if lane_st == "DANGER":
        fill_color = config.LDW["danger_fill"]
    elif lane_st == "CAUTION":
        fill_color = config.LDW["caution_fill"]
    else:
        fill_color = _GREEN_FILL

    lx_bot, lx_top = left_fit
    rx_bot, rx_top = right_fit

    pts = np.array([
        [lx_bot, y_bottom],
        [lx_top, y_top],
        [rx_top, y_top],
        [rx_bot, y_bottom],
    ], dtype=np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], fill_color)
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
    좌/우 차선선을 그린다. DANGER ON 시 이탈 방향 차선을 빨간색+두꺼운 선으로 강조.

    warning=True, side="LEFT"  → 왼쪽 차선선 빨강+두꺼움
    warning=True, side="RIGHT" → 오른쪽 차선선 빨강+두꺼움
    그 외 → 양쪽 모두 초록
    """
    danger_thick = config.LDW["danger_line_thick"]

    def _color(lane_side: str) -> tuple[int, int, int]:
        if warning and side == lane_side:
            return _RED_LINE
        return _GREEN_LINE

    def _thick(lane_side: str) -> int:
        if warning and side == lane_side:
            return danger_thick
        return _LINE_THICKNESS

    if left_fit is not None:
        lx_bot, lx_top = left_fit
        cv2.line(frame,
                 (lx_bot, y_bottom), (lx_top, y_top),
                 _color("LEFT"), _thick("LEFT"), lineType=cv2.LINE_AA)

    if right_fit is not None:
        rx_bot, rx_top = right_fit
        cv2.line(frame,
                 (rx_bot, y_bottom), (rx_top, y_top),
                 _color("RIGHT"), _thick("RIGHT"), lineType=cv2.LINE_AA)


def draw_ldw_banner(
    frame: np.ndarray,
    side: str,
) -> None:
    """
    경고 ON 시 화면 상단에 빨강 배너와 텍스트를 그린다 (in-place).

    배너 높이: config.LDW["banner_height"] (px).
    표시 텍스트: "LANE DEPARTURE → LEFT" 또는 "LANE DEPARTURE → RIGHT".

    NOTE: DANGER 상태에서는 draw_danger_banner()가 이 배너를 대체한다.
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


def draw_caution_strip(frame: np.ndarray) -> None:
    """
    CAUTION 상태: 상단 황색 좁은 띠 + "차선 근접 / LANE PROXIMITY" 텍스트 (in-place).
    """
    H, W = frame.shape[:2]
    sh = config.LDW["caution_strip_h"]
    bg_color = config.LDW["caution_strip_color"]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, sh), bg_color, -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    text = "  LANE PROXIMITY  /  CAUTION  "
    font      = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.75
    thickness  = 2

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = (W - tw) // 2
    ty = (sh + th) // 2

    cv2.putText(frame, text, (tx, ty), font, font_scale, _BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty), font, font_scale, _WHITE, thickness,     cv2.LINE_AA)


def draw_danger_border(frame: np.ndarray, frame_no: int) -> None:
    """
    DANGER 상태: 프레임 전체 두꺼운 적색 테두리 + 홀짝 프레임으로 강도 교체 (플래시).

    frame_no 짝수 → 밝은 순수 적색
    frame_no 홀수 → 어두운 적색 (약 60% 밝기)
    테두리 두께: config.LDW["border_px"] (두꺼운 직사각형은 중심선이 좌표 안쪽으로 들어감).
    """
    H, W = frame.shape[:2]
    bp = config.LDW["border_px"]
    base_color = config.LDW["danger_border_color"]  # BGR

    # 플래시: 홀수 프레임은 밝기 낮춤 (BGR * 0.6)
    if frame_no % 2 == 0:
        color = base_color
    else:
        color = tuple(int(c * 0.55) for c in base_color)  # type: ignore[assignment]

    # 테두리를 frame 안쪽에 그려야 클리핑 없이 전체 표시됨
    inset = bp // 2
    cv2.rectangle(frame,
                  (inset, inset),
                  (W - inset - 1, H - inset - 1),
                  color, bp)


def draw_danger_banner(frame: np.ndarray, side: str, frame_no: int) -> None:
    """
    DANGER 상태: 화면 상단 대형 배너 "⚠ 차선 이탈 / LANE DEPARTURE" (in-place).

    플래시와 동기화: 짝수 프레임 → 진한 배경, 홀수 → 약간 밝은 배경.
    """
    H, W = frame.shape[:2]
    bh = config.LDW["danger_banner_h"]
    bg = config.LDW["danger_banner_bg"]

    alpha = 0.85 if frame_no % 2 == 0 else 0.65
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, bh), bg, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 한글+영어 2행: ⚠ + 이탈 방향
    line1 = f"!  {side} LANE DEPARTURE  !"
    line2 = f"[  WARNING  --  {side}  ]"
    font      = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.05
    thickness  = 2

    for i, text in enumerate([line1, line2]):
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        tx = (W - tw) // 2
        # 두 줄 세로 배치: 상단 여백 약간 두고 줄간 th+4
        ty = (bh // 2) - th // 2 + i * (th + 5) - (th // 2)
        cv2.putText(frame, text, (tx, ty), font, font_scale, _BLACK, thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, ty), font, font_scale, _WHITE, thickness,     cv2.LINE_AA)


def draw_drift_arrow(frame: np.ndarray, side: str) -> None:
    """
    DANGER 상태: 이탈 방향을 가리키는 큰 화살표를 배너 아래에 그린다 (in-place).

    side="LEFT"  → ← 화살표 (왼쪽)
    side="RIGHT" → → 화살표 (오른쪽)
    """
    H, W = frame.shape[:2]
    bh = config.LDW["danger_banner_h"]

    # 화살표 위치: 배너 바로 아래 중앙에서 약간 옆
    cy = bh + 30  # 배너 아래 30px
    cx = W // 2

    arrow_len = 60
    arrow_color = (0, 0, 255)  # 적색 BGR
    tip_size    = 0.5

    if side == "LEFT":
        start = (cx + arrow_len, cy)
        end   = (cx - arrow_len, cy)
    else:
        start = (cx - arrow_len, cy)
        end   = (cx + arrow_len, cy)

    cv2.arrowedLine(frame, start, end, _BLACK, 9, cv2.LINE_AA, tipLength=tip_size)
    cv2.arrowedLine(frame, start, end, arrow_color, 5, cv2.LINE_AA, tipLength=tip_size)


def draw_offset_gauge(
    frame: np.ndarray,
    off: float | None,
    lane_st: str,
) -> None:
    """
    화면 하단 오프셋 게이지 바 (공통, 모든 상태에서 표시).

    게이지:
      ─ 회색 배경 바 (전체 폭에서 좌우 gauge_h_margin 안쪽)
      ─ 중앙 흰색 틱
      ─ 차량 위치 마커 (원) — SAFE=흰, CAUTION=황, DANGER=적
      ─ 수치 오프셋 텍스트
    """
    H, W = frame.shape[:2]
    gh = config.LDW["gauge_height_px"]
    gm = config.LDW["gauge_h_margin"]
    bm = config.LDW["gauge_b_margin"]
    mr = config.LDW["gauge_marker_r"]

    # HUD 텍스트 영역 위 배치 (HUD 최대 6줄: frame/offset/WARN/MODE/Radius/DEMO)
    hud_lines = 6  # demo+CURVE 모드 최대 행 수 (안전 상한)
    hud_height = hud_lines * 24 + 8
    gauge_y2 = H - hud_height - bm
    gauge_y1 = gauge_y2 - gh

    x_left  = gm
    x_right = W - gm

    # 배경 바 (반투명 어두운 회색)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_left, gauge_y1), (x_right, gauge_y2), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # 외곽선
    cv2.rectangle(frame, (x_left, gauge_y1), (x_right, gauge_y2), (120, 120, 120), 1)

    # 중앙 흰색 틱
    cx = (x_left + x_right) // 2
    mid_y = (gauge_y1 + gauge_y2) // 2
    cv2.line(frame, (cx, gauge_y1 - 3), (cx, gauge_y2 + 3), _WHITE, 2)

    # 차량 위치 마커
    if off is not None:
        # off 범위: 대략 [-1, 1] → 게이지 폭에 매핑
        bar_w = x_right - x_left
        clamp_off = max(-1.0, min(1.0, off))
        # off 부호: 음수=LEFT, 양수=RIGHT. 게이지는 왼쪽 음수/오른쪽 양수.
        # off=0 → 게이지 중앙(cx)
        # off 부호: 양수=RIGHT 이탈(차량이 오른쪽으로 흘러감) → 마커를 오른쪽으로
        mx = cx + int(clamp_off * (bar_w / 2))
        mx = max(x_left + mr, min(x_right - mr, mx))

        if lane_st == "DANGER":
            marker_color = (0, 0, 255)
        elif lane_st == "CAUTION":
            marker_color = (0, 200, 220)
        else:
            marker_color = _WHITE

        cv2.circle(frame, (mx, mid_y), mr, _BLACK, -1)
        cv2.circle(frame, (mx, mid_y), mr - 2, marker_color, -1)

        # 수치 텍스트 (마커 위)
        off_str = f"{off:+.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs   = 0.45
        th   = 1
        (tw, tth), _ = cv2.getTextSize(off_str, font, fs, th)
        tx = max(x_left, min(mx - tw // 2, x_right - tw))
        ty = gauge_y1 - 5
        cv2.putText(frame, off_str, (tx, ty), font, fs, _BLACK,  th + 1, cv2.LINE_AA)
        cv2.putText(frame, off_str, (tx, ty), font, fs, marker_color, th, cv2.LINE_AA)
    else:
        # offset N/A
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "N/A", (cx - 15, mid_y + 5), font, 0.45, (160, 160, 160), 1, cv2.LINE_AA)


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
    curve_mode: str | None = None,
    curvature_r: float | None = None,
    demo: bool = False,
) -> None:
    """
    화면 좌하단 HUD: 오프셋 수치 + 경고 상태 + 프레임 정보.

    표시 항목:
      - frame N/total
      - L:<state_abbr>  R:<state_abbr>  segs:<n>
      - offset: -0.12  (또는 N/A)
      - WARN: OFF | LEFT | RIGHT
      - MODE: CURVE  (또는 STRAIGHT(fallback))  ← M6 추가, None이면 미표시
      - Radius: 1234px  ← curve_mode="CURVE" 일 때만 표시
      - [LDW DEMO — drift simulated]  ← demo=True 일 때 추가 행 (황색 강조)

    curve_mode: "CURVE" | "STRAIGHT(fallback)" | None (None이면 M6 HUD 숨김)
    curvature_r: 곡률 반경 (픽셀), curve_mode="CURVE"일 때만 의미 있음
    demo: True이면 하단 "LDW DEMO — drift simulated" 태그 표시 (정직성 표시)
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

    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"frame {frame_no}/{total_frames}  "
         f"L:{_abbr.get(left_state, left_state)}  "
         f"R:{_abbr.get(right_state, right_state)}  "
         f"segs:{n_segs}",
         _WHITE),
        (f"offset: {off_str}", _YELLOW_HUD),
        (f"WARN: {warn_str}", warn_color),
    ]

    # M6 추가 HUD 행 (curve_mode가 None이 아닐 때만)
    if curve_mode is not None:
        mode_color = _YELLOW_HUD if curve_mode == "CURVE" else _WHITE
        lines.append((f"MODE: {curve_mode}", mode_color))
        if curve_mode == "CURVE" and curvature_r is not None:
            r_str = f"{curvature_r:.0f}px" if curvature_r < 9_000_000 else "straight"
            lines.append((f"Radius: {r_str}", _YELLOW_HUD))

    # demo 태그 (정직성)
    if demo:
        lines.append(("LDW DEMO  --  drift simulated", (0, 200, 220)))

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


def draw_curved_area(
    frame: np.ndarray,
    polygon: np.ndarray,
    left_pts: np.ndarray,
    right_pts: np.ndarray,
    lane_st: str = "SAFE",
) -> None:
    """
    곡선 주행 가능 영역을 반투명 폴리곤으로 채우고 양쪽 차선 곡선을 그린다 (in-place).

    Args:
        frame    : 출력 프레임 (BGR, in-place 수정).
        polygon  : curve.curved_lane_points()의 unwarped 폴리곤 (shape: (2N, 2) int32).
                   fillPoly용. 좌측 위→아래 + 우측 아래→위 순서.
        left_pts : 왼쪽 차선 곡선 포인트 (shape: (N, 2) int32). polylines용.
        right_pts: 오른쪽 차선 곡선 포인트 (shape: (N, 2) int32). polylines용.
        lane_st  : "SAFE"(초록) | "CAUTION"(황색) | "DANGER"(적색).

    fill_alpha: config.LDW["fill_alpha"] (직선 오버레이와 동일 설정).
    """
    alpha = config.LDW["fill_alpha"]

    if lane_st == "DANGER":
        fill_color = config.LDW["danger_fill"]
    elif lane_st == "CAUTION":
        fill_color = config.LDW["caution_fill"]
    else:
        fill_color = _GREEN_FILL

    # 반투명 폴리곤 (drivable area)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon.reshape((-1, 1, 2))], fill_color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 차선 곡선 (실선, 초록)
    cv2.polylines(frame,
                  [left_pts.reshape((-1, 1, 2))],
                  isClosed=False,
                  color=_GREEN_LINE,
                  thickness=_LINE_THICKNESS,
                  lineType=cv2.LINE_AA)
    cv2.polylines(frame,
                  [right_pts.reshape((-1, 1, 2))],
                  isClosed=False,
                  color=_GREEN_LINE,
                  thickness=_LINE_THICKNESS,
                  lineType=cv2.LINE_AA)


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
    draw_area: bool = True,
    draw_lines: bool = True,
    curve_mode: str | None = None,
    curvature_r: float | None = None,
    lane_st: str = "SAFE",
    demo: bool = False,
) -> None:
    """
    한 프레임에 모든 오버레이를 합성한다 (in-place).

    순서:
      1. 주행영역 반투명 폴리곤 (draw_area=True이고 양쪽 차선이 있을 때만, lane_st 색상)
      2. 차선선 (draw_lines=True일 때만; DANGER 방향은 빨강+두꺼움)
      3. 상태별 배너/띠 (SAFE=없음, CAUTION=황색 띠, DANGER=플래시 테두리+대형 배너+화살표)
      4. 오프셋 게이지 (모든 상태)
      5. HUD

    M6 파라미터 (기본값=None → 기존 동작 유지):
      draw_area    : False이면 직선 주행영역 폴리곤 스킵 (곡선 모드에서 사용).
      draw_lines   : False이면 직선 차선선 스킵 (곡선 모드에서 사용).
      curve_mode   : "CURVE" | "STRAIGHT(fallback)" | None. HUD에 표시.
      curvature_r  : 곡률 반경 (픽셀). curve_mode="CURVE"일 때 HUD에 표시.

    3-state 파라미터:
      lane_st : "SAFE" | "CAUTION" | "DANGER" — 색상 및 추가 요소 선택.
      demo    : True이면 HUD에 "LDW DEMO -- drift simulated" 태그 표시.
    """
    # 1. 주행영역 폴리곤 (STRAIGHT 모드 또는 fallback 모드에서만)
    if draw_area and left_fit is not None and right_fit is not None:
        draw_drivable_area(frame, left_fit, right_fit, y_bottom, y_top, lane_st)

    # 2. 차선선 (STRAIGHT 모드 또는 fallback 모드에서만)
    if draw_lines:
        draw_lane_lines_ldw(frame, left_fit, right_fit, y_bottom, y_top, warning, side)

    # 3. 상태별 배너/테두리/화살표
    if lane_st == "DANGER":
        # DANGER: 플래시 테두리 (최하위 레이어로 먼저, 이후 배너가 위에)
        draw_danger_border(frame, frame_no)
        # 대형 배너 (기존 draw_ldw_banner 대체)
        if side is not None:
            draw_danger_banner(frame, side, frame_no)
            # 방향 화살표
            draw_drift_arrow(frame, side)
    elif lane_st == "CAUTION":
        draw_caution_strip(frame)
    # SAFE: 추가 배너 없음 (기존과 동일 — 초록 영역만)

    # 4. 오프셋 게이지 (공통)
    draw_offset_gauge(frame, off, lane_st)

    # 5. HUD (curve_mode/curvature_r 전달)
    draw_hud(frame, off, warning, side,
             frame_no, total_frames, n_segs, left_state, right_state,
             curve_mode=curve_mode, curvature_r=curvature_r,
             demo=demo)
