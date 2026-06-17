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
from src import departure as _dep
from src import textkr as _tkr  # 한글 텍스트 렌더링 (Pillow 기반)

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
    if lane_st == "DANGER":
        fill_color = config.LDW["danger_fill"]
        alpha = config.LDW["fill_alpha"]
    elif lane_st == "CAUTION":
        fill_color = config.LDW["caution_fill"]
        alpha = config.LDW["caution_fill_alpha"]  # CAUTION은 더 선명한 별도 알파값 사용
    else:
        fill_color = _GREEN_FILL
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
    표시 텍스트: "차선 이탈 경고 → 왼쪽" 또는 "차선 이탈 경고 → 오른쪽" (한글).

    NOTE: DANGER 상태에서는 draw_danger_banner()가 이 배너를 대체한다.
    """
    H, W = frame.shape[:2]
    bh = config.LDW["banner_height"]

    # 배너 배경 (반투명) — 불투명도: config.LDW["banner_alpha"]
    banner_alpha = config.LDW["banner_alpha"]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, bh), _RED_BANNER, -1)
    cv2.addWeighted(overlay, banner_alpha, frame, 1 - banner_alpha, 0, frame)

    side_kr = "왼쪽" if side == "LEFT" else "오른쪽"
    text = f"차선 이탈 경고  →  {side_kr}"
    fs = config.KOREAN_FONT_SIZE_BANNER
    font = _tkr.get_font(fs)
    # 텍스트 너비 측정 (PIL)
    from PIL import Image as _PILImg, ImageDraw as _PILDraw
    _tmp = _PILDraw.Draw(_PILImg.new("RGB", (1, 1)))
    bbox = _tmp.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = max(0, (W - tw) // 2)
    ty = max(0, (bh - th) // 2)
    _tkr.put_kr(frame, text, (tx, ty), fs, _WHITE, outline_color=_BLACK, outline_px=2)


def draw_caution_strip(frame: np.ndarray) -> None:
    """
    CAUTION 상태: 상단 황색 띠 + "차선 근접 · 주의" 텍스트 (in-place, 한글).

    caution_strip_alpha: config.LDW["caution_strip_alpha"] — 0.90으로 선명하게.
    """
    H, W = frame.shape[:2]
    sh = config.LDW["caution_strip_h"]
    bg_color = config.LDW["caution_strip_color"]
    strip_alpha = config.LDW["caution_strip_alpha"]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, sh), bg_color, -1)
    cv2.addWeighted(overlay, strip_alpha, frame, 1.0 - strip_alpha, 0, frame)

    text = "차선 근접 · 주의"
    fs = config.KOREAN_FONT_SIZE_BANNER
    font = _tkr.get_font(fs)
    from PIL import Image as _PILImg, ImageDraw as _PILDraw
    _tmp = _PILDraw.Draw(_PILImg.new("RGB", (1, 1)))
    bbox = _tmp.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = max(0, (W - tw) // 2)
    ty = max(0, (sh - th) // 2)
    _tkr.put_kr(frame, text, (tx, ty), fs, _BLACK, outline_color=None)


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
    DANGER 상태: 화면 상단 대형 배너 "차선 이탈 경고 / 왼쪽|오른쪽" (in-place, 한글).

    플래시와 동기화: 짝수 프레임 → 진한 배경, 홀수 → 약간 밝은 배경.
    """
    H, W = frame.shape[:2]
    bh = config.LDW["danger_banner_h"]
    bg = config.LDW["danger_banner_bg"]

    alpha = 0.85 if frame_no % 2 == 0 else 0.65
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, bh), bg, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    side_kr = "왼쪽" if side == "LEFT" else "오른쪽"
    # 한글 2행: 이탈 방향 + 경고
    lines_kr = [
        f"차선 이탈 경고 — {side_kr}",
        "[ 경고 ]",
    ]
    fs = config.KOREAN_FONT_SIZE_BANNER
    font = _tkr.get_font(fs)
    from PIL import Image as _PILImg, ImageDraw as _PILDraw
    _tmp = _PILDraw.Draw(_PILImg.new("RGB", (1, 1)))
    line_h = fs + 4
    total_h = len(lines_kr) * line_h
    y_start = max(0, (bh - total_h) // 2)
    for i, text in enumerate(lines_kr):
        bbox = _tmp.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        tx = max(0, (W - tw) // 2)
        ty = y_start + i * line_h
        _tkr.put_kr(frame, text, (tx, ty), fs, _WHITE, outline_color=_BLACK, outline_px=2)


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

    # HUD 텍스트 영역 위 배치 (HUD 최대 7줄: frame/offset/wheel->line/WARN/MODE/Radius/DEMO)
    hud_lines = 7  # demo+CURVE 모드 최대 행 수 (안전 상한)
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

        # 수치 텍스트 (마커 위): 바퀴–차선 거리(m) + 근사값 노트 (한글)
        wtl = _dep.wheel_to_line_m(off)
        lat_m = _dep.lateral_offset_m(off)
        wtl_str  = f"바퀴–차선: {wtl:.2f} m  ※ 근사값(미보정)"
        lat_str  = f"오프셋: {lat_m:+.2f} m"
        fs_small = config.KOREAN_FONT_SIZE_SMALL
        from PIL import Image as _PILImg2, ImageDraw as _PILDraw2
        font_small = _tkr.get_font(fs_small)
        _tmp2 = _PILDraw2.Draw(_PILImg2.new("RGB", (1, 1)))
        # wtl_str 행 (마커 위)
        bbox_w = _tmp2.textbbox((0, 0), wtl_str, font=font_small)
        tw = bbox_w[2] - bbox_w[0]
        tth = bbox_w[3] - bbox_w[1]
        tx = max(x_left, min(mx - tw // 2, x_right - tw))
        ty = gauge_y1 - tth - 5
        _tkr.put_kr(frame, wtl_str, (tx, ty), fs_small, marker_color, outline_color=_BLACK)
        # lat_str 행 (wtl 위)
        bbox_l = _tmp2.textbbox((0, 0), lat_str, font=font_small)
        tw2 = bbox_l[2] - bbox_l[0]
        th2 = bbox_l[3] - bbox_l[1]
        tx2 = max(x_left, min(mx - tw2 // 2, x_right - tw2))
        ty2 = ty - th2 - 3
        _tkr.put_kr(frame, lat_str, (tx2, ty2), fs_small, marker_color, outline_color=_BLACK)
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
        "raw_detected":        "검출",
        "rejected_as_outlier": "이상치",
        "held_from_previous":  "유지",
        "consecutive_missing": "미검출",
    }

    off_str = f"{off:+.3f}" if off is not None else "N/A"
    side_kr = ("왼쪽" if side == "LEFT" else "오른쪽") if side else None
    warn_str = f"{side_kr}" if (warning and side_kr) else "없음"
    warn_color = _RED_LINE if warning else _GREEN_LINE

    # wheel→line 거리 및 측방향 오프셋(m) HUD 표시
    if off is not None:
        wtl    = _dep.wheel_to_line_m(off)
        lat_m  = _dep.lateral_offset_m(off)  # 부호 포함 미터 (음수=LEFT, 양수=RIGHT)
        wtl_str = f"{wtl:.2f} m"
        lat_str = f"{lat_m:+.2f} m"
    else:
        wtl_str = "N/A"
        lat_str = "N/A"

    # HUD 항목 (한글 레이블 + 영숫자 수치)
    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"프레임 {frame_no}/{total_frames}  "
         f"좌:{_abbr.get(left_state, left_state)}  "
         f"우:{_abbr.get(right_state, right_state)}  "
         f"선분:{n_segs}",
         _WHITE),
        (f"오프셋: {off_str}  횡방향: {lat_str}", _YELLOW_HUD),
        (f"바퀴–차선: {wtl_str}  ※ 근사값(미보정)", _YELLOW_HUD),
        (f"경고: {warn_str}", warn_color),
    ]

    # M6 추가 HUD 행 (curve_mode가 None이 아닐 때만)
    if curve_mode is not None:
        mode_kr = "곡선" if curve_mode == "CURVE" else "직선(폴백)"
        mode_color = _YELLOW_HUD if curve_mode == "CURVE" else _WHITE
        lines.append((f"모드: {mode_kr}", mode_color))
        if curve_mode == "CURVE" and curvature_r is not None:
            r_str = f"{curvature_r:.0f}px" if curvature_r < 9_000_000 else "직선"
            lines.append((f"곡률반경: {r_str}", _YELLOW_HUD))

    # demo 태그 (정직성)
    if demo:
        lines.append(("LDW DEMO  --  드리프트 시뮬레이션", (0, 200, 220)))

    fs_hud = config.KOREAN_FONT_SIZE_SMALL
    line_h = fs_hud + 6
    x0     = 10
    y0     = H - line_h * len(lines) - 8

    for i, (text, color) in enumerate(lines):
        y = y0 + i * line_h
        _tkr.put_kr(frame, text, (x0, y), fs_hud, color, outline_color=_BLACK, outline_px=1)


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
    if lane_st == "DANGER":
        fill_color = config.LDW["danger_fill"]
        alpha = config.LDW["fill_alpha"]
    elif lane_st == "CAUTION":
        fill_color = config.LDW["caution_fill"]
        alpha = config.LDW["caution_fill_alpha"]  # CAUTION은 더 선명한 별도 알파값 사용
    else:
        fill_color = _GREEN_FILL
        alpha = config.LDW["fill_alpha"]

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

    # "버드아이(위에서 본 도로)" 레이블 (패널 좌하단 안쪽, 한글)
    label  = "버드아이(위에서 본 도로)"
    fs_pip = config.KOREAN_FONT_SIZE_SMALL
    from PIL import Image as _PILImg3, ImageDraw as _PILDraw3
    font_pip = _tkr.get_font(fs_pip)
    _tmp3 = _PILDraw3.Draw(_PILImg3.new("RGB", (1, 1)))
    bbox_pip = _tmp3.textbbox((0, 0), label, font=font_pip)
    th_pip = bbox_pip[3] - bbox_pip[1]
    lx = x1 + 4
    ly = y2 - th_pip - 4
    _tkr.put_kr(frame, label, (lx, ly), fs_pip,
                pip_cfg["label_color"], outline_color=_BLACK, outline_px=1)


def draw_legend(frame: np.ndarray) -> None:
    """
    화면 좌상단에 반투명 설명 범례 패널을 그린다 (in-place).

    config.SHOW_LEGEND=True 일 때만 render_frame()에서 호출됨.
    내용: 화면 오버레이 요소의 한글 설명 목록.
    위치: 좌상단 (우상단=버드아이, 하단=HUD+게이지와 겹치지 않게).
    """
    legend_items = [
        "[ 범례 ]",
        "초록 영역 = 주행 가능 차로",
        "노랑 = 차선 근접 주의",
        "빨강 = 이탈 위험",
        "게이지 = 차로 내 차량 위치",
        "우측상단 = 버드아이 뷰",
    ]
    fs = config.KOREAN_FONT_SIZE_LEGEND
    line_h = fs + 5
    pad = 8
    panel_h = len(legend_items) * line_h + pad * 2
    panel_w = 240  # 고정 폭 (한글 텍스트 최대 길이 기준)

    # 배경 반투명 박스 (좌상단, 배너 아래 약간 띄움)
    bh = config.LDW["banner_height"]
    top_y = bh + 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (4, top_y), (4 + panel_w, top_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (4, top_y), (4 + panel_w, top_y + panel_h), (160, 160, 160), 1)

    for i, item in enumerate(legend_items):
        color = (0, 220, 255) if i == 0 else _WHITE  # 첫 행(헤더)은 황색
        x = 4 + pad
        y = top_y + pad + i * line_h
        _tkr.put_kr(frame, item, (x, y), fs, color, outline_color=_BLACK, outline_px=1)


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

    # 6. 설명 범례 패널 (config.SHOW_LEGEND=True 시에만, 좌상단)
    if config.SHOW_LEGEND:
        draw_legend(frame)


# ── 차량 추적 오버레이 (L10 데모) ────────────────────────────────────────────────

_TRACK_OK_COLOR   = (0, 255, 0)     # 초록 — 추적 성공
_TRACK_LOST_COLOR = (0, 0, 255)     # 적색 — LOST
_TRACK_TRAIL_COLOR = (0, 200, 100)  # 옅은 초록 — 궤적 점
_TRACK_BOX_THICK  = 3
_TRACK_TRAIL_R    = 3               # 궤적 점 반경 (px)
_TRACK_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_TRACK_FONT_SCALE = 0.65
_TRACK_FONT_THICK = 2


def draw_track(
    frame: np.ndarray,
    box: tuple[int, int, int, int] | None,
    ok: bool,
    trail: list[tuple[int, int]],
) -> None:
    """
    CSRT 추적 결과를 프레임에 in-place로 그린다.

    Args:
        frame : BGR 프레임.
        box   : (x, y, w, h) 추적 박스. ok=False이면 None 허용.
        ok    : True=추적 성공(초록), False=LOST(적색).
        trail : 과거 중심 좌표 목록 [(cx, cy), ...]. 최신이 마지막.
                config.VEHICLE["track_trail_len"] 개로 제한.

    시각 요소:
      - 박스: ok=True→초록, False→적색(점선 느낌의 얇은 박스).
      - 라벨: "TRACKING (CSRT)" 또는 "LOST".
      - 궤적: 최근 trail 점을 작은 원으로 표시.
    """
    from src import config as _cfg  # 지연 import로 순환 방지

    # 1) 궤적 점 (trail) — 박스·라벨보다 먼저 그려 레이어 순서 유지
    for cx, cy in trail:
        cv2.circle(frame, (cx, cy), _TRACK_TRAIL_R, _TRACK_TRAIL_COLOR, -1)

    # 2) 박스 + 라벨
    # 기본값 — box가 None인 경우 또는 ok=False에도 좌표 정의
    x, ly = 20, 80

    if ok and box is not None:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), _TRACK_OK_COLOR, _TRACK_BOX_THICK)
        label = "차량 추적 중"
        color = _TRACK_OK_COLOR
        # 라벨: 박스 위에 표시 (화면 상단 밖으로 나가지 않도록 클램핑)
        ly = max(y - 22, 2)
    else:
        label = "추적 놓침"
        color = _TRACK_LOST_COLOR
        # LOST 시: 마지막 알려진 박스가 있으면 그 위치에 적색 박스
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), _TRACK_LOST_COLOR, 1)
            ly = max(y - 22, 2)
        # box=None: 기본값 (x=20, ly=80) 사용

    # 한글 텍스트 (PIL 렌더링)
    fs_tr = config.KOREAN_FONT_SIZE_NORMAL
    _tkr.put_kr(frame, label, (x, ly), fs_tr, color, outline_color=_BLACK, outline_px=2)

    # 3) 우하단 상태 태그 (한글)
    H, W = frame.shape[:2]
    tag_ok = "OK" if ok else "놓침"
    tag = f"L10 추적  {tag_ok}"
    from PIL import Image as _PILImg4, ImageDraw as _PILDraw4
    font_tag = _tkr.get_font(fs_tr)
    _tmp4 = _PILDraw4.Draw(_PILImg4.new("RGB", (1, 1)))
    bbox_tag = _tmp4.textbbox((0, 0), tag, font=font_tag)
    tw = bbox_tag[2] - bbox_tag[0]
    tx = W - tw - 10
    ty = H - fs_tr - 10
    _tkr.put_kr(frame, tag, (tx, ty), fs_tr, color, outline_color=_BLACK, outline_px=1)
