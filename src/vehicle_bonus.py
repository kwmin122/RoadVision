"""
M7 보너스 — 전방 차량 후보 시각화 실험 (vehicle_bonus.py).

이 모듈은 "차량 검출기"가 아니라 classical CV 한계를 보여주는 실험이다.
Haar cascade + 전방 ROI 제한으로 차량 후보 박스를 시각화하고,
오검(FP) / 미검(missed) 사례를 정직하게 드러내어
"이 한계가 왜 DL(YOLO)이 필요한가"의 동기로 연결한다.

공개 API:
  detect_candidates(frame, clip_key) -> list[(x,y,w,h)]
      전방 ROI 내 Haar cascade 후보 박스를 전체 프레임 좌표로 반환.
  draw_candidates(frame, boxes) -> None
      박스를 "vehicle?"(물음표 포함) 라벨로 in-place 그린다.
  compute_flow_magnitude(prev_gray, curr_gray, roi_rect) -> np.ndarray | None
      전방 ROI 영역에서 Farneback 광류 크기를 계산한다.
  draw_flow_heatmap(frame, flow_mag, roi_rect) -> None
      광류 크기 히트맵을 전방 ROI에 반투명으로 합성한다.

설계 규칙:
  - cascade는 모듈 로드 시 1회만 초기화 (프레임당 재읽기 금지).
  - 모든 임계/상수는 config.VEHICLE에서 가져온다. 함수 내 magic number 없음.
  - forward_roi_ratio는 비율(0~1)로 받아 실행 시 픽셀 좌표로 변환한다.
    클립별 분기 없음 — 비율 적용이 유일한 ROI 계산 경로.
  - draw_candidates / draw_flow_heatmap은 lane pipeline 변수를 건드리지 않는다.
"""
from __future__ import annotations

import cv2
import numpy as np

from src import config

# ── 모듈 레벨 cascade 초기화 (프레임당 재로드 방지) ───────────────────────────
_cascade_path: str = config.VEHICLE["cascade_path"]  # "models/cars.xml"
_cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(_cascade_path)

if _cascade.empty():
    raise RuntimeError(
        f"Haar cascade 로드 실패: {_cascade_path}\n"
        "models/cars.xml 파일이 올바른 위치에 있는지 확인하세요."
    )

# ── 드로잉 상수 ────────────────────────────────────────────────────────────────
_BOX_COLOR   = (0, 255, 255)   # 청록 (BGR) — lane green과 구분
_BOX_THICK   = 2
_LABEL       = "vehicle?"      # 물음표 필수 — 후보이지 확정 아님
_FONT        = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE  = 0.55
_FONT_THICK  = 2
_BLACK       = (0, 0, 0)

_HUD_TAG     = "VEHICLE EXP"   # HUD 오버레이 태그
_HUD_COLOR   = (0, 255, 255)
_HUD_FONT_SCALE = 0.55
_HUD_THICK   = 2

_FLOW_ALPHA  = 0.45            # 광류 히트맵 불투명도


# ── 내부 헬퍼 ──────────────────────────────────────────────────────────────────

def _forward_roi_rect(W: int, H: int) -> tuple[int, int, int, int]:
    """
    config.VEHICLE["forward_roi_ratio"] = [(rx1, ry1), (rx2, ry2)] (비율)에서
    전체 프레임 픽셀 좌표 (x1, y1, x2, y2)를 계산한다.

    forward_roi_ratio 형식:
      [(좌상단 x비율, 좌상단 y비율), (우하단 x비율, 우하단 y비율)]
    """
    ratio = config.VEHICLE["forward_roi_ratio"]  # [(rx1,ry1),(rx2,ry2)]
    (rx1, ry1), (rx2, ry2) = ratio
    x1 = int(rx1 * W)
    y1 = int(ry1 * H)
    x2 = int(rx2 * W)
    y2 = int(ry2 * H)
    return x1, y1, x2, y2


# ── 공개 API ───────────────────────────────────────────────────────────────────

def detect_candidates(frame: np.ndarray, clip_key: str) -> list[tuple[int, int, int, int]]:
    """
    전방 ROI 내에서 Haar cascade로 차량 후보 박스를 검출한다.

    Args:
        frame    : 원본 BGR 프레임.
        clip_key : 클립 이름 (현재 미사용, 해상도는 frame에서 직접 읽음).

    Returns:
        list of (x, y, w, h) in **전체 프레임 좌표**.
        빈 리스트 = 해당 프레임 후보 없음.

    구현 메모:
      - forward ROI를 crop → grayscale → detectMultiScale.
      - 반환 box의 x,y에 roi_x1, roi_y1을 더해 전체 프레임 좌표로 변환.
      - config.VEHICLE의 scale_factor / min_neighbors / min_size 사용.
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = _forward_roi_rect(W, H)

    # ROI crop
    roi_img = frame[y1:y2, x1:x2]
    if roi_img.size == 0:
        return []

    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    # Haar detectMultiScale — 모든 파라미터는 config.VEHICLE에서 읽음
    sf          = config.VEHICLE["scale_factor"]
    min_nbrs    = config.VEHICLE["min_neighbors"]
    min_sz      = config.VEHICLE["min_size"]    # (w, h) tuple

    raw = _cascade.detectMultiScale(
        gray,
        scaleFactor=sf,
        minNeighbors=min_nbrs,
        minSize=min_sz,
    )

    if raw is None or not hasattr(raw, '__len__') or len(raw) == 0:
        return []

    # ROI 상대 좌표 → 전체 프레임 좌표 변환
    boxes: list[tuple[int, int, int, int]] = []
    for (bx, by, bw, bh) in raw:
        boxes.append((int(x1 + bx), int(y1 + by), int(bw), int(bh)))
    return boxes


def draw_candidates(frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> None:
    """
    후보 박스를 "vehicle?" 라벨과 함께 프레임에 in-place로 그린다.

    Args:
        frame : BGR 프레임 (in-place 수정).
        boxes : detect_candidates()가 반환한 전체 프레임 좌표 박스 목록.
    """
    H, W = frame.shape[:2]

    for (x, y, w, h) in boxes:
        # 박스
        cv2.rectangle(frame, (x, y), (x + w, y + h), _BOX_COLOR, _BOX_THICK)

        # 레이블 — 박스 위에 표시, 화면 밖으로 나가지 않도록 클램핑
        lx = x
        ly = max(y - 6, 14)   # 상단 여백 확보
        # 검정 외곽선 + 청록 텍스트 (가독성)
        cv2.putText(frame, _LABEL, (lx, ly),
                    _FONT, _FONT_SCALE, _BLACK, _FONT_THICK + 1, cv2.LINE_AA)
        cv2.putText(frame, _LABEL, (lx, ly),
                    _FONT, _FONT_SCALE, _BOX_COLOR, _FONT_THICK, cv2.LINE_AA)


def draw_vehicle_hud_tag(frame: np.ndarray, n_boxes: int) -> None:
    """
    프레임 우하단에 "VEHICLE EXP" HUD 태그와 후보 수를 표시한다 (in-place).

    lane pipeline의 HUD와 겹치지 않도록 우하단에 배치한다.
    """
    H, W = frame.shape[:2]
    tag = f"{_HUD_TAG}  candidates:{n_boxes}"
    (tw, _th), _ = cv2.getTextSize(tag, _FONT, _HUD_FONT_SCALE, _HUD_THICK)
    x = W - tw - 10
    y = H - 10
    cv2.putText(frame, tag, (x, y),
                _FONT, _HUD_FONT_SCALE, _BLACK, _HUD_THICK + 1, cv2.LINE_AA)
    cv2.putText(frame, tag, (x, y),
                _FONT, _HUD_FONT_SCALE, _HUD_COLOR, _HUD_THICK, cv2.LINE_AA)


def draw_roi_boundary(frame: np.ndarray) -> None:
    """
    전방 ROI 경계를 점선 느낌의 얇은 노란 사각형으로 표시한다 (in-place).
    어느 영역을 검색했는지 투명성 확보 목적.
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = _forward_roi_rect(W, H)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 200), 1)


def compute_flow_magnitude(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    frame_shape: tuple[int, int],
) -> np.ndarray | None:
    """
    전방 ROI 영역에서 Farneback 광류 크기(magnitude)를 계산한다.

    Args:
        prev_gray    : 직전 프레임 grayscale (전체 프레임 크기).
        curr_gray    : 현재 프레임 grayscale (전체 프레임 크기).
        frame_shape  : (H, W) 원본 프레임 크기.

    Returns:
        ROI 크기의 광류 크기 배열 (float32, shape=(roi_h, roi_w)).
        None: 이전 프레임이 없거나 ROI가 비었을 때.
    """
    if prev_gray is None or curr_gray is None:
        return None

    H, W = frame_shape
    x1, y1, x2, y2 = _forward_roi_rect(W, H)

    prev_roi = prev_gray[y1:y2, x1:x2]
    curr_roi = curr_gray[y1:y2, x1:x2]

    if prev_roi.size == 0 or curr_roi.size == 0:
        return None

    flow = cv2.calcOpticalFlowFarneback(
        prev_roi, curr_roi,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag


def draw_flow_heatmap(
    frame: np.ndarray,
    flow_mag: np.ndarray,
    frame_shape: tuple[int, int],
) -> None:
    """
    광류 크기를 열화상 히트맵으로 전방 ROI에 반투명 합성한다 (in-place).

    Args:
        frame       : 출력 BGR 프레임.
        flow_mag    : compute_flow_magnitude()가 반환한 크기 배열.
        frame_shape : (H, W).
    """
    if flow_mag is None:
        return

    H, W = frame_shape
    x1, y1, x2, y2 = _forward_roi_rect(W, H)

    roi_h = y2 - y1
    roi_w = x2 - x1
    if roi_h <= 0 or roi_w <= 0:
        return

    # 0~255 정규화 → COLORMAP_JET 적용 (접근하는 물체 = 높은 크기 = 빨강)
    mag_norm = cv2.normalize(flow_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)

    # ROI에 반투명 합성
    overlay = frame.copy()
    overlay[y1:y2, x1:x2] = heat_bgr
    cv2.addWeighted(overlay, _FLOW_ALPHA, frame, 1 - _FLOW_ALPHA, 0, frame)

    # "FLOW" 레이블 (ROI 좌상단)
    cv2.putText(frame, "FLOW", (x1 + 4, y1 + 14),
                _FONT, 0.45, _BLACK, 2, cv2.LINE_AA)
    cv2.putText(frame, "FLOW", (x1 + 4, y1 + 14),
                _FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
