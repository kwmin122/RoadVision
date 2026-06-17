"""
Lecture 10 차량 추적 데모 — CSRT 상관 필터 추적기 (vehicle_track.py).

이 모듈은 "차량 추적기" 데모다. 핵심 설계 철학:
  - 검출(detection)이 아닌 추적(tracking)을 시연한다.
  - CSRT(Discriminative Correlation Filter with Channel and Spatial Reliability)는
    초기 seed 박스 하나만 있으면 이후 프레임에서 상관 필터로 타깃을 추적한다.
  - seed 박스는 config.VEHICLE["track_seed"]에 클립별로 고정 (데이터, 로직 아님).
  - lane 파이프라인 변수(smoother, CSV, state_counts, departure)는 절대 건드리지 않는다.

공개 API:
  create_tracker() -> cv2.Tracker
      CSRT 추적기 인스턴스 생성 (cv2 fallback 포함).
  seed_box(clip_key) -> tuple[int, frame_idx] | "auto"
      클립별 초기 박스 (x, y, w, h) + seed 프레임 인덱스(1-based) 반환.
      "auto" 반환 시 Haar 자동 탐색 시도.
  init_tracker(tracker, frame, box) -> None
      추적기 초기화 (frame에서 box를 타깃으로 등록).
  update(tracker, frame) -> tuple[bool, tuple]
      다음 프레임으로 추적기 갱신. (ok, (x,y,w,h)) 반환.
      ok=False → LOST 상태.

한계 (정직 기록):
  - 초기 seed 위치 결정이 취약점 — Haar cascade로 자동 탐색하지 못하는 경우
    클립별 하드코딩 박스(config.VEHICLE["track_seed"])로 대체.
  - 타깃이 프레임 밖으로 나가면 LOST.
  - 급격한 조명/스케일/가림 변화 시 드리프트 후 LOST 가능.
  - CSRT는 검출 없는 추적기이므로 LOST 후 재검출 불가.
"""
from __future__ import annotations

import cv2
import numpy as np

from src import config
from src import vehicle_bonus as vb


# ── 추적기 생성 ─────────────────────────────────────────────────────────────────

def create_tracker() -> cv2.Tracker:
    """
    CSRT 추적기를 생성한다. opencv-contrib 4.13 기준 cv2.TrackerCSRT_create()가
    기본 경로이며, 없으면 cv2.legacy.TrackerCSRT_create()로 fallback한다.

    Returns:
        cv2.TrackerCSRT 인스턴스.

    Raises:
        RuntimeError: CSRT가 모두 없을 경우 (opencv-contrib 미설치).
    """
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError(
        "CSRT 추적기를 찾을 수 없습니다. "
        "opencv-contrib-python을 설치하세요:\n"
        "  pip install opencv-contrib-python==4.13.0.92"
    )


# ── seed 박스 조회 ──────────────────────────────────────────────────────────────

def seed_box(clip_key: str) -> tuple[int, tuple[int, int, int, int]] | tuple[str, None]:
    """
    클립별 초기 추적 박스와 seed 프레임 인덱스(1-based)를 반환한다.

    반환 형식:
      (frame_idx_1based, (x, y, w, h))   — 하드코딩 박스가 있을 때
      ("auto", None)                      — 자동 탐색 시도 시 (현재 미사용)

    config.VEHICLE["track_seed"][clip_key]에 클립별 값이 있으면 그것을 반환한다.
    클립 이름 분기는 이 함수에서만 하며, 추적 루프에서는 분기 없음.
    """
    seeds = config.VEHICLE.get("track_seed", {})
    entry = seeds.get(clip_key)
    if entry is None:
        return ("auto", None)
    return (entry["frame"], tuple(entry["box"]))


def auto_seed(clip_key: str, clip_path: str, scan_start: int = 1, scan_frames: int = 300) -> tuple[int, tuple] | None:
    """
    Haar cascade로 seed 박스를 자동 탐색한다.

    scan_start 프레임부터 scan_frames 개까지 순서대로 읽어 전방 ROI 내
    차량 후보가 처음 나타나는 프레임+박스를 반환한다.

    Returns:
        (frame_idx_1based, (x, y, w, h)) — 발견 시
        None — 못 찾았을 때
    """
    cap = cv2.VideoCapture(clip_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, scan_start - 1)
    found = None
    for i in range(scan_frames):
        ret, frame = cap.read()
        if not ret:
            break
        boxes = vb.detect_candidates(frame, clip_key)
        if boxes:
            found = (scan_start + i, boxes[0])
            break
    cap.release()
    return found


# ── 추적기 초기화 ───────────────────────────────────────────────────────────────

def init_tracker(
    tracker: cv2.Tracker,
    frame: np.ndarray,
    box: tuple[int, int, int, int],
) -> None:
    """
    추적기를 초기화한다. frame에서 box를 타깃으로 등록.

    Args:
        tracker : create_tracker()로 생성한 CSRT 인스턴스.
        frame   : seed 프레임 (BGR).
        box     : (x, y, w, h) — 타깃 바운딩 박스 (정수, 전체 프레임 좌표).
    """
    tracker.init(frame, tuple(int(v) for v in box))


# ── 추적 갱신 ───────────────────────────────────────────────────────────────────

def update(
    tracker: cv2.Tracker,
    frame: np.ndarray,
) -> tuple[bool, tuple[int, int, int, int] | None]:
    """
    다음 프레임으로 추적기를 갱신한다.

    Args:
        tracker : 초기화된 CSRT 추적기.
        frame   : 현재 BGR 프레임.

    Returns:
        (ok, box)
          ok=True  → 추적 성공.  box=(x,y,w,h) int 튜플.
          ok=False → LOST.       box=None.
    """
    ok, raw_box = tracker.update(frame)
    if ok and raw_box is not None:
        box = tuple(int(v) for v in raw_box)
        return True, box
    return False, None
