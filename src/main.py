"""
Slice 5 — 차선 이탈 경고(LDW) + 주행영역 오버레이 (M4).

각 프레임에 대해:
  1. 색상필터+Canny 결합 → lane_mask
  2. ROI 마스킹 → roi_mask
  3. HoughLinesP → raw segments
  4. 기울기 기준 좌/우 분리 (split_segments)
  5. 가중 폴리핏으로 좌/우 각 1개 직선 피팅 (fit_lane) → raw_left, raw_right
  6. LaneSmoother.update() → smoothed_left/right + left_state/right_state
  7. departure.offset() → off
  8. DepartureState.update(off) → (warning, side)
  9. overlay.render_frame() → 주행영역+차선+배너+HUD 합성
  10. CSV 행 기록 (frame, left_state, right_state)
  11. 결과 mp4 출력

종료 후:
  - output/{clip}_detect_log.csv 저장 (header + 1행/프레임)
  - 콘솔에 side별 상태 카운트 + raw_detected rate 출력

디버그: --debug-frame N 으로 지정한 프레임(기본=130)에서
  frames/slice4_colormask.png, frames/slice4_lanemask.png, frames/slice4_overlay.png 저장.

debug-warn: --debug-warn 플래그 시 frames/ldw_off.png, frames/ldw_on.png 생성.
  ldw_on.png = 합성 오프셋(1.0)으로 강제 경고 트리거한 프레임.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

import cv2
import numpy as np

from src import config
from src import preprocess
from src import roi
from src import lane_detect
from src.smoothing import LaneSmoother, VALID_STATES
from src import departure as dep
from src import overlay as ov
from src import birdeye
from src import curve as curvemod
from src import vehicle_bonus as vb
from src import vehicle_track as vt


def _roi_y_top(W: int, H: int, clip_key: str) -> int:
    """
    ROI 사다리꼴의 상단 y좌표를 config.ROI_TRAPEZOID_RATIO에서 유도.

    ROI 꼭짓점 중 최소 ry 비율 × H = 차선선 외삽 상단 경계.
    """
    key = config.res_key(clip_key)
    ratios = config.ROI_TRAPEZOID_RATIO[key]
    min_ry = min(ry for _, ry in ratios)
    return int(min_ry * H)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RoadVision — Slice 6: Bird-eye view (M5) + LDW + drivable area overlay"
    )
    parser.add_argument(
        "--clip",
        choices=list(config.CLIPS.keys()),
        default="solidYellowLeft",
        help="처리할 클립 키 (config.CLIPS 기준)",
    )
    parser.add_argument(
        "--debug-frame",
        type=int,
        default=130,
        help="디버그 프레임 저장할 프레임 번호 (1-based, 기본=130)",
    )
    parser.add_argument(
        "--debug-warn",
        action="store_true",
        help="ldw_off.png / ldw_on.png 생성 (경고 OFF/ON 디버그 프레임)",
    )
    parser.add_argument(
        "--debug-birdeye",
        action="store_true",
        help="{clip}_birdeye_debug.png 생성 (원본+src 사다리꼴 | 탑다운 워프 나란히)",
    )
    parser.add_argument(
        "--vehicle",
        action="store_true",
        help="M7 보너스: 전방 차량 후보 시각화 실험 ON (기본=OFF). "
             "출력: {clip}_vehicle.mp4 (lane-only 출력 덮어쓰지 않음).",
    )
    parser.add_argument(
        "--ldw-demo",
        action="store_true",
        help=(
            "LDW 데모 모드: 스크립트 드리프트를 오프셋에 주입해 "
            "SAFE→CAUTION→DANGER(LEFT)→SAFE→DANGER(RIGHT)→SAFE 사이클을 시연. "
            "실제 경로는 변경 없음(시뮬레이션). "
            "출력: {clip}_ldwdemo.mp4 (lane-only 덮어쓰지 않음). "
            "HUD에 'LDW DEMO -- drift simulated' 태그 표시."
        ),
    )
    parser.add_argument(
        "--vehicle-track",
        action="store_true",
        help=(
            "L10 CSRT 차량 추적 데모 ON. "
            "config.VEHICLE['track_seed']에 클립별 seed 박스를 등록해야 함. "
            "출력: output/{clip}_track.mp4 (lane-only 덮어쓰지 않음). "
            "lane 오버레이를 유지하면서 추적 박스·궤적을 위에 합성. "
            "3개 검증 프레임: frames/track_{1,2,3}.png."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip_key = args.clip
    clip_cfg = config.CLIPS[clip_key]
    debug_frame_idx = args.debug_frame  # 1-based

    input_path = clip_cfg["path"]

    # 출력 디렉토리 생성
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.FRAMES_DIR, exist_ok=True)

    # 출력 파일 경로 — 모드별로 분리 (lane-only 덮어쓰기 방지)
    if args.vehicle:
        output_path = os.path.join(config.OUTPUT_DIR, f"{clip_key}_vehicle.mp4")
    elif args.ldw_demo:
        output_path = os.path.join(config.OUTPUT_DIR, f"{clip_key}_ldwdemo.mp4")
    elif args.vehicle_track:
        output_path = os.path.join(config.OUTPUT_DIR, f"{clip_key}_track.mp4")
    else:
        output_path = os.path.join(config.OUTPUT_DIR, f"{clip_key}_overlay.mp4")
    csv_path    = os.path.join(config.OUTPUT_DIR, f"{clip_key}_detect_log.csv")

    # 캡처 열기
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {input_path}")

    # 메타데이터를 캡처에서 직접 읽음 (하드코딩 금지)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"입력: {input_path}  해상도={width}x{height}  fps={fps:.2f}  총프레임={total}")

    # ROI 상단 y좌표 (차선선 외삽 범위)
    y_bottom = height
    y_top    = _roi_y_top(width, height, clip_key)
    print(f"ROI y 범위: y_top={y_top}, y_bottom={y_bottom}")
    print(f"스무딩 설정: SMOOTH_WINDOW={config.SMOOTH_WINDOW}  "
          f"OUTLIER_SLOPE_DEV={config.OUTLIER_SLOPE_DEV}  "
          f"HOLD_MAX_FRAMES={config.HOLD_MAX_FRAMES}")
    print(f"LDW 설정 (물리 모델): "
          f"lane={config.LDW['lane_width_m']}m  "
          f"car={config.LDW['vehicle_width_m']}m  "
          f"danger≤{config.LDW['danger_dist_m']}m  "
          f"caution≤{config.LDW['caution_dist_m']}m  "
          f"exit>{config.LDW['danger_exit_dist_m']}m  "
          f"fill_alpha={config.LDW['fill_alpha']}")

    # 스무더 + 이탈 상태 + 베이스라인 보정기 초기화
    smoother   = LaneSmoother()
    dep_state  = dep.DepartureState()
    calibrator = dep.BaselineCalibrator()  # 카메라 마운트 바이어스 보정 (config.BASELINE_ENABLE)

    # LDW 데모 모드: 4-state 대표 프레임 저장 추적 (DANGER는 LEFT/RIGHT 분리)
    # 키: "SAFE", "CAUTION", "DANGER_L", "DANGER_R"
    demo_saved: dict[str, bool] = {
        "SAFE": False, "CAUTION": False, "DANGER_L": False, "DANGER_R": False
    }

    # CSV 버퍼
    csv_rows: list[tuple[int, str, str]] = []

    # 상태 카운터 (honesty gate)
    state_counts: dict[str, dict[str, int]] = {
        "left":  {s: 0 for s in VALID_STATES},
        "right": {s: 0 for s in VALID_STATES},
    }

    # M6 곡선 모드 카운터 (CURVE / fallback 프레임 수 집계)
    curve_frame_counts = {"CURVE": 0, "STRAIGHT(fallback)": 0}

    # 3-state LDW 상태 카운터 (정직성 게이트: 정상 주행에서 CAUTION/DANGER = 0이어야)
    lane_st_counts: dict[str, int] = {"SAFE": 0, "CAUTION": 0, "DANGER": 0}

    # M7 차량 보너스 상태 (vehicle 플래그 ON 시에만 사용)
    _prev_gray: np.ndarray | None = None   # optical flow용 직전 프레임 gray
    _veh_frames_with_boxes = 0             # 박스가 1개 이상 뜬 프레임 수
    _veh_total_boxes       = 0             # 누적 박스 수
    _veh_saved_samples     = 0            # 캡처된 샘플 프레임 수
    _veh_saved_fp          = False         # FP 샘플 저장 여부
    _veh_saved_missed      = False         # missed 샘플 저장 여부
    if args.vehicle:
        print(f"[M7] 차량 보너스 모드 ON  cascade={config.VEHICLE['cascade_path']}  "
              f"scale_factor={config.VEHICLE['scale_factor']}  "
              f"min_neighbors={config.VEHICLE['min_neighbors']}  "
              f"min_size={config.VEHICLE['min_size']}")

    # ── L10 CSRT 추적 상태 (--vehicle-track 시에만 사용) ──────────────────────────
    _track_tracker   = None          # CSRT tracker 인스턴스 (None = 아직 초기화 전)
    _track_seeded    = False         # seed 완료 여부
    _track_ok        = False         # 현재 프레임 추적 성공 여부
    _track_box: tuple | None = None  # 현재 추적 박스 (x, y, w, h)
    _track_trail: list[tuple[int, int]] = []   # 중심 좌표 이력
    _track_trail_len = config.VEHICLE.get("track_trail_len", 30)
    _track_ok_count  = 0             # 추적 성공 프레임 수 (집계용)
    _track_total_after_seed = 0      # seed 이후 처리 프레임 수 (분모)
    _track_first_loss: int | None = None  # 첫 LOST 프레임 번호
    # 검증 프레임 저장: seed 이후 +10, +100, +300 프레임
    _track_verify_offsets = {10, 100, 300}
    _track_verify_saved: set[int] = set()
    _track_verify_count = 0  # 저장된 검증 프레임 수 (최대 3)

    if args.vehicle_track:
        seed_entry = vt.seed_box(clip_key)
        if seed_entry[0] == "auto":
            print(f"[L10 TRACK] clip={clip_key}에 track_seed 미설정 — lane-only 출력.")
            _track_seed_frame = None
            _track_seed_box   = None
        else:
            _track_seed_frame, _track_seed_box = seed_entry
            print(f"[L10 TRACK] CSRT 추적 모드 ON  "
                  f"clip={clip_key}  seed_frame={_track_seed_frame}  box={_track_seed_box}")
    else:
        _track_seed_frame = None
        _track_seed_box   = None

    # 디버그 프레임 저장 여부 추적
    saved_ldw_off   = False
    saved_ldw_on    = False
    saved_birdeye   = False
    saved_curve_pv  = set()  # project_video 곡선 디버그 프레임 저장 번호 집합

    # bird-eye 디버그 프레임 번호: project_video는 직선 구간인 100번 프레임 사용
    birdeye_debug_frame = 100 if clip_key == "project_video" else 130

    frames_read    = 0
    frames_written = 0
    t_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_read += 1
        W, H = width, height

        # 1. 색상필터 + Canny 결합 → lane_mask
        lmask = preprocess.lane_mask(frame)

        # 2. ROI 마스킹
        roi_mask = roi.apply(lmask, W, H, clip_key)

        # 3. HoughLinesP → raw segments
        segments = lane_detect.raw_segments(roi_mask)

        # 4. 기울기 기준 좌/우 분리
        left_segs, right_segs = lane_detect.split_segments(segments)

        # 5. 가중 폴리핏 (raw)
        raw_left  = lane_detect.fit_lane(left_segs, y_bottom, y_top)
        raw_right = lane_detect.fit_lane(right_segs, y_bottom, y_top)

        # 6. 시간평활 + 이상치 거부
        smoothed_left,  left_state  = smoother.update("left",  raw_left)
        smoothed_right, right_state = smoother.update("right", raw_right)

        # 상태 카운트 누적
        state_counts["left"][left_state]   += 1
        state_counts["right"][right_state] += 1

        # CSV 행 버퍼 (schema 유지: frame, left_state, right_state)
        csv_rows.append((frames_read, left_state, right_state))

        # 7. LDW 오프셋 계산
        off = dep.offset(smoothed_left, smoothed_right, W)

        # 7c. 베이스라인 바이어스 보정 (카메라 마운트 편향 제거)
        # raw off는 CSV·smoother·state_counts에 이미 반영됨 → 보정은 표시·판정에만 적용.
        # calibrator.feed()는 수집 완료 전까지 샘플을 쌓고, 완료 후 no-op.
        # calibrator.apply()는 수집 완료 전에는 off 그대로, 완료 후 off−bias 반환.
        # 가정: 클립 시작 구간에서 차량이 차선 중앙 주행 (PLAN §8 honesty note).
        if not args.ldw_demo:
            calibrator.feed(off)
        off_cal = calibrator.apply(off) if not args.ldw_demo else off

        # 7b. LDW DEMO 드리프트 주입 (--ldw-demo 시에만, off가 None이 아닐 때만)
        # 실제 CSV·smoother·state_counts는 7 이전에 모두 기록됨 → 무변경 보장.
        # 드리프트: 사인파. amplitude × sin(2π × frame / period)
        #   frame 0~period/4 : 0 → +amplitude (RIGHT caution→danger)
        #   frame period/4~period/2: +amplitude → 0 (safe)
        #   frame period/2~3period/4: 0 → -amplitude (LEFT danger)
        #   frame 3period/4~period: -amplitude → 0 (safe)
        # 부호: 음수=LEFT 이탈, 양수=RIGHT 이탈 (departure.py 규약)
        # 실제 off는 미세하므로 drift가 상태 전환을 주도함.
        _demo_drift_off = off_cal  # 데모/일반 분기 후 동일 변수명 유지
        if args.ldw_demo and off is not None:
            amp    = config.LDW["demo_drift_amplitude"]
            period = config.LDW["demo_drift_period_frames"]
            # 부호: 음수 먼저 → LEFT DANGER 먼저, 양수 → RIGHT DANGER (task 시퀀스)
            drift  = -amp * math.sin(2 * math.pi * frames_read / period)
            _demo_drift_off = off_cal + drift

        # 8. 히스테리시스 경고 상태 갱신
        # 데모 모드: 드리프트 적용된 오프셋으로 경고 판단
        # 일반 모드: 보정된 off_cal 사용 (기존 동작에 바이어스 보정 추가)
        if args.ldw_demo:
            warning, side = dep_state.update(_demo_drift_off)
        else:
            warning, side = dep_state.update(off_cal)

        # debug-warn용 + birdeye-debug용 원본 복사 (렌더링 전)
        need_clean_copy = (
            (args.debug_warn and frames_read == debug_frame_idx)
            or (args.debug_birdeye and frames_read == birdeye_debug_frame and not saved_birdeye)
        )
        frame_clean = frame.copy() if need_clean_copy else None

        # Bird-eye 워프 — render_frame 전에 깨끗한 원본 프레임에서 계산 (오버레이 없는 탑다운)
        warped_frame = birdeye.warp(frame, clip_key)

        # ── M6 곡선 슬라이딩 윈도우 (fail-safe: 실패 시 직선 fallback) ──────────────
        # 주의: 아래 블록은 CSV·state_counts·smoother·off·LDW를 절대 수정하지 않음.
        #       오버레이 렌더링 방식만 결정한다.
        curve_mode   : str | None     = None
        curvature_r  : float | None   = None
        curve_polygon: np.ndarray | None = None
        curve_left   : np.ndarray | None = None
        curve_right  : np.ndarray | None = None

        try:
            lmask_warped = preprocess.lane_mask(frame)  # 원본 컬러 마스크 재계산 (오버레이 전)
            warped_mask  = birdeye.warp(lmask_warped, clip_key)

            left_poly, right_poly = curvemod.sliding_window_fit(warped_mask)

            if curvemod.is_valid(left_poly, right_poly, warped_mask):
                polygon, l_pts, r_pts = curvemod.curved_lane_points(
                    left_poly, right_poly, height, clip_key
                )
                # 곡률반경: 하단 (y = H-1) 기준
                r_left  = curvemod.curvature_radius(left_poly,  height - 1)
                r_right = curvemod.curvature_radius(right_poly, height - 1)
                avg_r   = (r_left + r_right) / 2.0

                curve_mode    = "CURVE"
                curvature_r   = avg_r
                curve_polygon = polygon
                curve_left    = l_pts
                curve_right   = r_pts
            else:
                curve_mode = "STRAIGHT(fallback)"
        except Exception:
            # 예외 발생 시 항상 fallback — Core 보호
            curve_mode = "STRAIGHT(fallback)"

        # 카운터 누적
        if curve_mode in curve_frame_counts:
            curve_frame_counts[curve_mode] += 1
        # ── M6 끝 ──────────────────────────────────────────────────────────────────

        # 9. 오버레이 합성 (in-place)
        # 데모 모드에서는 드리프트 오프셋(_demo_drift_off)으로 3-state 계산.
        # 일반 모드에서는 보정된 off_cal으로 계산.
        _off_for_display = _demo_drift_off if args.ldw_demo else off_cal
        lane_st = dep.lane_state(_off_for_display, warning, dep_state.caution)

        # 3-state LDW 카운터 누적 (정직성 게이트 출력용)
        if not args.ldw_demo:
            lane_st_counts[lane_st] = lane_st_counts.get(lane_st, 0) + 1

        # LDW 데모: 4-state 대표 프레임 저장 (처음 만나는 각 상태에서 1회)
        # 키: "SAFE", "CAUTION", "DANGER_L", "DANGER_R"
        if args.ldw_demo and not all(demo_saved.values()):
            if lane_st == "DANGER" and side == "LEFT" and not demo_saved["DANGER_L"]:
                _demo_save_key = "DANGER_L"
            elif lane_st == "DANGER" and side == "RIGHT" and not demo_saved["DANGER_R"]:
                _demo_save_key = "DANGER_R"
            elif lane_st in ("SAFE", "CAUTION") and not demo_saved[lane_st]:
                _demo_save_key = lane_st
            else:
                _demo_save_key = None
        else:
            _demo_save_key = None

        if curve_mode == "CURVE" and curve_polygon is not None:
            # CURVE 모드: 곡선 drivable area + 곡선 차선선 먼저, 배너+HUD는 render_frame
            ov.draw_curved_area(frame, curve_polygon, curve_left, curve_right,
                                lane_st=lane_st)
            ov.render_frame(
                frame,
                smoothed_left, smoothed_right,
                y_bottom, y_top,
                _off_for_display, warning, side,
                frames_read, total,
                len(segments),
                left_state, right_state,
                draw_area=False,        # 직선 폴리곤 스킵 (곡선으로 대체)
                draw_lines=False,       # 직선 차선선 스킵 (곡선으로 대체)
                curve_mode=curve_mode,
                curvature_r=curvature_r,
                lane_st=lane_st,
                demo=args.ldw_demo,
            )
        else:
            # STRAIGHT(fallback) 모드: 기존 직선 파이프라인 그대로
            ov.render_frame(
                frame,
                smoothed_left, smoothed_right,
                y_bottom, y_top,
                _off_for_display, warning, side,
                frames_read, total,
                len(segments),
                left_state, right_state,
                curve_mode=curve_mode,  # "STRAIGHT(fallback)" 또는 None
                lane_st=lane_st,
                demo=args.ldw_demo,
            )

        # LDW 데모: 렌더 완료 후 대표 프레임 저장 (ldw3_ 접두사, DANGER는 L/R 분리)
        if _demo_save_key is not None and not demo_saved[_demo_save_key]:
            state_frame_path = os.path.join(
                config.FRAMES_DIR, f"ldw3_{_demo_save_key.lower()}.png"
            )
            cv2.imwrite(state_frame_path, frame)
            demo_saved[_demo_save_key] = True
            off_disp = f"{_off_for_display:.4f}" if _off_for_display is not None else "N/A"
            import src.departure as _dep_mod
            wtl_disp = (f"{_dep_mod.wheel_to_line_m(_off_for_display):.3f} m"
                        if _off_for_display is not None else "N/A")
            print(
                f"  [ldw-demo] {_demo_save_key} 프레임 저장: {state_frame_path}"
                f"  (frame={frames_read}, drift_off={off_disp}, wheel->line={wtl_disp})"
            )

        # Bird-eye PiP 합성 (render_frame 이후, 오버레이된 프레임 우상단에 삽입)
        ov.draw_birdeye_pip(frame, warped_frame)

        # kr_ 검증 프레임 저장 (frames/kr_{clip}.png): 안정 구간 1회
        # 목적: 한글 렌더링·범례·상태 SAFE 시각적 검증.
        # 프레임 50 (스무더 워밍업 이후, calibrator 수집 완료 이후 안정 구간)
        _KR_FRAME = 50
        if frames_read == _KR_FRAME and not args.ldw_demo:
            kr_path = os.path.join(config.FRAMES_DIR, f"kr_{clip_key}.png")
            cv2.imwrite(kr_path, frame)
            print(f"  [kr-verify] 한글 검증 프레임 저장: {kr_path}  "
                  f"(frame={frames_read}, lane_st={lane_st})")

        # M6 project_video 곡선 디버그 프레임 저장 (frames/curve_pv_{n}.png)
        _CURVE_PV_FRAMES = {1000, 1030, 1060}
        if frames_read in _CURVE_PV_FRAMES and frames_read not in saved_curve_pv:
            pv_path = os.path.join(config.FRAMES_DIR, f"curve_pv_{frames_read}.png")
            cv2.imwrite(pv_path, frame)
            saved_curve_pv.add(frames_read)
            print(f"  [curve-pv] {pv_path} 저장 (mode={curve_mode})")

        # 10. 디버그 프레임 저장
        if frames_read == debug_frame_idx:
            # 기존 slice4 디버그 파일들 (호환)
            cmask = preprocess.color_mask(frame)
            cv2.imwrite(os.path.join(config.FRAMES_DIR, "slice4_colormask.png"), cmask)
            cv2.imwrite(os.path.join(config.FRAMES_DIR, "slice4_lanemask.png"), lmask)
            overlay_debug = frame.copy()
            roi_poly = roi.polygon(W, H, clip_key)
            cv2.polylines(overlay_debug, [roi_poly], True, (255, 0, 0), 2)
            cv2.imwrite(os.path.join(config.FRAMES_DIR, "slice4_overlay.png"), overlay_debug)
            off_str = f"{off:.4f}" if off is not None else "N/A"
            print(
                f"  [debug] 프레임 {frames_read}: "
                f"segs={len(segments)}  "
                f"L_raw={'OK' if raw_left else 'MISS'}({left_state})  "
                f"R_raw={'OK' if raw_right else 'MISS'}({right_state})  "
                f"off={off_str}  "
                f"warn={warning} side={side}"
            )

        # --debug-warn: debug_frame_idx(기본=130)에서 ldw_off.png + ldw_on.png 저장.
        # debug_frame_idx는 스무더가 충분히 워밍업된 안정적인 프레임 — 대표성 보장.
        if args.debug_warn and frames_read == debug_frame_idx:
            # ldw_off.png: 실제 렌더링된 정상 프레임 (현재 frame에 이미 render_frame 적용됨)
            if not saved_ldw_off and off is not None:
                cv2.imwrite(os.path.join(config.FRAMES_DIR, "ldw_off.png"), frame)
                saved_ldw_off = True
                off_str = f"{off:.4f}" if off is not None else "N/A"
                print(f"  [debug-warn] ldw_off.png 저장 (frame={frames_read}, off={off_str})")

            # ldw_on.png: 렌더링 전 원본(frame_clean)에 합성 경고 오버레이 적용
            if not saved_ldw_on and frame_clean is not None \
                    and smoothed_left is not None and smoothed_right is not None:
                synth_frame = frame_clean.copy()
                # 합성 파라미터: 우측 이탈 (|offset|=1.0 >> warn_on=0.35)
                synth_off, synth_warning, synth_side = 1.0, True, "RIGHT"
                ov.render_frame(
                    synth_frame,
                    smoothed_left, smoothed_right,
                    y_bottom, y_top,
                    synth_off, synth_warning, synth_side,
                    frames_read, total,
                    len(segments),
                    left_state, right_state,
                )
                cv2.imwrite(os.path.join(config.FRAMES_DIR, "ldw_on.png"), synth_frame)
                saved_ldw_on = True
                print(f"  [debug-warn] ldw_on.png 저장 (frame={frames_read}, synth_off=1.0, side=RIGHT)")

        # --debug-birdeye: birdeye_debug_frame에서 side-by-side 디버그 이미지 저장.
        # 좌: 원본 프레임(frame_clean)에 src 사다리꼴(초록 폴리라인) 그린 것.
        # 우: 탑다운 워프된 이미지 (warped_frame — 오버레이 전 원본에서 워프됨).
        if args.debug_birdeye and frames_read == birdeye_debug_frame and not saved_birdeye:
            if frame_clean is not None:
                # src 사다리꼴 그리기 (BGR 초록)
                debug_src = frame_clean.copy()
                src_pts = np.array(config.BIRDEYE[clip_key]["src"], dtype=np.int32)
                # tl→tr→br→bl 순 — 닫힌 폴리라인
                cv2.polylines(debug_src, [src_pts], isClosed=True,
                              color=(0, 255, 0), thickness=3)
                # 꼭짓점 레이블
                labels = ["TL", "TR", "BR", "BL"]
                for (px, py), lbl in zip(config.BIRDEYE[clip_key]["src"], labels):
                    cv2.circle(debug_src, (px, py), 6, (0, 255, 0), -1)
                    cv2.putText(debug_src, lbl, (px + 8, py - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # side-by-side: [원본+사다리꼴] | [탑다운 워프]
                side_by_side = np.hstack([debug_src, warped_frame])

                debug_path = os.path.join(config.FRAMES_DIR,
                                          f"{clip_key}_birdeye_debug.png")
                cv2.imwrite(debug_path, side_by_side)
                saved_birdeye = True
                print(f"  [debug-birdeye] {debug_path} 저장 (frame={frames_read})")

        # ── M7 차량 보너스 (ADDITIVE 레이어, lane pipeline 변수 불변) ─────────────
        # 이 블록은 args.vehicle 플래그 ON 시에만 실행된다.
        # 변경 대상: frame 픽셀(드로잉) + _prev_gray(옵티컬플로 버퍼) + 통계 변수.
        # 변경 금지: smoother, dep_state, csv_rows, state_counts, curve_mode.
        if args.vehicle:
            # grayscale 계산 (flow용 + cascade 내부용 공용)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1) Haar cascade 후보 검출
            veh_boxes = vb.detect_candidates(frame, clip_key)
            n_boxes   = len(veh_boxes)
            if n_boxes > 0:
                _veh_frames_with_boxes += 1
                _veh_total_boxes       += n_boxes

            # 2) 박스 + ROI 경계 + HUD 태그 그리기 (차선 오버레이 위에 마지막으로)
            vb.draw_roi_boundary(frame)
            vb.draw_candidates(frame, veh_boxes)
            vb.draw_vehicle_hud_tag(frame, n_boxes)

            # 3) 광류 히트맵 (이전 프레임이 있을 때만)
            if _prev_gray is not None:
                flow_mag = vb.compute_flow_magnitude(
                    _prev_gray, curr_gray, (height, width)
                )
                vb.draw_flow_heatmap(frame, flow_mag, (height, width))

            _prev_gray = curr_gray  # 다음 프레임용 보존

            # 4) 샘플 프레임 캡처 (vehicle_{n}.png — 박스 있는 프레임)
            if _veh_saved_samples < 3 and n_boxes > 0:
                sample_path = os.path.join(
                    config.FRAMES_DIR, f"vehicle_{frames_read}.png"
                )
                cv2.imwrite(sample_path, frame)
                _veh_saved_samples += 1
                print(f"  [M7] vehicle sample: {sample_path}  boxes={n_boxes}")

            # 5) FP 캡처: 박스가 뜨는 두 번째 hit 프레임 저장.
            #    첫 번째 hit 이후의 프레임으로 FP 예시를 보존 (두 개의 독립 샘플).
            #    실제로 boxes 좌표가 도로 질감/소실점에 걸리는지는 frames/vehicle_fp.png로 육안 확인.
            if not _veh_saved_fp and n_boxes > 0 and _veh_saved_samples >= 2:
                fp_path = os.path.join(config.FRAMES_DIR, "vehicle_fp.png")
                cv2.imwrite(fp_path, frame)
                _veh_saved_fp = True
                print(f"  [M7] vehicle FP candidate: {fp_path}  frame={frames_read}")

            # 6) missed 캡처: 박스 0개인 프레임 (100번 이후, 안정 구간에서 한 번만 저장).
            #    실제 차량이 있을지는 육안으로 vehicle_missed.png 확인 필요.
            if not _veh_saved_missed and n_boxes == 0 and frames_read > 100:
                missed_path = os.path.join(config.FRAMES_DIR, "vehicle_missed.png")
                cv2.imwrite(missed_path, frame)
                _veh_saved_missed = True
                print(f"  [M7] vehicle missed candidate: {missed_path}  frame={frames_read}")
        # ── M7 끝 ─────────────────────────────────────────────────────────────────

        # ── L10 CSRT 차량 추적 (ADDITIVE, lane pipeline 불변) ─────────────────────
        # 변경 대상: frame 픽셀(드로잉) + _track_* 상태.
        # 변경 금지: smoother, dep_state, csv_rows, state_counts, curve_mode.
        if args.vehicle_track and _track_seed_frame is not None:

            # seed 프레임에서 추적기 초기화 (1회)
            if not _track_seeded and frames_read == _track_seed_frame:
                _track_tracker = vt.create_tracker()
                vt.init_tracker(_track_tracker, frame, _track_seed_box)
                _track_seeded = True
                _track_ok    = True                          # seed 프레임 자체는 OK
                _track_box   = tuple(_track_seed_box)
                cx = _track_box[0] + _track_box[2] // 2
                cy = _track_box[1] + _track_box[3] // 2
                _track_trail.append((cx, cy))
                print(f"  [L10 TRACK] seed 초기화 완료 frame={frames_read}  box={_track_box}")

            elif _track_seeded and _track_tracker is not None:
                # seed 이후: 매 프레임 추적기 갱신
                _track_ok, _track_box = vt.update(_track_tracker, frame)
                _track_total_after_seed += 1

                if _track_ok:
                    _track_ok_count += 1
                    # 궤적 추가 (최대 trail_len개 유지)
                    cx = _track_box[0] + _track_box[2] // 2
                    cy = _track_box[1] + _track_box[3] // 2
                    _track_trail.append((cx, cy))
                    if len(_track_trail) > _track_trail_len:
                        _track_trail.pop(0)
                else:
                    if _track_first_loss is None:
                        _track_first_loss = frames_read
                        print(f"  [L10 TRACK] LOST at frame={frames_read}  "
                              f"tracked={_track_ok_count}/{_track_total_after_seed}")

                # 검증 프레임 저장 (seed 이후 +10, +100, +300)
                offset = frames_read - _track_seed_frame
                if offset in _track_verify_offsets and offset not in _track_verify_saved:
                    _track_verify_count += 1
                    vpath = os.path.join(
                        config.FRAMES_DIR, f"track_{_track_verify_count}.png"
                    )
                    # 현재 오버레이(lane + 추적박스 추가 전) 복사본에 박스 그리기
                    verify_frame = frame.copy()
                    ov.draw_track(verify_frame, _track_box, _track_ok,
                                  list(_track_trail))
                    # 추가: seed frame 번호와 현재 프레임 번호 표시
                    cv2.putText(verify_frame,
                                f"frame={frames_read}  seed={_track_seed_frame}  +{offset}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(verify_frame,
                                f"frame={frames_read}  seed={_track_seed_frame}  +{offset}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imwrite(vpath, verify_frame)
                    _track_verify_saved.add(offset)
                    print(f"  [L10 TRACK] 검증 프레임 저장: {vpath}  "
                          f"ok={_track_ok}  frame={frames_read}")

            # 오버레이: 추적 박스 + 궤적 그리기 (lane overlay 위에)
            if _track_seeded:
                ov.draw_track(frame, _track_box, _track_ok, list(_track_trail))
        # ── L10 CSRT 끝 ───────────────────────────────────────────────────────────

        writer.write(frame)
        frames_written += 1

    cap.release()
    writer.release()

    elapsed = time.perf_counter() - t_start
    avg_fps = frames_written / elapsed if elapsed > 0 else 0.0

    # ------------------------------------------------------------------
    # CSV 저장
    # ------------------------------------------------------------------
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "left_state", "right_state"])
        w.writerows(csv_rows)
    print(f"\nCSV 저장: {csv_path}  ({len(csv_rows)} 데이터 행)")

    # ------------------------------------------------------------------
    # 상태 집계 검증 (honesty gate)
    # ------------------------------------------------------------------
    total_frames = frames_read

    print("\n=== 검출 상태 요약 (C2 메트릭) ===")
    for side_name in ("left", "right"):
        cnts   = state_counts[side_name]
        raw    = cnts["raw_detected"]
        rej    = cnts["rejected_as_outlier"]
        hld    = cnts["held_from_previous"]
        mis    = cnts["consecutive_missing"]
        raw_rate = raw / total_frames * 100 if total_frames else 0.0

        print(f"\n  [{side_name.upper()}]")
        print(f"    raw_detected        : {raw:4d}  ({raw_rate:.1f}% — C2 기준)")
        print(f"    rejected_as_outlier : {rej:4d}  ({rej/total_frames*100:.1f}%)")
        print(f"    held_from_previous  : {hld:4d}  ({hld/total_frames*100:.1f}%)")
        print(f"    consecutive_missing : {mis:4d}  ({mis/total_frames*100:.1f}%)")
        print(f"    합계 검증           : {raw+rej+hld+mis} / {total_frames}"
              f"  {'OK' if raw+rej+hld+mis == total_frames else '*** MISMATCH ***'}")

    # 베이스라인 보정 상태 보고
    if not args.ldw_demo:
        print("\n=== 베이스라인 바이어스 보정 ===")
        print(f"  BASELINE_ENABLE  : {config.BASELINE_ENABLE}")
        print(f"  BASELINE_FRAMES  : {config.BASELINE_FRAMES}")
        print(f"  수집 샘플 수     : {calibrator.n_collected}")
        print(f"  보정 완료        : {calibrator.calibrated}")
        print(f"  bias (중앙값)    : {calibrator.bias:+.4f}"
              f"  → 차선 중앙 offset이 {calibrator.bias:+.4f}에서 ≈0으로 보정됨")
        print("  ※ 가정: 클립 시작 구간에서 차량이 차선 중앙 주행 (PLAN §8 honesty note)")

    # 3-state LDW 상태 분포 (정직성 게이트)
    if not args.ldw_demo:
        safe_n    = lane_st_counts.get("SAFE", 0)
        caution_n = lane_st_counts.get("CAUTION", 0)
        danger_n  = lane_st_counts.get("DANGER", 0)
        total_st  = safe_n + caution_n + danger_n
        print("\n=== 3-state LDW 판정 분포 (정직성 게이트) ===")
        print(f"  SAFE    : {safe_n:5d}프레임 ({100*safe_n/max(total_st,1):.1f}%)")
        print(f"  CAUTION : {caution_n:5d}프레임 ({100*caution_n/max(total_st,1):.1f}%)")
        print(f"  DANGER  : {danger_n:5d}프레임 ({100*danger_n/max(total_st,1):.1f}%)")
        cd_total = caution_n + danger_n
        gate = "OK (정상주행=SAFE)" if cd_total == 0 else f"*** 주의: {cd_total}프레임 비정상 ***"
        print(f"  CAUTION+DANGER : {cd_total}프레임 → {gate}")

    # M6 곡선 모드 집계 보고
    print("\n=== M6 곡선 모드 집계 ===")
    for mode_name, cnt in curve_frame_counts.items():
        pct = cnt / frames_read * 100 if frames_read else 0.0
        print(f"  {mode_name:25s}: {cnt:5d}프레임 ({pct:.1f}%)")

    # M7 차량 보너스 집계 보고
    if args.vehicle:
        hit_rate = _veh_frames_with_boxes / frames_read * 100 if frames_read else 0.0
        avg_boxes = _veh_total_boxes / _veh_frames_with_boxes if _veh_frames_with_boxes else 0.0
        print("\n=== M7 차량 후보 실험 집계 ===")
        print(f"  cascade 파라미터: scale_factor={config.VEHICLE['scale_factor']}  "
              f"min_neighbors={config.VEHICLE['min_neighbors']}  "
              f"min_size={config.VEHICLE['min_size']}")
        print(f"  전방 ROI: {config.VEHICLE['forward_roi_ratio']}")
        print(f"  박스 ≥1 프레임: {_veh_frames_with_boxes}/{frames_read} ({hit_rate:.1f}%)")
        print(f"  누적 박스 수  : {_veh_total_boxes}  평균(hit프레임): {avg_boxes:.1f}")
        print(f"  sample 캡처   : {_veh_saved_samples}개  FP={_veh_saved_fp}  missed={_veh_saved_missed}")
        print("  ※ 이 수치는 Haar cascade 특성상 FP(오검)이 다수 포함됨. 정량 정확도 아님.")

    # L10 CSRT 추적 집계 보고
    if args.vehicle_track and _track_seed_frame is not None and _track_seeded:
        track_rate = _track_ok_count / _track_total_after_seed * 100 if _track_total_after_seed else 0.0
        print("\n=== L10 CSRT 차량 추적 집계 ===")
        print(f"  clip        : {clip_key}")
        print(f"  seed 프레임 : {_track_seed_frame}  box={_track_seed_box}")
        print(f"  추적 성공   : {_track_ok_count}/{_track_total_after_seed}  ({track_rate:.1f}%)")
        print(f"  첫 LOST     : {'없음 (전 구간 추적 성공)' if _track_first_loss is None else f'frame {_track_first_loss}'}")
        print(f"  검증 프레임 : {_track_verify_count}개  (frames/track_{{1,2,3}}.png)")
        print("  ※ seed 박스는 클립별 하드코딩. 검출(detection) 없이 추적(tracking)만 수행.")
        print("  ※ CSRT 특성상 타깃이 프레임 밖으로 나가거나 심한 가림/스케일 변화 시 LOST 가능.")

    print("\n=== 처리 완료 ===")
    print(f"  입력 경로  : {input_path}")
    print(f"  출력 경로  : {output_path}")
    print(f"  해상도     : {width}x{height}")
    print(f"  fps        : {fps:.2f}")
    print(f"  읽은 프레임: {frames_read}")
    print(f"  쓴 프레임  : {frames_written}")
    print(f"  소요 시간  : {elapsed:.2f}s")
    print(f"  처리 FPS   : {avg_fps:.1f}")


if __name__ == "__main__":
    main()
