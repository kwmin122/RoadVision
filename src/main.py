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
    print(f"LDW 설정: warn_on={config.LDW['warn_on']}  "
          f"warn_off={config.LDW['warn_off']}  "
          f"fill_alpha={config.LDW['fill_alpha']}")

    # 스무더 + 이탈 상태 초기화
    smoother  = LaneSmoother()
    dep_state = dep.DepartureState()

    # CSV 버퍼
    csv_rows: list[tuple[int, str, str]] = []

    # 상태 카운터 (honesty gate)
    state_counts: dict[str, dict[str, int]] = {
        "left":  {s: 0 for s in VALID_STATES},
        "right": {s: 0 for s in VALID_STATES},
    }

    # M6 곡선 모드 카운터 (CURVE / fallback 프레임 수 집계)
    curve_frame_counts = {"CURVE": 0, "STRAIGHT(fallback)": 0}

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

        # 8. 히스테리시스 경고 상태 갱신
        warning, side = dep_state.update(off)

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
        if curve_mode == "CURVE" and curve_polygon is not None:
            # CURVE 모드: 곡선 drivable area + 곡선 차선선 먼저, 배너+HUD는 render_frame
            ov.draw_curved_area(frame, curve_polygon, curve_left, curve_right)
            ov.render_frame(
                frame,
                smoothed_left, smoothed_right,
                y_bottom, y_top,
                off, warning, side,
                frames_read, total,
                len(segments),
                left_state, right_state,
                draw_area=False,        # 직선 폴리곤 스킵 (곡선으로 대체)
                draw_lines=False,       # 직선 차선선 스킵 (곡선으로 대체)
                curve_mode=curve_mode,
                curvature_r=curvature_r,
            )
        else:
            # STRAIGHT(fallback) 모드: 기존 직선 파이프라인 그대로
            ov.render_frame(
                frame,
                smoothed_left, smoothed_right,
                y_bottom, y_top,
                off, warning, side,
                frames_read, total,
                len(segments),
                left_state, right_state,
                curve_mode=curve_mode,  # "STRAIGHT(fallback)" 또는 None
            )

        # Bird-eye PiP 합성 (render_frame 이후, 오버레이된 프레임 우상단에 삽입)
        ov.draw_birdeye_pip(frame, warped_frame)

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

    # M6 곡선 모드 집계 보고
    print("\n=== M6 곡선 모드 집계 ===")
    for mode_name, cnt in curve_frame_counts.items():
        pct = cnt / frames_read * 100 if frames_read else 0.0
        print(f"  {mode_name:25s}: {cnt:5d}프레임 ({pct:.1f}%)")

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
