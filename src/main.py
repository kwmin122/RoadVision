"""
Slice 4 — 시간평활 + 이상치 거부 + 검출 상태 로깅 (M3).

각 프레임에 대해:
  1. 색상필터+Canny 결합 → lane_mask
  2. ROI 마스킹 → roi_mask
  3. HoughLinesP → raw segments
  4. 기울기 기준 좌/우 분리 (split_segments)
  5. 가중 폴리핏으로 좌/우 각 1개 직선 피팅 (fit_lane) → raw_left, raw_right
  6. LaneSmoother.update() → smoothed_left/right + left_state/right_state
  7. smoothed 차선선 오버레이 (draw_lane_lines)
  8. HUD: 좌/우 상태 표시
  9. CSV 행 기록 (frame, left_state, right_state)
  10. 결과 mp4 출력

종료 후:
  - output/{clip}_detect_log.csv 저장 (header + 1행/프레임)
  - 콘솔에 side별 상태 카운트 + raw_detected rate 출력

디버그: --debug-frame N 으로 지정한 프레임(기본=130)에서
  frames/slice4_colormask.png, frames/slice4_lanemask.png, frames/slice4_overlay.png 저장.
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
        description="RoadVision — Slice 4: temporal smoothing + outlier rejection + detect logging"
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
    csv_path = os.path.join(config.OUTPUT_DIR, f"{clip_key}_detect_log.csv")

    # 캡처 열기
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {input_path}")

    # 메타데이터를 캡처에서 직접 읽음 (하드코딩 금지)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"입력: {input_path}  해상도={width}x{height}  fps={fps:.2f}  총프레임={total}")

    # ROI 상단 y좌표 (차선선 외삽 범위)
    y_bottom = height
    y_top = _roi_y_top(width, height, clip_key)
    print(f"ROI y 범위: y_top={y_top}, y_bottom={y_bottom}")
    print(f"스무딩 설정: SMOOTH_WINDOW={config.SMOOTH_WINDOW}  "
          f"OUTLIER_SLOPE_DEV={config.OUTLIER_SLOPE_DEV}  "
          f"HOLD_MAX_FRAMES={config.HOLD_MAX_FRAMES}")

    # 스무더 초기화 (슬라이스 4 핵심)
    smoother = LaneSmoother()

    # CSV 버퍼 (메모리에 쌓아서 종료 시 한 번에 씀)
    csv_rows: list[tuple[int, str, str]] = []

    # 상태 카운터 (honesty gate: 반드시 4개 상태 문자열로만 집계)
    state_counts: dict[str, dict[str, int]] = {
        "left":  {s: 0 for s in VALID_STATES},
        "right": {s: 0 for s in VALID_STATES},
    }

    frames_read = 0
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

        # 5. 가중 폴리핏으로 각 1개 직선 피팅 (raw — 평활화 전)
        raw_left = lane_detect.fit_lane(left_segs, y_bottom, y_top)
        raw_right = lane_detect.fit_lane(right_segs, y_bottom, y_top)

        # 6. 시간평활 + 이상치 거부 → smoothed 출력 + 상태
        smoothed_left, left_state = smoother.update("left", raw_left)
        smoothed_right, right_state = smoother.update("right", raw_right)

        # 상태 카운트 누적
        state_counts["left"][left_state] += 1
        state_counts["right"][right_state] += 1

        # CSV 행 버퍼
        csv_rows.append((frames_read, left_state, right_state))

        # 7. smoothed 차선선 오버레이 (raw 대신 smoothed 출력만 그림)
        lane_detect.draw_lane_lines(frame, smoothed_left, smoothed_right, y_bottom, y_top)

        # 8. 디버그 프레임 저장 (지정 프레임에서만)
        if frames_read == debug_frame_idx:
            cmask = preprocess.color_mask(frame)
            cv2.imwrite(os.path.join(config.FRAMES_DIR, "slice4_colormask.png"), cmask)
            cv2.imwrite(os.path.join(config.FRAMES_DIR, "slice4_lanemask.png"), lmask)
            overlay_debug = frame.copy()
            roi_poly = roi.polygon(W, H, clip_key)
            cv2.polylines(overlay_debug, [roi_poly], True, (255, 0, 0), 2)
            cv2.imwrite(os.path.join(config.FRAMES_DIR, "slice4_overlay.png"), overlay_debug)
            print(
                f"  [debug] 프레임 {frames_read}: "
                f"segs={len(segments)}  L_raw={'OK' if raw_left else 'MISS'}({left_state})  "
                f"R_raw={'OK' if raw_right else 'MISS'}({right_state})"
            )

        # 9. HUD: 프레임 카운터 + 좌/우 스무딩 상태
        # 상태 문자열 축약 (화면 공간 절약)
        _abbr = {
            "raw_detected": "raw",
            "rejected_as_outlier": "rej",
            "held_from_previous": "hld",
            "consecutive_missing": "mis",
        }
        label = (
            f"frame {frames_read}/{total}  "
            f"L:{_abbr[left_state]}  R:{_abbr[right_state]}  "
            f"segs:{len(segments)}"
        )
        cv2.putText(
            frame, label, (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
        )

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
    # 총 프레임 수 = frames_read 로 분모 고정 (부풀리기 원천차단)
    # ------------------------------------------------------------------
    total_frames = frames_read

    print("\n=== 검출 상태 요약 (C2 메트릭) ===")
    for side in ("left", "right"):
        cnts = state_counts[side]
        raw = cnts["raw_detected"]
        rej = cnts["rejected_as_outlier"]
        hld = cnts["held_from_previous"]
        mis = cnts["consecutive_missing"]
        raw_rate = raw / total_frames * 100 if total_frames else 0.0

        print(f"\n  [{side.upper()}]")
        print(f"    raw_detected        : {raw:4d}  ({raw_rate:.1f}% — C2 기준)")
        print(f"    rejected_as_outlier : {rej:4d}  ({rej/total_frames*100:.1f}%)")
        print(f"    held_from_previous  : {hld:4d}  ({hld/total_frames*100:.1f}%)")
        print(f"    consecutive_missing : {mis:4d}  ({mis/total_frames*100:.1f}%)")
        print(f"    합계 검증           : {raw+rej+hld+mis} / {total_frames}"
              f"  {'OK' if raw+rej+hld+mis == total_frames else '*** MISMATCH ***'}")

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
