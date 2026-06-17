"""
Slice 3 — 견고한 단일 차선선 (색상필터 + 기울기 분리 + 가중 폴리핏).

각 프레임에 대해:
  1. 색상필터+Canny 결합 → lane_mask
  2. ROI 마스킹 → roi_mask
  3. HoughLinesP → raw segments
  4. 기울기 기준 좌/우 분리 (split_segments)
  5. 가중 폴리핏으로 좌/우 각 1개 직선 피팅 (fit_lane)
  6. 두꺼운 초록 선으로 오버레이 (draw_lane_lines)
  7. HUD: 좌/우 검출 여부 표시
  8. 결과 mp4 출력

디버그: --debug-frame N 으로 지정한 프레임(기본=130)에서
  frames/slice3_colormask.png, frames/slice3_lanemask.png, frames/slice3_overlay.png 저장.
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from src import config
from src import preprocess
from src import roi
from src import lane_detect


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
    parser = argparse.ArgumentParser(description="RoadVision — Slice 3 robust lane lines")
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

        # 5. 가중 폴리핏으로 각 1개 직선 피팅
        left_line = lane_detect.fit_lane(left_segs, y_bottom, y_top)
        right_line = lane_detect.fit_lane(right_segs, y_bottom, y_top)

        # 6. 두꺼운 초록 차선선 오버레이
        lane_detect.draw_lane_lines(frame, left_line, right_line, y_bottom, y_top)

        # 7. 디버그 프레임 저장 (지정 프레임에서만)
        if frames_read == debug_frame_idx:
            # color_mask 저장
            cmask = preprocess.color_mask(frame)
            cv2.imwrite(
                os.path.join(config.FRAMES_DIR, "slice3_colormask.png"),
                cmask,
            )
            # lane_mask 저장 (ROI 적용 전)
            cv2.imwrite(
                os.path.join(config.FRAMES_DIR, "slice3_lanemask.png"),
                lmask,
            )
            # overlay: ROI 폴리곤 경계 + 차선선 오버레이 저장
            overlay_debug = frame.copy()
            roi_poly = roi.polygon(W, H, clip_key)
            cv2.polylines(overlay_debug, [roi_poly], True, (255, 0, 0), 2)  # 파란 ROI 경계
            cv2.imwrite(
                os.path.join(config.FRAMES_DIR, "slice3_overlay.png"),
                overlay_debug,
            )
            print(
                f"  [debug] 프레임 {frames_read}: "
                f"segs={len(segments)}  left_segs={len(left_segs)}  right_segs={len(right_segs)}  "
                f"left={'OK' if left_line else 'MISS'}  right={'OK' if right_line else 'MISS'}  "
                f"디버그 프레임 저장 완료"
            )

        # 8. HUD: 프레임 카운터 + 좌/우 검출 여부
        left_status = "L:OK" if left_line else "L:--"
        right_status = "R:OK" if right_line else "R:--"
        label = f"frame {frames_read}/{total}  {left_status}  {right_status}  segs:{len(segments)}"
        cv2.putText(
            frame,
            label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frames_written += 1

    cap.release()
    writer.release()

    elapsed = time.perf_counter() - t_start
    avg_fps = frames_written / elapsed if elapsed > 0 else 0.0

    print("=== 처리 완료 ===")
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
