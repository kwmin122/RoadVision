"""
Slice 2 — 원시 차선 오버레이 (grayscale → Canny → ROI → Hough).

각 프레임에 대해:
  1. 전처리(gray→blur→Canny) → edges
  2. ROI 마스킹 → roi_edges
  3. HoughLinesP → raw segments
  4. 빨간 선분 오버레이
  5. HUD 유지
  6. 결과 mp4 출력

디버그: --debug-frame N 으로 지정한 프레임에서
  frames/slice2_edges.png, frames/slice2_roi_edges.png, frames/slice2_overlay.png 저장.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoadVision — Slice 2 raw lane overlay")
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

    frames_read = 0
    frames_written = 0
    t_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_read += 1
        W, H = width, height

        # 1. 전처리: grayscale → GaussianBlur → Canny
        edges = preprocess.to_edges(frame)

        # 2. ROI 마스킹
        roi_edges = roi.apply(edges, W, H, clip_key)

        # 3. Hough 선분 검출
        segments = lane_detect.raw_segments(roi_edges)

        # 4. 선분 오버레이 (빨간색)
        lane_detect.draw_segments(frame, segments)

        # 5. 디버그 프레임 저장 (지정 프레임에서만)
        if frames_read == debug_frame_idx:
            # edges 저장
            cv2.imwrite(
                os.path.join(config.FRAMES_DIR, "slice2_edges.png"),
                edges,
            )
            # roi_edges 저장
            cv2.imwrite(
                os.path.join(config.FRAMES_DIR, "slice2_roi_edges.png"),
                roi_edges,
            )
            # overlay: ROI 폴리곤(초록) + 선분(빨간) 위에 ROI 경계 추가
            overlay_debug = frame.copy()
            roi_poly = roi.polygon(W, H, clip_key)
            cv2.polylines(overlay_debug, [roi_poly], True, (0, 255, 0), 2)
            cv2.imwrite(
                os.path.join(config.FRAMES_DIR, "slice2_overlay.png"),
                overlay_debug,
            )
            print(f"  [debug] 프레임 {frames_read}: 선분 {len(segments)}개 검출, 디버그 프레임 저장 완료")

        # 6. HUD: 프레임 카운터 + 선분 수
        label = f"frame {frames_read}/{total}  segs:{len(segments)}"
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
