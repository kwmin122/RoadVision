"""
Slice 1 — 영상 I/O 하네스 엔드-투-엔드 통과 검증.
차선 검출 없음. 프레임별 HUD 텍스트만 삽입해 루프 동작을 증명한다.
"""
from __future__ import annotations

import argparse
import os
import time

import cv2

from src import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoadVision — Slice 1 passthrough")
    parser.add_argument(
        "--clip",
        choices=list(config.CLIPS.keys()),
        default="solidYellowLeft",
        help="처리할 클립 키 (config.CLIPS 기준)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip_key = args.clip
    clip_cfg = config.CLIPS[clip_key]

    input_path = clip_cfg["path"]

    # 출력 디렉토리 생성
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
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

        # HUD: 현재 프레임 / 총 프레임 — 루프 동작 증명용 최소 드로잉
        label = f"frame {frames_read}/{total}"
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
