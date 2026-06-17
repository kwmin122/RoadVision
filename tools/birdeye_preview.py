"""버드아이 4점 사전설계(슬라이스 아님): ROI 사다리꼴 기반 src→dst 직사각 워프 미리보기.
Slice 6에 넘길 src/dst 후보를 시각 검증한다. config는 수정하지 않음.
사용: python -m tools.birdeye_preview --clip solidYellowLeft --frame 130
"""
from __future__ import annotations
import argparse
import cv2
import numpy as np
from src import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="solidYellowLeft")
    ap.add_argument("--frame", type=int, default=130)
    args = ap.parse_args()
    clip = args.clip
    cap = cv2.VideoCapture(config.CLIPS[clip]["path"])
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    key = config.res_key(clip)
    ratios = config.ROI_TRAPEZOID_RATIO[key]  # (좌하, 좌상, 우상, 우하)
    # src = ROI 사다리꼴 (좌상, 우상, 우하, 좌하 순으로 재배열)
    bl, tl, tr, br = [(rx * W, ry * H) for rx, ry in ratios]
    src = np.float32([tl, tr, br, bl])
    # dst = 직사각형 (가장자리 여백 20%)
    mx = 0.25 * W
    dst = np.float32([(mx, 0), (W - mx, 0), (W - mx, H), (mx, H)])

    frame = None
    idx = 0
    while True:
        ret, f = cap.read()
        if not ret: break
        idx += 1
        if idx == args.frame:
            frame = f; break
    cap.release()
    if frame is None:
        print("프레임 못 읽음"); return

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (W, H))

    # src 사다리꼴 표시한 원본
    annot = frame.copy()
    cv2.polylines(annot, [src.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.imwrite("frames/birdeye_preview_src.png", annot)
    cv2.imwrite("frames/birdeye_preview_warped.png", warped)
    print(f"[{clip}] W={W} H={H}")
    print("src(tl,tr,br,bl)=", [tuple(map(int, p)) for p in src])
    print("dst(tl,tr,br,bl)=", [tuple(map(int, p)) for p in dst])
    print("저장: frames/birdeye_preview_src.png, frames/birdeye_preview_warped.png")


if __name__ == "__main__":
    main()
