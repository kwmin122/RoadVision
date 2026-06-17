"""PPT용 단계별 시각화 생성(슬라이스 아님). solidYellowLeft 한 프레임의 파이프라인 단계를
presentation/assets/ 에 깔끔한 PNG로 저장."""
from __future__ import annotations
import os
import cv2
import numpy as np
from src import config, preprocess, roi, lane_detect, birdeye

CLIP = "solidYellowLeft"
FRAME = 130
OUT = "presentation/assets"


def grab(clip, n):
    cap = cv2.VideoCapture(config.CLIPS[clip]["path"])
    i = 0
    f = None
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        i += 1
        if i == n:
            f = fr
            break
    cap.release()
    return f


def main():
    os.makedirs(OUT, exist_ok=True)
    f = grab(CLIP, FRAME)
    W = config.CLIPS[CLIP]["width"]; H = config.CLIPS[CLIP]["height"]

    cv2.imwrite(f"{OUT}/st01_input.png", f)

    cmask = preprocess.color_mask(f)
    cv2.imwrite(f"{OUT}/st02_colormask.png", cv2.cvtColor(cmask, cv2.COLOR_GRAY2BGR))

    edges = preprocess.to_edges(f)
    cv2.imwrite(f"{OUT}/st03_canny.png", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    lmask = preprocess.lane_mask(f)
    cv2.imwrite(f"{OUT}/st04_lanemask.png", cv2.cvtColor(lmask, cv2.COLOR_GRAY2BGR))

    roi_l = roi.apply(lmask, W, H, CLIP)
    segs = lane_detect.raw_segments(roi_l)
    hough = f.copy()
    lane_detect.draw_segments(hough, segs)
    poly = roi.polygon(W, H, CLIP)
    cv2.polylines(hough, [poly], True, (0, 255, 0), 2)
    cv2.imwrite(f"{OUT}/st05_hough.png", hough)

    # 버드아이 워프(마스크)
    warped = birdeye.warp(lmask, CLIP)
    cv2.imwrite(f"{OUT}/st06_birdeye_mask.png", cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR))

    print("저장 완료:", os.listdir(OUT))


if __name__ == "__main__":
    main()
