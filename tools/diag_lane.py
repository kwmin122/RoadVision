"""진단 도구(슬라이스 아님): 특정 프레임에서 우측 차선 검출이 왜 실패하는지 분해.
각 프레임에 대해 color_mask / lane_mask(ROI 후) / 우측 선분 수·끝점 수를 출력·저장.
사용: python -m tools.diag_lane --clip solidYellowLeft --frames 5 9 18 43
"""
from __future__ import annotations
import argparse, os
import cv2
import numpy as np
from src import config, preprocess, roi, lane_detect


def roi_top_y(clip, H):
    key = config.res_key(clip)
    ys = [ry for _, ry in config.ROI_TRAPEZOID_RATIO[key]]
    return int(min(ys) * H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="solidYellowLeft")
    ap.add_argument("--frames", type=int, nargs="+", default=[5, 9, 18, 43])
    args = ap.parse_args()
    clip = args.clip
    cap = cv2.VideoCapture(config.CLIPS[clip]["path"])
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs("frames/diag", exist_ok=True)
    want = set(args.frames)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        idx += 1
        if idx not in want: continue
        cmask = preprocess.color_mask(frame)
        lmask = preprocess.lane_mask(frame)
        roi_l = roi.apply(lmask, W, H, clip)
        roi_c = roi.apply(cmask, W, H, clip)
        segs = lane_detect.raw_segments(roi_l)
        left, right = lane_detect.split_segments(segs)
        # 우측 끝점 수
        rpts = len(right) * 2
        print(f"[frame {idx}] segs={len(segs)} left={len(left)} right={len(right)} "
              f"right_points={rpts} (FIT_MIN_POINTS={config.FIT_MIN_POINTS}) "
              f"-> right_fit={'OK' if rpts>=config.FIT_MIN_POINTS else 'FAIL(None)'}")
        # 우측 색마스크 픽셀(ROI내) 양
        print(f"          color_mask ROI 우측영역 픽셀수={int((roi_c[:, W//2:]>0).sum())}, "
              f"lane_mask ROI 우측 픽셀수={int((roi_l[:, W//2:]>0).sum())}")
        # 시각화 3-up: lane_mask ROI에 우측 선분 그려서 저장
        vis = cv2.cvtColor(roi_l, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in right:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(f"frames/diag/f{idx}_lanemask_roi.png", roi_l)
        cv2.imwrite(f"frames/diag/f{idx}_colormask_roi.png", roi_c)
        cv2.imwrite(f"frames/diag/f{idx}_right_segs.png", vis)
    cap.release()


if __name__ == "__main__":
    main()
