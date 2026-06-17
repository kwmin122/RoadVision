"""진단(슬라이스 아님): 드리프트 주입 '없이' 실제 영상에서 LDW가 어떻게 동작하는지 파악.
정상 주행 클립에서 프레임별 offset_norm / wheel_to_line_m / lane_state를 집계해,
'차는 중앙인데 경고가 뜨는가?'를 데이터로 확인한다.
사용: python -m tools.diag_ldw_real --clip solidYellowLeft
"""
from __future__ import annotations
import argparse
import numpy as np
import cv2
from src import config, preprocess, roi, lane_detect, smoothing, departure


def run(clip):
    cap = cv2.VideoCapture(config.CLIPS[clip]["path"])
    W = config.CLIPS[clip]["width"]; H = config.CLIPS[clip]["height"]
    key = config.res_key(clip)
    y_top = int(min(ry for _, ry in config.ROI_TRAPEZOID_RATIO[key]) * H)
    sm = smoothing.LaneSmoother()
    offs = []; wtls = []; states = {"SAFE": 0, "CAUTION": 0, "DANGER": 0, "None": 0}
    dep = departure.DepartureState()
    n = 0
    while True:
        ret, f = cap.read()
        if not ret: break
        n += 1
        mask = roi.apply(preprocess.lane_mask(f), W, H, clip)
        segs = lane_detect.raw_segments(mask)
        L, R = lane_detect.split_segments(segs)
        lf = lane_detect.fit_lane(L, H, y_top)
        rf = lane_detect.fit_lane(R, H, y_top)
        outL, _ = sm.update("left", lf); outR, _ = sm.update("right", rf)
        off = departure.offset(outL, outR, W)
        if off is None:
            states["None"] += 1; continue
        wtl = departure.wheel_to_line_m(off)
        warn, side = dep.update(off)
        st = departure.lane_state(off, warn, dep.caution)
        offs.append(off); wtls.append(wtl); states[st] = states.get(st, 0) + 1
    cap.release()
    offs = np.array(offs); wtls = np.array(wtls)
    print(f"\n=== {clip} (드리프트 없음, 실제) — {n}프레임 ===")
    print(f"offset_norm: mean={offs.mean():+.3f}  min={offs.min():+.3f}  max={offs.max():+.3f}  std={offs.std():.3f}")
    print(f"  → 평균이 0에서 멀면 '차선중앙=화면중앙' 가정이 이 영상에서 편향됨(차는 중앙이어도 offset≠0)")
    print(f"wheel_to_line_m: mean={wtls.mean():.3f}  min={wtls.min():.3f}  max={wtls.max():.3f}")
    print(f"상태 분포: {states}")
    caution_danger = states.get('CAUTION',0)+states.get('DANGER',0)
    print(f"  → CAUTION+DANGER = {caution_danger}/{len(offs)} 프레임 "
          f"({100*caution_danger/max(len(offs),1):.1f}%)  [정상주행이면 0%여야 정직]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--clip", default="solidYellowLeft")
    a = ap.parse_args(); run(a.clip)
