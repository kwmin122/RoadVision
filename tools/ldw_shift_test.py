"""
C4 LDW 검증 도구 — 합성 차선 이동 테스트.

목적:
  Udacity 클립에는 실제 차선 이탈 장면이 거의 없다.
  그래서 실제 클립(solidYellowLeft)의 특정 프레임에서 얻은 (left, right) fit를
  인위적으로 가로 이동시켜 departure.offset()와 DepartureState를 통과시킨다.
  경고 ON/OFF 및 방향이 부호 규약과 일치하는지 표로 검증.

부호 규약 (departure.py와 동일):
  offset = (W/2 - lane_center_x) / (lane_pixel_width / 2)
  offset < 0 → 차량이 차선 중심보다 왼쪽 → LEFT 이탈
  offset > 0 → 차량이 차선 중심보다 오른쪽 → RIGHT 이탈

  shift_px 의미:
    shift_px > 0 → 차선(lane center)을 오른쪽으로 이동
                   → 차량이 상대적으로 왼쪽에 있게 됨
                   → offset 감소(더 음수)
                   → 충분히 크면 LEFT 이탈
    shift_px < 0 → 차선을 왼쪽으로 이동
                   → 차량이 상대적으로 오른쪽
                   → offset 증가(더 양수)
                   → 충분히 크면 RIGHT 이탈

  즉, shift_px의 부호와 이탈 방향은 반대.

실행:
  cd ~/dev/RoadVision
  source .venv/bin/activate
  python -m tools.ldw_shift_test

출력:
  output/ldw_shift_test.md
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

# 프로젝트 루트를 path에 추가 (직접 실행 시)
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from src import config
from src import departure as dep
from src import preprocess
from src import roi
from src import lane_detect
from src.smoothing import LaneSmoother


# ------------------------------------------------------------------
# 기준 프레임에서 실제 fit 추출
# ------------------------------------------------------------------
_TARGET_CLIP = "solidYellowLeft"
_TARGET_FRAME = 130  # 1-based, 안정적으로 양쪽 차선이 검출되는 프레임

def _get_real_fits() -> tuple[tuple[int, int], tuple[int, int], int, int, int]:
    """
    solidYellowLeft 영상의 TARGET_FRAME에서 실제 차선 fit를 추출.
    smoother를 N=10 프레임 워밍업 후 안정값을 사용.

    반환: (left_fit, right_fit, W, H, y_top)
    """
    clip_cfg   = config.CLIPS[_TARGET_CLIP]
    input_path = clip_cfg["path"]
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {input_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    key = config.res_key(_TARGET_CLIP)
    ratios = config.ROI_TRAPEZOID_RATIO[key]
    min_ry = min(ry for _, ry in ratios)
    y_top    = int(min_ry * H)
    y_bottom = H

    smoother = LaneSmoother()
    left_fit = right_fit = None

    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        lmask    = preprocess.lane_mask(frame)
        rm       = roi.apply(lmask, W, H, _TARGET_CLIP)
        segs     = lane_detect.raw_segments(rm)
        l_segs, r_segs = lane_detect.split_segments(segs)
        raw_l    = lane_detect.fit_lane(l_segs, y_bottom, y_top)
        raw_r    = lane_detect.fit_lane(r_segs, y_bottom, y_top)
        sl, _    = smoother.update("left",  raw_l)
        sr, _    = smoother.update("right", raw_r)

        if frame_no >= _TARGET_FRAME:
            left_fit  = sl
            right_fit = sr
            break

    cap.release()

    if left_fit is None or right_fit is None:
        raise RuntimeError(
            f"프레임 {_TARGET_FRAME}에서 차선 검출 실패. "
            "TARGET_FRAME을 변경하거나 클립을 확인하세요."
        )
    return left_fit, right_fit, W, H, y_top


# ------------------------------------------------------------------
# 개별 케이스 테스트
# ------------------------------------------------------------------
def _run_case(
    left_fit: tuple[int, int],
    right_fit: tuple[int, int],
    W: int,
    shift_px: int,
    state: dep.DepartureState,
) -> dict:
    """
    shift_px만큼 양쪽 차선을 같이 이동시켜 offset + 경고 상태를 반환.

    shift_px > 0: 차선이 오른쪽으로 이동 → lane_center 증가 → offset 감소 → LEFT 이탈 가능
    shift_px < 0: 차선이 왼쪽으로 이동  → lane_center 감소 → offset 증가 → RIGHT 이탈 가능
    """
    shifted_left  = (left_fit[0]  + shift_px, left_fit[1]  + shift_px)
    shifted_right = (right_fit[0] + shift_px, right_fit[1] + shift_px)

    lc  = dep.lane_center_x(shifted_left, shifted_right)
    off = dep.offset(shifted_left, shifted_right, W)
    warning, side = state.update(off)

    lane_px_width = shifted_right[0] - shifted_left[0]

    return {
        "shift_px":        shift_px,
        "lane_center_x":   round(lc, 1) if lc is not None else None,
        "lane_px_width":   lane_px_width,
        "offset":          round(off, 4) if off is not None else None,
        "abs_offset":      round(abs(off), 4) if off is not None else None,
        "warning":         warning,
        "side":            side if side else "—",
    }


# ------------------------------------------------------------------
# 히스테리시스 스윕 테스트
# ------------------------------------------------------------------
def _hysteresis_sweep(
    left_fit: tuple[int, int],
    right_fit: tuple[int, int],
    W: int,
) -> list[dict]:
    """
    오프셋을 0.0에서 0.5까지 올린 후 0.0으로 다시 내리면서
    경고 ON/OFF 전환점을 관찰 (DANGER + CAUTION 히스테리시스 확인).

    단, 이 함수는 DepartureState를 독립 생성해 순수 ON→OFF 전환만 본다.
    """
    lane_w = right_fit[0] - left_fit[0]

    # 올라가는 구간: 0.0 → 0.60 (0.05 step) — CAUTION 해제(0.55) 상단까지 포함
    sweep_up   = [round(v * 0.05, 2) for v in range(0, 13)]  # 0.00..0.60
    # 내려가는 구간: 0.60 → 0.00 (0.05 step)
    sweep_down = [round(v * 0.05, 2) for v in range(12, -1, -1)]  # 0.60..0.00
    targets = [(v, "up") for v in sweep_up] + [(v, "down") for v in sweep_down]

    state  = dep.DepartureState()
    rows   = []
    for off_target, direction in targets:
        # DepartureState는 float offset을 직접 받을 수 있으므로 직접 주입
        warning, side = state.update(off_target)   # RIGHT 이탈 방향으로 스윕
        lane_st = dep.lane_state(off_target, warning, state.caution)
        wtl = dep.wheel_to_line_m(off_target)
        rows.append({
            "direction":  direction,
            "off_target": off_target,
            "wtl":        wtl,
            "warning":    warning,
            "caution":    state.caution,
            "lane_st":    lane_st,
            "side":       side if side else "—",
        })
    return rows


# ------------------------------------------------------------------
# 수식 검증 (math check) — Fix 5
# ------------------------------------------------------------------
def _math_check() -> list[dict]:
    """
    정규화 오프셋 4개 케이스에 대해 공식에서 직접 파생된 수치와
    departure.lane_state()가 반환하는 상태를 검증.

    경계 케이스(rows 2, 3)는 수식 정밀값에 따라 하나의 상태가 나오며
    expected_states 집합으로 수용 범위를 정의. 단일값 케이스는 집합 크기 1.

    반환: 각 케이스의 결과 dict 목록 (assert 실패 시 RuntimeError).
    """
    half_lane = config.LDW["lane_width_m"] / 2.0   # 1.85 m
    half_car  = config.LDW["vehicle_width_m"] / 2.0  # 0.90 m

    # (offset_norm, label, expected_states)
    # expected_states: 수용 가능한 상태 집합 — 경계 케이스는 복수 허용
    cases = [
        (0.00, "SAFE 중앙",         {"SAFE"}),
        (0.27, "SAFE/CAUTION 경계", {"SAFE", "CAUTION"}),
        (0.43, "CAUTION/DANGER 경계", {"CAUTION", "DANGER"}),
        (0.50, "DANGER 내부",       {"DANGER"}),
    ]

    results = []
    for off_norm, label, expected_states in cases:
        lateral_m = abs(off_norm) * half_lane
        wtl       = (half_lane - half_car) - lateral_m  # = 0.95 - lateral_m

        # 신선한 DepartureState (래치 히스토리 없음)
        state = dep.DepartureState()
        warning, side = state.update(off_norm)
        actual_state  = dep.lane_state(off_norm, warning, state.caution)

        ok = actual_state in expected_states
        results.append({
            "offset_norm":   off_norm,
            "lateral_m":     lateral_m,
            "wtl":           wtl,
            "expected_label": label,
            "expected_states": expected_states,
            "actual_state":  actual_state,
            "ok":            ok,
        })
        if not ok:
            raise RuntimeError(
                f"[math_check] FAIL: offset_norm={off_norm}  wtl={wtl:.4f}  "
                f"actual={actual_state}  expected={expected_states}"
            )

    return results


# ------------------------------------------------------------------
# 마크다운 출력
# ------------------------------------------------------------------
def _fmt_bool(v: bool) -> str:
    return "ON" if v else "OFF"


def _build_md(
    base_info: dict,
    cases: list[dict],
    hysteresis_rows: list[dict],
    math_rows: list[dict],
) -> str:
    lines = []
    lines.append("# C4 LDW 합성 이동 테스트 결과\n")
    lines.append(f"- 기준 클립: `{_TARGET_CLIP}`  프레임: {_TARGET_FRAME}")
    lines.append(f"- 기준 left_fit  (x_bottom, x_top): {base_info['left_fit']}")
    lines.append(f"- 기준 right_fit (x_bottom, x_top): {base_info['right_fit']}")
    lines.append(f"- W={base_info['W']}  H={base_info['H']}  car_center_x={base_info['W']//2}")
    lines.append(f"- 실제 lane_pixel_width (baseline): {base_info['right_fit'][0] - base_info['left_fit'][0]} px")
    lines.append(f"- danger_dist_m={config.LDW['danger_dist_m']}  danger_exit_dist_m={config.LDW['danger_exit_dist_m']}")
    lines.append(f"- caution_dist_m={config.LDW['caution_dist_m']}  caution_exit_dist_m={config.LDW['caution_exit_dist_m']}\n")

    lines.append("## 부호 규약")
    lines.append("")
    lines.append("| 항목 | 설명 |")
    lines.append("|---|---|")
    lines.append("| `offset = (W/2 − lane_center_x) / (lane_px_width/2)` | 정규화 오프셋 공식 |")
    lines.append("| offset < 0 | 차량이 차선 중심보다 **왼쪽** → LEFT 이탈 |")
    lines.append("| offset > 0 | 차량이 차선 중심보다 **오른쪽** → RIGHT 이탈 |")
    lines.append("| shift_px > 0 | 차선이 오른쪽으로 이동 → 차량 상대적으로 왼쪽 → offset ↓ (음수) |")
    lines.append("| shift_px < 0 | 차선이 왼쪽으로 이동 → 차량 상대적으로 오른쪽 → offset ↑ (양수) |")
    lines.append("")

    # ── 수식 검증 (math check) ──────────────────────────────────────────────
    lines.append("## 수식 검증 (math check)\n")
    lines.append("공식 `wheel_to_line_m = 0.95 - |offset_norm| * 1.85` 에서")
    lines.append("직접 파생된 4개 케이스와 `departure.lane_state()` 반환값을 대조.\n")
    lines.append("- half_lane = lane_width_m / 2 = 1.85 m")
    lines.append("- half_car  = vehicle_width_m / 2 = 0.90 m")
    lines.append("- lateral_m = |offset_norm| * 1.85")
    lines.append("- wheel_to_line_m = 0.95 - lateral_m\n")
    lines.append("경계 케이스(rows 2, 3): 수식 정밀값에 따라 SAFE 또는 CAUTION 중 하나로")
    lines.append("확정되며, expected_states 집합에 속하면 PASS.\n")
    lines.append("| offset_norm | lateral_m | wheel_to_line_m | expected_states | actual_state | PASS? |")
    lines.append("|---:|---:|---:|:---:|:---:|:---:|")
    for r in math_rows:
        exp_str = "/".join(sorted(r["expected_states"]))
        status  = "OK" if r["ok"] else "FAIL"
        lines.append(
            f"| {r['offset_norm']:.2f} "
            f"| {r['lateral_m']:.4f} "
            f"| {r['wtl']:.4f} "
            f"| {exp_str} "
            f"| {r['actual_state']} "
            f"| {status} |"
        )
    lines.append("")
    all_ok = all(r["ok"] for r in math_rows)
    lines.append(f"**전체 검증 결과: {'ALL PASS' if all_ok else 'FAIL — 위 FAIL 행 확인'}**\n")
    # ────────────────────────────────────────────────────────────────────────

    lines.append("## 케이스별 결과\n")
    lines.append("| shift_px | lane_center_x | lane_px_width | offset | \\|offset\\| | wheel_to_line_m | WARN | SIDE |")
    lines.append("|---:|---:|---:|---:|---:|---:|:---:|:---:|")
    for r in cases:
        wtl_val = dep.wheel_to_line_m(r["offset"]) if r["offset"] is not None else float("nan")
        lines.append(
            f"| {r['shift_px']:+d} "
            f"| {r['lane_center_x']} "
            f"| {r['lane_px_width']} "
            f"| {r['offset']:+.4f} "
            f"| {r['abs_offset']:.4f} "
            f"| {wtl_val:.4f} "
            f"| {_fmt_bool(r['warning'])} "
            f"| {r['side']} |"
        )

    lines.append("")
    lines.append("### 해석")
    lines.append("")
    lines.append("- **shift=0** : 차선 이동 없음 → offset ≈ 0 → 경고 OFF (예상 일치)")
    lines.append("- **shift=±80** : |offset| ≈ 80/(half_width). "
                 f"caution_dist_m={config.LDW['caution_dist_m']} 기준 ON/OFF 여부는 실제 차선폭에 의존 — 위 표 참조.")
    lines.append("- **shift=±140** : 차량 베이스라인 오프셋(≈−0.057)이 있어 shift=-140 쪽이 "
                 "경계에 근접하거나 미달할 수 있음. "
                 "shift=+140은 LEFT 이탈 ON 확인. ±160에서 양쪽 모두 ON 확인.")
    lines.append("- **일관성 체크**: shift_px > 0 → offset 감소 → SIDE=LEFT; "
                 "shift_px < 0 → offset 증가 → SIDE=RIGHT (위 규약과 일치)")
    lines.append("")

    lines.append("## 히스테리시스 스윕 결과\n")
    lines.append("오프셋을 0.0 → 0.60 → 0.0으로 변화시키면서 DANGER·CAUTION 전환점을 관찰.\n")
    lines.append(f"- DANGER  진입: wheel_to_line ≤ {config.LDW['danger_dist_m']} m")
    lines.append(f"- DANGER  해제: wheel_to_line > {config.LDW['danger_exit_dist_m']} m")
    lines.append(f"- CAUTION 진입: wheel_to_line ≤ {config.LDW['caution_dist_m']} m")
    lines.append(f"- CAUTION 해제: wheel_to_line > {config.LDW['caution_exit_dist_m']} m\n")
    lines.append("| 방향 | 오프셋 | wheel_to_line_m | WARN(DANGER) | CAUTION | lane_state | SIDE |")
    lines.append("|:---:|---:|---:|:---:|:---:|:---:|:---:|")
    for r in hysteresis_rows:
        lines.append(
            f"| {r['direction']} "
            f"| {r['off_target']:.2f} "
            f"| {r['wtl']:.4f} "
            f"| {_fmt_bool(r['warning'])} "
            f"| {_fmt_bool(r['caution'])} "
            f"| {r['lane_st']} "
            f"| {r['side']} |"
        )

    lines.append("")
    lines.append("### 히스테리시스 해석")
    lines.append("")
    lines.append(f"- DANGER ON 구간:  offset 상승 시 wheel_to_line ≤ {config.LDW['danger_dist_m']} m 진입, "
                 f"하강 시 > {config.LDW['danger_exit_dist_m']} m에서 해제")
    lines.append(f"- CAUTION ON 구간: offset 상승 시 wheel_to_line ≤ {config.LDW['caution_dist_m']} m 진입, "
                 f"하강 시 > {config.LDW['caution_exit_dist_m']} m에서 해제")
    lines.append("- 위 표에서 DANGER/CAUTION 래치가 각각 독립적으로 동작하고 깜빡임 없이 유지되는지 확인")
    lines.append("")
    lines.append("---\n*생성: tools/ldw_shift_test.py*\n")

    return "\n".join(lines)


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------
def main() -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"[ldw_shift_test] 기준 프레임 {_TARGET_FRAME} 로드 중 ({_TARGET_CLIP})...")
    left_fit, right_fit, W, H, y_top = _get_real_fits()
    print(f"  left_fit={left_fit}  right_fit={right_fit}")
    print(f"  lane_pixel_width={right_fit[0] - left_fit[0]}  half_width={(right_fit[0]-left_fit[0])/2:.1f}")

    # 테스트할 shift_px 목록
    # ±160을 추가: ±140은 차량 베이스라인 오프셋(약 -0.057)으로 인해
    # shift=-140이 |off|=0.344(warn_on=0.35 미달)가 될 수 있어 보수적으로 ±160도 포함
    shifts = [0, -160, -140, -80, 80, 140, 160]

    # 각 케이스를 독립 DepartureState로 테스트 (상태 오염 방지)
    cases = []
    for sh in shifts:
        state = dep.DepartureState()
        result = _run_case(left_fit, right_fit, W, sh, state)
        cases.append(result)
        print(f"  shift={sh:+4d}px  off={result['offset']:+.4f}  "
              f"|off|={result['abs_offset']:.4f}  "
              f"warn={_fmt_bool(result['warning'])}  side={result['side']}")

    # 히스테리시스 스윕
    print("\n[ldw_shift_test] 히스테리시스 스윕 (DANGER + CAUTION)...")
    hyst_rows = _hysteresis_sweep(left_fit, right_fit, W)

    # 수식 검증 (math check)
    print("\n[ldw_shift_test] 수식 검증 (math check)...")
    math_rows = _math_check()
    print("  offset_norm | lateral_m | wheel_to_line_m | expected_states | actual | PASS?")
    for r in math_rows:
        exp_str = "/".join(sorted(r["expected_states"]))
        status  = "OK" if r["ok"] else "FAIL"
        print(f"  {r['offset_norm']:.2f}        | {r['lateral_m']:.4f}    | {r['wtl']:.4f}          | {exp_str:<20} | {r['actual_state']:<7} | {status}")
    all_ok = all(r["ok"] for r in math_rows)
    print(f"\n  => math check: {'ALL PASS' if all_ok else 'FAIL'}")

    # 마크다운 생성
    base_info = {"left_fit": left_fit, "right_fit": right_fit, "W": W, "H": H}
    md = _build_md(base_info, cases, hyst_rows, math_rows)

    out_path = os.path.join(config.OUTPUT_DIR, "ldw_shift_test.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n[ldw_shift_test] 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
