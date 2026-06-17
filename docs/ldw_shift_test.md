# C4 LDW 합성 이동 테스트 결과

- 기준 클립: `solidYellowLeft`  프레임: 130
- 기준 left_fit  (x_bottom, x_top): (151, 445)
- 기준 right_fit (x_bottom, x_top): (849, 525)
- W=960  H=540  car_center_x=480
- 실제 lane_pixel_width (baseline): 698 px
- warn_on=0.35  warn_off=0.25

## 부호 규약

| 항목 | 설명 |
|---|---|
| `offset = (W/2 − lane_center_x) / (lane_px_width/2)` | 정규화 오프셋 공식 |
| offset < 0 | 차량이 차선 중심보다 **왼쪽** → LEFT 이탈 |
| offset > 0 | 차량이 차선 중심보다 **오른쪽** → RIGHT 이탈 |
| shift_px > 0 | 차선이 오른쪽으로 이동 → 차량 상대적으로 왼쪽 → offset ↓ (음수) |
| shift_px < 0 | 차선이 왼쪽으로 이동 → 차량 상대적으로 오른쪽 → offset ↑ (양수) |

## 케이스별 결과

| shift_px | lane_center_x | lane_px_width | offset | \|offset\| | |offset|>warn_on? | WARN | SIDE |
|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| +0 | 500.0 | 698 | -0.0573 | 0.0573 | NO | OFF | — |
| -160 | 340.0 | 698 | +0.4011 | 0.4011 | YES | ON | RIGHT |
| -140 | 360.0 | 698 | +0.3438 | 0.3438 | NO | OFF | — |
| -80 | 420.0 | 698 | +0.1719 | 0.1719 | NO | OFF | — |
| +80 | 580.0 | 698 | -0.2865 | 0.2865 | NO | OFF | — |
| +140 | 640.0 | 698 | -0.4585 | 0.4585 | YES | ON | LEFT |
| +160 | 660.0 | 698 | -0.5158 | 0.5158 | YES | ON | LEFT |

### 해석

- **shift=0** : 차선 이동 없음 → offset ≈ 0 → 경고 OFF (예상 일치)
- **shift=±80** : |offset| ≈ 80/(half_width). warn_on=0.35 기준 ON/OFF 여부는 실제 차선폭에 의존 — 위 표 참조.
- **shift=±140** : 차량 베이스라인 오프셋(≈−0.057)이 있어 shift=-140 쪽이 warn_on(0.35)에 근접하거나 미달할 수 있음. shift=+140은 LEFT 이탈 ON 확인. ±160에서 양쪽 모두 ON 확인.
- **일관성 체크**: shift_px > 0 → offset 감소 → SIDE=LEFT; shift_px < 0 → offset 증가 → SIDE=RIGHT (위 규약과 일치)

## 히스테리시스 스윕 결과

오프셋을 0.0 → 0.5 → 0.0으로 변화시키면서 경고 전환점을 관찰.

- 경고 ON 임계: `warn_on = 0.35`
- 경고 OFF 임계: `warn_off = 0.25`

| 방향 | 오프셋 | WARN | SIDE |
|:---:|---:|:---:|:---:|
| up | 0.00 | OFF | — |
| up | 0.05 | OFF | — |
| up | 0.10 | OFF | — |
| up | 0.15 | OFF | — |
| up | 0.20 | OFF | — |
| up | 0.25 | OFF | — |
| up | 0.30 | OFF | — |
| up | 0.35 | OFF | — |
| up | 0.40 | ON | RIGHT |
| up | 0.45 | ON | RIGHT |
| up | 0.50 | ON | RIGHT |
| down | 0.50 | ON | RIGHT |
| down | 0.45 | ON | RIGHT |
| down | 0.40 | ON | RIGHT |
| down | 0.35 | ON | RIGHT |
| down | 0.30 | ON | RIGHT |
| down | 0.25 | ON | RIGHT |
| down | 0.20 | OFF | — |
| down | 0.15 | OFF | — |
| down | 0.10 | OFF | — |
| down | 0.05 | OFF | — |
| down | 0.00 | OFF | — |

### 히스테리시스 해석

- 오프셋 상승 시: `>0.35` 에서 경고 ON (즉 0.40 이상)
- 오프셋 하강 시: `<0.25` 에서 경고 OFF (즉 0.20 이하)
- 중간 구간 (0.25~0.35): 이전 상태 유지 (깜빡임 방지)

---
*생성: tools/ldw_shift_test.py*
