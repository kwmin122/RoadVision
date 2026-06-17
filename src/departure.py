"""
차선 이탈 경고(LDW) 로직 모듈 — Slice 5 (M4).

-- 부호 규약 (한 번 정하고 전체 일관성 유지) --
  lane_center_x = (left_x_bottom + right_x_bottom) / 2
  car_center_x  = W / 2  (카메라가 차량 중앙 고정 가정)
  lane_pixel_width = right_x_bottom - left_x_bottom

  offset = (car_center_x - lane_center_x) / (lane_pixel_width / 2)

  → offset < 0 : 차량이 차선 중심보다 왼쪽 → LEFT 이탈
  → offset > 0 : 차량이 차선 중심보다 오른쪽 → RIGHT 이탈

  직관: 차가 왼쪽으로 흘러가면 lane_center가 차량 오른쪽에 남으므로 (W/2 - center) < 0.

-- 물리 기반 휠-차선 거리 모델 --
  가정:
    lane_width_m   = 3.7 m  (미국 고속도로 표준)
    vehicle_width_m = 1.8 m (승용차 대표 차폭)
    카메라 = 차량 횡방향 중심에 고정.
  파생:
    half_lane = 1.85 m,  half_car = 0.90 m
    lateral_m = |offset_norm| × half_lane
    wheel_to_line_m = (half_lane - half_car) - lateral_m  =  0.95 - lateral_m
    → 양수: 근접 휠에서 근접 차선까지 남은 여유 (m).  음수: 이미 침범.

-- 3-state 판정 (wheel_to_line_m 기준, 히스테리시스 래치) --
  DANGER  : wheel_to_line_m ≤ danger_dist_m (0.15)        — 진입
            wheel_to_line_m ≤ danger_exit_dist_m (0.25)   — 히스테리시스 유지
  CAUTION : wheel_to_line_m ≤ caution_dist_m (0.45)       — 진입 (DANGER 아닐 때)
            wheel_to_line_m ≤ caution_exit_dist_m (0.55)  — 히스테리시스 유지 (DANGER 아닐 때)
  SAFE    : wheel_to_line_m > caution_exit_dist_m (0.55)  (CAUTION 래치 해제 후)
  상태 우선순위: DANGER > CAUTION > SAFE

-- None 처리 --
  offset이 None(한쪽 차선 미검출)이면 경고 상태를 변경하지 않고 직전 상태를 유지.
  이는 일시적 미검출로 인한 경고 오N/오FF 방지.
"""
from __future__ import annotations

from src import config


def lane_center_x(
    left_fit: tuple[int, int] | None,
    right_fit: tuple[int, int] | None,
) -> float | None:
    """
    좌/우 차선 피팅에서 이미지 하단 기준 차선 중심 x좌표를 반환.

    left_fit, right_fit: (x_bottom, x_top). 어느 한쪽이라도 None이면 None.
    """
    if left_fit is None or right_fit is None:
        return None
    return (left_fit[0] + right_fit[0]) / 2.0


def offset(
    left_fit: tuple[int, int] | None,
    right_fit: tuple[int, int] | None,
    W: int,
) -> float | None:
    """
    차량과 차선 중심 사이의 정규화 측방향 오프셋을 반환.

    offset = (car_center_x - lane_center_x) / (lane_pixel_width / 2)
    결과 범위: 대략 [-1, 1].  None이면 어느 한쪽 차선이 미검출.

    부호:
      음수(< 0) → 차량이 차선 중심보다 왼쪽 → LEFT 이탈
      양수(> 0) → 차량이 차선 중심보다 오른쪽 → RIGHT 이탈
    """
    center = lane_center_x(left_fit, right_fit)
    if center is None:
        return None
    if left_fit is None or right_fit is None:
        return None

    lane_pixel_width = right_fit[0] - left_fit[0]
    if lane_pixel_width <= 0:
        # 비정상적 검출(좌우 역전) → None
        return None

    car_center = W / 2.0
    return (car_center - center) / (lane_pixel_width / 2.0)


def lateral_offset_m(offset_norm: float) -> float:
    """
    정규화 오프셋에서 차량 중심의 횡방향 편차(부호 포함)를 미터로 반환.

    lateral_m = offset_norm × half_lane  (부호 유지: 음수=LEFT, 양수=RIGHT)

    ※ 근사값(미보정): 차선폭 3.7 m 가정 기반 스케일 추정이며 캘리브레이션된 계측값 아님.
    """
    half_lane = config.LDW["lane_width_m"] / 2.0  # 1.85 m
    return offset_norm * half_lane


def wheel_to_line_m(offset_norm: float) -> float:
    """
    정규화 오프셋에서 근접 휠과 근접 차선 사이의 거리(m)를 반환.

    모델:
      half_lane = lane_width_m / 2      (1.85 m)
      half_car  = vehicle_width_m / 2   (0.90 m)
      lateral_m = |offset_norm| × half_lane
      wheel_to_line_m = (half_lane - half_car) - lateral_m  =  0.95 - lateral_m

    양수: 여유 있음.  0: 휠이 차선에 닿음.  음수: 휠이 차선 침범.

    가정:
      lane_width_m   = config.LDW["lane_width_m"]   = 3.7 m
      vehicle_width_m = config.LDW["vehicle_width_m"] = 1.8 m
    """
    half_lane = config.LDW["lane_width_m"] / 2.0     # 1.85 m
    half_car  = config.LDW["vehicle_width_m"] / 2.0  # 0.90 m
    lateral_m = abs(offset_norm) * half_lane
    return (half_lane - half_car) - lateral_m


def lane_state(off: float | None, warning: bool, caution: bool = False) -> str:
    """
    3-state 차선 상태를 반환.  오버레이 색상 선택 및 데모 프레임 덤프 기준.

    SAFE    : CAUTION/DANGER 래치 모두 OFF, wheel_to_line > caution_exit_dist_m  (초록)
    CAUTION : caution 래치 ON (DANGER 아닐 때)  (황색)
    DANGER  : warning 래치 ON  (적색)

    상태 우선순위: DANGER > CAUTION > SAFE.

    Note: DANGER/CAUTION 판정은 각각의 래치값 기반.  wheel_to_line 재비교 안 함.
      이유: 히스테리시스 유지 구간에서 배너/테두리가 플리커하지 않게.

    off=None이고 래치가 모두 OFF인 경우 SAFE 반환 (미검출 보수).
    """
    if warning:
        return "DANGER"
    if caution:
        return "CAUTION"
    return "SAFE"


class DepartureState:
    """
    프레임별 오프셋을 입력받아 히스테리시스 경고 상태를 추적.

    물리 기반 판정 (우선순위: DANGER > CAUTION > SAFE):
      DANGER 래치:
        wheel_to_line_m(off) ≤ danger_dist_m (0.15)      → DANGER ON
        wheel_to_line_m(off) > danger_exit_dist_m (0.25) → DANGER OFF
        그 사이에서는 이전 DANGER 상태를 유지 (히스테리시스).
      CAUTION 래치 (DANGER 아닐 때 독립적으로 동작):
        wheel_to_line_m(off) ≤ caution_dist_m (0.45)      → CAUTION ON
        wheel_to_line_m(off) > caution_exit_dist_m (0.55) → CAUTION OFF
        그 사이에서는 이전 CAUTION 상태를 유지 (히스테리시스).

    offset이 None인 경우(차선 미검출 프레임)에는 상태를 변경하지 않음.
    """

    def __init__(self) -> None:
        self._warning: bool = False      # DANGER 래치
        self._caution: bool = False      # CAUTION 래치
        self._side: str | None = None    # "LEFT" | "RIGHT" | None

    @property
    def warning(self) -> bool:
        return self._warning

    @property
    def caution(self) -> bool:
        return self._caution

    @property
    def side(self) -> str | None:
        return self._side

    def update(
        self, off: float | None
    ) -> tuple[bool, str | None]:
        """
        오프셋을 입력해 (warning_bool, side_or_None)을 반환.

        off: departure.offset()의 반환값. None이면 상태 유지.
        반환:
          (True,  "LEFT")  — 왼쪽 이탈 경고 ON (DANGER)
          (True,  "RIGHT") — 오른쪽 이탈 경고 ON (DANGER)
          (False, None)    — DANGER 경고 OFF (CAUTION/SAFE는 .caution 프로퍼티 참조)
          (True,  기존)    — offset None이어서 직전 상태 유지

        CAUTION 상태는 .caution 프로퍼티로 별도 확인.
        lane_state(off, warning, caution) 에 두 래치를 모두 전달할 것.
        """
        if off is None:
            # 미검출: 이전 상태 그대로 유지
            return self._warning, self._side

        wtl           = wheel_to_line_m(off)
        danger_dist   = config.LDW["danger_dist_m"]        # 0.15 m — DANGER 진입
        danger_exit   = config.LDW["danger_exit_dist_m"]   # 0.25 m — DANGER 해제
        caution_dist  = config.LDW["caution_dist_m"]       # 0.45 m — CAUTION 진입
        caution_exit  = config.LDW["caution_exit_dist_m"]  # 0.55 m — CAUTION 해제

        # ── DANGER 래치 ──
        if wtl <= danger_dist:
            self._warning = True
            # 부호: 음수=LEFT, 양수=RIGHT (모듈 상단 부호 규약)
            self._side = "LEFT" if off < 0 else "RIGHT"
        elif wtl > danger_exit:
            self._warning = False
            self._side = None
        # else: DANGER 히스테리시스 구간 (danger_dist < wtl ≤ danger_exit) — 변경 없음

        # ── CAUTION 래치 (DANGER와 독립적으로 갱신) ──
        if wtl <= caution_dist:
            self._caution = True
        elif wtl > caution_exit:
            self._caution = False
        # else: CAUTION 히스테리시스 구간 (caution_dist < wtl ≤ caution_exit) — 변경 없음

        return self._warning, self._side
