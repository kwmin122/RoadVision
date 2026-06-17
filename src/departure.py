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

-- 히스테리시스 --
  |offset| > warn_on   → 경고 ON
  |offset| < warn_off  → 경고 OFF
  그 사이 구간         → 직전 상태 유지 (깜빡임 방지)

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


def lane_state(off: float | None, warning: bool) -> str:
    """
    3-state 차선 상태를 반환.  오버레이 색상 선택 및 데모 프레임 덤프 기준.

    SAFE    : |offset| < warn_off (완전 안전, 초록)
    CAUTION : warn_off ≤ |offset| < warn_on (히스테리시스 유지 구간, 황색)
    DANGER  : warning latched ON (위험, 적색)

    Note: DANGER 판정은 warning 래치값 기반.  raw |off|>warn_on 재비교 안 함.
      이유: 히스테리시스 유지 구간에서 배너/테두리가 플리커하지 않게.
    """
    if warning:
        return "DANGER"
    if off is None:
        return "SAFE"
    warn_off = config.LDW["warn_off"]
    warn_on  = config.LDW["warn_on"]
    if warn_off <= abs(off) < warn_on:
        return "CAUTION"
    return "SAFE"


class DepartureState:
    """
    프레임별 오프셋을 입력받아 히스테리시스 경고 상태를 추적.

    warn_on  = config.LDW["warn_on"]   : |offset| 이 이상이면 경고 ON
    warn_off = config.LDW["warn_off"]  : |offset| 이 미만이면 경고 OFF
    그 사이에서는 이전 상태를 유지.

    offset이 None인 경우(차선 미검출 프레임)에는 상태를 변경하지 않음.
    """

    def __init__(self) -> None:
        self._warning: bool = False
        self._side: str | None = None  # "LEFT" | "RIGHT" | None

    @property
    def warning(self) -> bool:
        return self._warning

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
          (True,  "LEFT")  — 왼쪽 이탈 경고 ON
          (True,  "RIGHT") — 오른쪽 이탈 경고 ON
          (False, None)    — 경고 OFF
          (True,  기존)    — offset None이어서 직전 상태 유지
        """
        if off is None:
            # 미검출: 이전 상태 그대로 유지
            return self._warning, self._side

        abs_off = abs(off)
        warn_on  = config.LDW["warn_on"]
        warn_off = config.LDW["warn_off"]

        if abs_off > warn_on:
            self._warning = True
            # 부호: 음수=LEFT, 양수=RIGHT (모듈 상단 부호 규약)
            self._side = "LEFT" if off < 0 else "RIGHT"
        elif abs_off < warn_off:
            self._warning = False
            self._side = None
        # else: 히스테리시스 구간 — 변경 없음

        return self._warning, self._side
