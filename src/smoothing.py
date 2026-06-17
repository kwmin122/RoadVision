"""
차선 시간평활 모듈 — 슬라이스 4 (M3): 이상치 거부 + 검출 유지 + 상태 로깅.

규칙: magic number 없음. 모든 임계·윈도우는 config에서 가져옴.
상태 문자열은 VALID_STATES 집합에 정확히 정의됨. 다른 값이 나오면 버그.

-- 차선 표현 --
  fit = (x_bottom, x_top): y_bottom(= H) 과 y_top(= ROI 상단) 에서의 x좌표.
  slope_proxy = x_top - x_bottom  (픽셀 단위 기울기 대리값, 양수/음수 모두 가능)

-- 이상치 정규화 공식 (OUTLIER_SLOPE_DEV) --
  s_candidate = x_top - x_bottom  (이번 프레임 기울기 대리값)
  s_mean      = deque 평균 slope_proxy
  eps = 1.0   (zero-division 방지)

  normalized_dev = |s_candidate - s_mean| / (|s_mean| + eps)

  if normalized_dev > config.OUTLIER_SLOPE_DEV → 이상치(rejected_as_outlier)
  else → 통과(raw_detected)

  근거: |s_mean|으로 정규화하면 기울기 자체 크기에 무관하게 상대적 편차를 측정.
  OUTLIER_SLOPE_DEV = 0.30 → 직전 평균 대비 30% 이상 변화 시 거부.
  차선이 완전히 수평에 가까운 경우 s_mean≈0 이 될 수 있으므로 eps 필수.

-- 상태별 동작 요약 --
  "raw_detected"        : 실제 픽셀 검출 + 이상치 통과. deque에 push. c_miss = 0.
  "rejected_as_outlier" : 검출했지만 이상치. deque push 안 함. c_miss++.
  "held_from_previous"  : None이지만 deque 있고 c_miss < HOLD_MAX_FRAMES. c_miss++.
  "consecutive_missing" : None이고 (deque 없음 OR c_miss >= HOLD_MAX_FRAMES). c_miss++.
"""
from __future__ import annotations

import collections
from typing import Literal

import numpy as np

from src import config

# 정확히 이 4개 문자열만 허용. 스펙 변경 시 여기 먼저 수정.
VALID_STATES = frozenset({
    "raw_detected",
    "rejected_as_outlier",
    "held_from_previous",
    "consecutive_missing",
})

StateStr = Literal[
    "raw_detected",
    "rejected_as_outlier",
    "held_from_previous",
    "consecutive_missing",
]

# 내부 수치 안정용
_EPS = 1.0


class _SideState:
    """단일 방향(left 또는 right)의 스무딩 상태를 관리."""

    def __init__(self) -> None:
        self.deque: collections.deque[tuple[int, int]] = collections.deque(
            maxlen=config.SMOOTH_WINDOW
        )
        self.consecutive_missing: int = 0
        self.reject_count: int = 0

    # ------------------------------------------------------------------
    # 이상치 검사
    # ------------------------------------------------------------------
    def _is_outlier(self, candidate: tuple[int, int]) -> bool:
        """
        deque가 비어 있으면 이상치 검사 불가 → 항상 통과(False 반환).
        비어 있지 않으면 정규화 공식으로 판정:
          normalized_dev = |s_cand - s_mean| / (|s_mean| + eps)
          > OUTLIER_SLOPE_DEV → 이상치
        """
        if not self.deque:
            return False  # cold-start: 비교할 기준 없음 → 이상치 아님

        s_cand = candidate[1] - candidate[0]  # x_top - x_bottom
        slopes = [t[1] - t[0] for t in self.deque]
        s_mean = float(np.mean(slopes))

        normalized_dev = abs(s_cand - s_mean) / (abs(s_mean) + _EPS)
        return normalized_dev > config.OUTLIER_SLOPE_DEV

    # ------------------------------------------------------------------
    # 현재 deque 평균 (반올림해서 (int, int) 반환)
    # ------------------------------------------------------------------
    def _mean_fit(self) -> tuple[int, int]:
        arr = np.array(self.deque, dtype=float)
        mean = arr.mean(axis=0)
        return int(round(mean[0])), int(round(mean[1]))

    # ------------------------------------------------------------------
    # 핵심 업데이트
    # ------------------------------------------------------------------
    def update(
        self, raw_fit: tuple[int, int] | None
    ) -> tuple[tuple[int, int] | None, StateStr]:
        """
        raw_fit: 이번 프레임 fit_lane() 결과 또는 None.
        반환: (output_fit_or_None, state_string)
        """
        if raw_fit is not None:
            if self._is_outlier(raw_fit):
                # ------ 이상치 거부 ------
                self.reject_count += 1
                self.consecutive_missing += 1  # 스펙: 이상치는 miss로 처리
                output = self._mean_fit() if self.deque else None
                return output, "rejected_as_outlier"
            else:
                # ------ 정상 검출 ------
                self.deque.append(raw_fit)
                self.consecutive_missing = 0   # 리셋
                output = self._mean_fit()
                return output, "raw_detected"
        else:
            # raw_fit is None
            if self.deque and self.consecutive_missing < config.HOLD_MAX_FRAMES:
                # ------ 이전 값 유지 ------
                self.consecutive_missing += 1
                output = self._mean_fit()
                return output, "held_from_previous"
            else:
                # ------ 연속 미검출 ------
                self.consecutive_missing += 1
                return None, "consecutive_missing"


class LaneSmoother:
    """
    좌(left) · 우(right) 각 1개의 _SideState를 독립 관리.

    사용법:
        smoother = LaneSmoother()
        out_left, left_state = smoother.update("left", raw_left)
        out_right, right_state = smoother.update("right", raw_right)

    update()는 항상 (output_fit_or_None, state_str) 튜플을 반환.
    state_str ∈ VALID_STATES — 위반 시 AssertionError.
    """

    def __init__(self) -> None:
        self._sides: dict[str, _SideState] = {
            "left": _SideState(),
            "right": _SideState(),
        }

    def update(
        self, side: str, raw_fit: tuple[int, int] | None
    ) -> tuple[tuple[int, int] | None, StateStr]:
        """
        side: "left" | "right"
        raw_fit: fit_lane() 결과 또는 None
        반환: (smoothed_fit_or_None, state)
        """
        assert side in self._sides, f"알 수 없는 side: {side!r}"
        output, state = self._sides[side].update(raw_fit)
        assert state in VALID_STATES, f"허용되지 않는 상태: {state!r}"
        return output, state
