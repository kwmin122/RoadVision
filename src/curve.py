"""
곡선 차선 검출 모듈 — M6 Core-plus (bird-eye 슬라이딩 윈도우 + 2차 폴리핏).

외부에서 사용하는 함수:
  find_lane_bases(warped_mask)          → (left_base_x, right_base_x)
  sliding_window_fit(warped_mask)       → (left_poly | None, right_poly | None)
  curvature_radius(poly, y_eval)        → float (픽셀 단위)
  curved_lane_points(left_poly, right_poly, H, clip) → (polygon, left_pts, right_pts)
  is_valid(left_poly, right_poly, warped_mask) → bool (유효성 검사)

규칙:
- 모든 파라미터는 config.SLIDING_WINDOW / config.CURVE_VALIDITY에서만 읽음.
- 클립 이름 분기 (clip == 'project_video' 등) 절대 금지.
  유효성 판단은 데이터 기반 검사(픽셀 수, 교차 여부 등)로만 결정.
- fail-safe: 슬라이딩 윈도우 실패 또는 유효성 미달 시 None을 반환해
  main.py가 직선 오버레이로 폴백하도록 보장.
"""
from __future__ import annotations

import numpy as np

from src import config
from src import birdeye


def find_lane_bases(warped_mask: np.ndarray) -> tuple[int, int]:
    """
    워프된 이진 마스크의 히스토그램으로 좌/우 차선 베이스 x좌표를 찾는다.

    방법:
      - 마스크 하단 절반의 열별 합산으로 히스토그램 생성.
      - 좌측 절반 (W//2 미만): argmax → left_base_x
      - 우측 절반 (W//2 이상): argmax + W//2 → right_base_x

    반환: (left_base_x, right_base_x)
    """
    H, W = warped_mask.shape[:2]
    mid_y = H // 2
    mid_x = W // 2

    histogram = warped_mask[mid_y:, :].sum(axis=0).astype(np.int64)

    left_base_x  = int(np.argmax(histogram[:mid_x]))
    right_base_x = int(np.argmax(histogram[mid_x:]) + mid_x)

    return left_base_x, right_base_x


def sliding_window_fit(
    warped_mask: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    슬라이딩 윈도우 방식으로 좌/우 차선 픽셀을 수집하고 2차 폴리핏을 적용한다.

    파이프라인:
      1. find_lane_bases()로 초기 베이스 x좌표 결정.
      2. 위에서 아래로 n_windows개 윈도우 슬라이딩.
         각 윈도우 내 비제로 픽셀 수 > minpix이면 평균 x로 재센터링.
      3. 수집된 (y, x) 좌표에 np.polyfit(ys, xs, 2)로 x = a*y^2 + b*y + c 피팅.
      4. 최소 픽셀 수 미달 시 해당 측 None 반환.

    파라미터: config.SLIDING_WINDOW (n_windows, margin, minpix)
              config.CURVE_VALIDITY["min_pixels_per_side"] (피팅 최소 픽셀 수)

    반환: (left_poly | None, right_poly | None)
      poly = np.array([a, b, c]) — x = poly[0]*y^2 + poly[1]*y + poly[2]
    """
    sw = config.SLIDING_WINDOW
    n_windows = sw["n_windows"]
    margin    = sw["margin"]
    minpix    = sw["minpix"]
    min_pixels = config.CURVE_VALIDITY["min_pixels_per_side"]

    H, W = warped_mask.shape[:2]
    window_height = H // n_windows

    # 비제로 픽셀 전체 좌표 (y, x)
    nonzero = warped_mask.nonzero()
    nzy = np.array(nonzero[0])
    nzx = np.array(nonzero[1])

    left_base_x, right_base_x = find_lane_bases(warped_mask)
    lx_current = left_base_x
    rx_current = right_base_x

    left_lane_inds:  list[np.ndarray] = []
    right_lane_inds: list[np.ndarray] = []

    for win in range(n_windows):
        # 각 윈도우의 y 범위 (위에서 아래 방향: win=0 → 맨 아래)
        y_lo = H - (win + 1) * window_height
        y_hi = H - win * window_height

        # 윈도우 x 범위
        lx_lo = lx_current - margin
        lx_hi = lx_current + margin
        rx_lo = rx_current - margin
        rx_hi = rx_current + margin

        # 비제로 픽셀 중 이 윈도우 안에 있는 인덱스
        good_left = np.where(
            (nzy >= y_lo) & (nzy < y_hi) &
            (nzx >= lx_lo) & (nzx < lx_hi)
        )[0]
        good_right = np.where(
            (nzy >= y_lo) & (nzy < y_hi) &
            (nzx >= rx_lo) & (nzx < rx_hi)
        )[0]

        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)

        # minpix 이상이면 평균으로 재센터링
        if len(good_left) > minpix:
            lx_current = int(np.mean(nzx[good_left]))
        if len(good_right) > minpix:
            rx_current = int(np.mean(nzx[good_right]))

    # 수집된 인덱스 합치기
    if left_lane_inds:
        left_all  = np.concatenate(left_lane_inds)
    else:
        left_all  = np.array([], dtype=int)

    if right_lane_inds:
        right_all = np.concatenate(right_lane_inds)
    else:
        right_all = np.array([], dtype=int)

    left_poly: np.ndarray | None = None
    right_poly: np.ndarray | None = None

    if len(left_all) >= min_pixels:
        left_ys = nzy[left_all]
        left_xs = nzx[left_all]
        try:
            left_poly = np.polyfit(left_ys, left_xs, 2)
        except (np.linalg.LinAlgError, ValueError):
            left_poly = None

    if len(right_all) >= min_pixels:
        right_ys = nzy[right_all]
        right_xs = nzx[right_all]
        try:
            right_poly = np.polyfit(right_ys, right_xs, 2)
        except (np.linalg.LinAlgError, ValueError):
            right_poly = None

    return left_poly, right_poly


def curvature_radius(poly: np.ndarray, y_eval: float) -> float:
    """
    2차 폴리핏 계수에서 곡률 반경(픽셀 단위)을 계산한다.

    공식: R = (1 + (2a*y + b)^2)^1.5 / |2a|
    poly = [a, b, c] — x = a*y^2 + b*y + c

    a == 0 이면 직선(R = 무한대) → 9999999.0 반환 (실용적 상한).
    """
    a, b, _ = float(poly[0]), float(poly[1]), float(poly[2])
    denom = abs(2.0 * a)
    if denom < 1e-8:
        return 9999999.0  # 직선에 가까움

    R = (1.0 + (2.0 * a * y_eval + b) ** 2) ** 1.5 / denom
    return R


def curved_lane_points(
    left_poly: np.ndarray,
    right_poly: np.ndarray,
    H: int,
    clip: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    폴리핏 계수로 탑다운 공간의 좌/우 차선 곡선 포인트를 생성하고,
    birdeye.unwarp_points()로 원근 공간으로 역변환한다.

    방법:
      - y = 0 ~ H-1 범위를 샘플링 (config.CURVE_VALIDITY["y_sample_count"]개).
      - 각 y에서 poly(y) = a*y^2 + b*y + c → x 계산.
      - 탑다운 포인트 배열을 unwarp_points로 원근 변환.
      - 폴리곤: 왼쪽 위에서 아래 + 오른쪽 아래에서 위 (시계 방향).

    반환:
      polygon  : shape (2N, 2) int32 — fillPoly용 전체 폴리곤
      left_pts : shape (N, 2) int32  — 왼쪽 차선 곡선 포인트 (polylines용)
      right_pts: shape (N, 2) int32  — 오른쪽 차선 곡선 포인트 (polylines용)
    """
    n_samples = config.CURVE_VALIDITY["y_sample_count"]
    ys = np.linspace(0, H - 1, n_samples)

    def _eval(poly: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
        a, b, c = poly
        return a * y_arr ** 2 + b * y_arr + c

    left_xs  = _eval(left_poly,  ys)
    right_xs = _eval(right_poly, ys)

    # 탑다운 포인트 — shape (N, 2)
    left_warped  = np.column_stack([left_xs,  ys]).astype(np.float32)
    right_warped = np.column_stack([right_xs, ys]).astype(np.float32)

    # 원근 역변환
    left_unwarped  = birdeye.unwarp_points(left_warped,  clip)
    right_unwarped = birdeye.unwarp_points(right_warped, clip)

    left_pts  = left_unwarped.astype(np.int32)
    right_pts = right_unwarped.astype(np.int32)

    # fillPoly 폴리곤: 왼쪽 위→아래, 오른쪽 아래→위
    polygon = np.concatenate([left_pts, right_pts[::-1]], axis=0)

    return polygon, left_pts, right_pts


def is_valid(
    left_poly:  np.ndarray | None,
    right_poly: np.ndarray | None,
    warped_mask: np.ndarray,
) -> bool:
    """
    슬라이딩 윈도우 결과의 유효성을 검사한다.

    통과 조건 (모두 충족해야 CURVE 모드 활성화):
      1. 양쪽 poly 모두 존재 (None이면 즉시 False).
      2. 샘플 y 범위에서 좌/우 차선이 교차하지 않음
         (left_x < right_x for 모든 샘플 y).
      3. 샘플 y 범위 내 최소 및 최대 차선 폭이 허용 범위 이내
         (config.CURVE_VALIDITY: min_lane_width_px, max_lane_width_px).
      4. 이차항 계수 |a| < config.CURVE_VALIDITY["max_poly_a"]
         (과도하게 구부러진 폴리 필터링).

    클립 이름을 사용하지 않음 — 데이터 기반 판단만.
    """
    if left_poly is None or right_poly is None:
        return False

    cv = config.CURVE_VALIDITY
    H = warped_mask.shape[0]
    n_samples = cv["y_sample_count"]
    ys = np.linspace(0, H - 1, n_samples)

    a_l, b_l, c_l = left_poly
    a_r, b_r, c_r = right_poly

    left_xs  = a_l * ys ** 2 + b_l * ys + c_l
    right_xs = a_r * ys ** 2 + b_r * ys + c_r

    # 1. 교차 검사: 모든 y에서 left_x < right_x
    if not np.all(left_xs < right_xs):
        return False

    # 2. 차선 폭 검사
    widths = right_xs - left_xs
    if widths.min() < cv["min_lane_width_px"]:
        return False
    if widths.max() > cv["max_lane_width_px"]:
        return False

    # 3. 이차항 크기 검사 (과도한 곡률 방지)
    if abs(a_l) > cv["max_poly_a"] or abs(a_r) > cv["max_poly_a"]:
        return False

    return True
