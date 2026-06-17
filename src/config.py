"""
RoadVision 중앙 설정 — 모든 튜닝 파라미터의 단일 통제 지점.

규칙(중요):
- 파이프라인 어디서도 magic number를 새로 만들지 말 것. 모든 임계/상수는 여기서만 정의·import.
- 값은 '실제 프레임을 보고' 튜닝한다(깜깜이 금지). TODO-TUNE 표시는 슬라이스에서 채운다.
- 클립별 차이(해상도·ROI·homography 4점)는 CLIPS[name] 안에 둔다.
- 권장: solidYellowLeft에서만 튜닝하고 나머지는 테스트셋으로 둔다(일반화 보고용).

좌표계: 이미지 좌상단 원점, x=가로(0..W), y=세로(0..H). 모든 ROI/4점은 (x, y) 픽셀.
"""
from __future__ import annotations

# 입력 클립 레지스트리 (mp4는 clips/ 에 fetch_clips.sh로 받음)
CLIPS = {
    "solidYellowLeft": {            # 메인 데모 (C2 pass/fail 대상)
        "path": "clips/solidYellowLeft.mp4",
        "width": 960, "height": 540,
        "role": "main",
    },
    "solidWhiteRight": {            # 검증 #2 (C2 pass/fail 대상)
        "path": "clips/solidWhiteRight.mp4",
        "width": 960, "height": 540,
        "role": "test",
    },
    "project_video": {              # 커브/HD 쇼케이스 (C2 미포함, 별도 보고)
        "path": "clips/project_video.mp4",
        "width": 1280, "height": 720,
        "role": "showcase",
    },
}

# --- 전처리 / 색상필터 (7강) ---  TODO-TUNE: 실제 프레임 보고 확정
GAUSSIAN_KSIZE = (5, 5)
HSV_WHITE = {"lower": (0, 0, 200), "upper": (180, 40, 255)}    # V 높고 S 낮음
HSV_YELLOW = {"lower": (15, 80, 120), "upper": (35, 255, 255)} # H 15~35

# --- 엣지 (6강 Canny) ---
CANNY = {"low": 50, "high": 150, "aperture": 3}

# 마스크 결합 전략: 기본 = color ∩ canny (PLAN §5-4). 폴백은 슬라이스에서 실험.
MASK_COMBINE = "intersection"  # "intersection" | "union"

# --- 모폴로지 클로징 (7강) — 파선 차선 검출 강화 ---
# 파선 차선 문제: color ∩ Canny 교집합은 파선 interior(200~700px)를 버려 Hough 입력이
# 희박(~70px)해지고 HoughLinesP가 0~1 선분만 반환 → fit_lane None (FIT_MIN_POINTS=4 미달).
# 수정: color_mask에 모폴로지 클로징(파선 갭 브리지)을 적용한 뒤 팽창된 Canny와 AND.
# 이렇게 하면 파선 갭이 이어지고 동시에 실제 엣지와 교집합을 취해 팬텀 차선을 방지.
#
# CLOSE_KERNEL_DASH: 수직 편향 커널 — (width=3, height=15).
#   파선은 거의 수직이므로 세로로 긴 커널이 수직 갭을 닫으면서 가로 번짐을 최소화.
#   height=15 기준: 실측 960×540 클립에서 파선 갭 브리지 충분, 도로 크랙 불검출 확인.
CLOSE_KERNEL_DASH = (3, 15)  # (width, height) — cv2.getStructuringElement(MORPH_RECT, ...)

# CANNY_DILATE_KERNEL: Canny 팽창용 커널 — (5×5) 정방형.
#   클로징 후 color_mask와 AND 시 팽창된 Canny가 파선 body를 포함할 만큼 충분히 넓어야 함.
#   5×5는 Canny 선 주변 ~2px 팽창 → 파선 body 포함 + 도로 내부 텍스처 미포함.
CANNY_DILATE_KERNEL = (5, 5)  # (width, height) for dilating Canny edges before AND

# --- ROI 사다리꼴 (해상도별 비율, 화면 하단 도로영역) ---  TODO-TUNE
# (좌하, 좌상, 우상, 우하) 를 W,H 비율로. 실제 프레임 보고 조정.
ROI_TRAPEZOID_RATIO = {
    "960x540":  [(0.10, 1.00), (0.43, 0.62), (0.57, 0.62), (0.95, 1.00)],
    "1280x720": [(0.12, 1.00), (0.45, 0.64), (0.58, 0.64), (0.92, 1.00)],
}

# --- Hough (6강 HoughLinesP) ---  TODO-TUNE
HOUGH = {"rho": 1, "theta_deg": 1, "threshold": 20,
         "min_line_len": 20, "max_line_gap": 300}
SLOPE_RANGE = (0.5, 2.0)  # |기울기| 필터 (수평·수직 잡선 제거)

# --- 차선 피팅 (슬라이스 3) ---
# fit_lane에서 폴리핏 시 최소 끝점 수. 2선분 × 2끝점 = 4가 수학적 최소보다 보수적.
# 노이즈 단일 선분에서의 가짜 피팅 방지.
FIT_MIN_POINTS = 4

# --- 시간평활 / 이상치 거부 (10강) ---  TODO-TUNE
SMOOTH_WINDOW = 10                 # 러닝평균 프레임 수 N
OUTLIER_SLOPE_DEV = 0.30           # 직전평균 대비 기울기 편차 임계 → 초과시 reject
HOLD_MAX_FRAMES = 8                # 미검출 시 직전 안정값 유지 최대 프레임

# --- LDW (차선 이탈 경고) ---  TODO-TUNE  (히스테리시스: ON>0.35, OFF<0.25)
LDW = {"warn_on": 0.35, "warn_off": 0.25, "lane_width_m": 3.7}

# --- Bird-eye homography (9강) ---  TODO-TUNE: 클립별 4점 직접 확정
# src=원근 도로 사다리꼴 4점, dst=탑다운 직사각 4점. (x,y) 픽셀.
BIRDEYE = {
    "solidYellowLeft": {"src": None, "dst": None},   # TODO-TUNE
    "solidWhiteRight": {"src": None, "dst": None},    # TODO-TUNE
    "project_video":   {"src": None, "dst": None},    # TODO-TUNE
}

# --- 곡선 슬라이딩 윈도우 (M6, project_video) ---  TODO-TUNE
SLIDING_WINDOW = {"n_windows": 9, "margin": 80, "minpix": 50}

# --- 차량 보너스 (M7, forward-ROI Haar) ---  TODO-TUNE
VEHICLE = {"cascade_path": None, "scale_factor": 1.1, "min_neighbors": 3,
           "forward_roi_ratio": [(0.30, 0.55), (0.70, 1.00)]}  # (좌상, 우하) 비율

# --- 출력 ---
OUTPUT_DIR = "output"
FRAMES_DIR = "frames"


def res_key(name: str) -> str:
    c = CLIPS[name]
    return f"{c['width']}x{c['height']}"
