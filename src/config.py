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

# --- LDW (차선 이탈 경고) --- 물리 기반 휠-차선 거리 모델
#
# 가정 (§8 설계 문서와 동기화):
#   lane_width_m  = 3.7  : 미국 고속도로 표준 차선폭 (m)
#   vehicle_width_m = 1.8: 승용차 대표 차폭 (m)
#   카메라는 차량 횡방향 중심에 고정됨.
#
# 파생 상수:
#   half_lane = lane_width_m / 2 = 1.85 m
#   half_car  = vehicle_width_m / 2 = 0.90 m
#
# 정규화 오프셋(offset_norm) → 미터 변환:
#   lateral_m = |offset_norm| × half_lane   (차량 중심의 횡방향 편차, 미터)
#   wheel_to_line_m = (half_lane - half_car) - lateral_m
#                   = 0.95 - lateral_m
#   해석: 근접 휠에서 근접 차선까지의 여유 거리 (m). 음수 = 휠이 차선 침범.
#
# 3-state 판정 (wheel_to_line_m 기준, 우선순위: DANGER > CAUTION > SAFE):
#   DANGER  : wheel_to_line_m ≤ danger_dist_m (0.15)  [진입]
#             wheel_to_line_m ≤ danger_exit_dist_m (0.25)  [히스테리시스 유지]
#   CAUTION : wheel_to_line_m ≤ caution_dist_m (0.45)  [진입, DANGER 아닐 때]
#             wheel_to_line_m ≤ caution_exit_dist_m (0.55)  [히스테리시스 유지]
#   SAFE    : wheel_to_line_m > caution_exit_dist_m (0.55)  (CAUTION 래치 해제 후)
#
# fill_alpha: 주행영역 반투명 초록 폴리곤 불투명도 (0.0=완전 투명 / 1.0=완전 불투명).
#   0.3 = 원본 영상이 비치면서도 초록 영역이 식별 가능한 밸런스.
# banner_height: 경고 ON 시 화면 상단 빨강 배너 높이 (px). 50은 960×540에서 글자 1줄 여유.
#
# gauge_*: 화면 하단 게이지 (공통, 3-state 모두 표시)
#   gauge_height_px  : 게이지 바 높이 (px)
#   gauge_h_margin   : 좌우 여백 (px)
#   gauge_b_margin   : 하단 여백 — HUD 텍스트와 겹치지 않게 (px)
#   gauge_marker_r   : 차량 위치 마커 반경 (px)
#
# border_px: DANGER 플래시 테두리 두께 (px).
#
# 색상 (BGR):
#   caution_fill    : 황색 주행영역 채움
#   caution_strip   : 황색 상단 띠 배경
#   danger_fill     : 적색 주행영역 채움
#   danger_banner_bg: DANGER 배너 배경 (진한 적색)
#   danger_border   : 플래시 테두리 색
#
# LDW DEMO 드리프트 파라미터 (--ldw-demo 플래그 시에만 적용):
#   demo_drift_amplitude: 최대 드리프트 정규화 오프셋.
#     0.55 → 피크 wheel_to_line ≈ -0.07 m (DANGER 진입 0.15 m 기준 충분히 초과).
#             CAUTION 구간(0.15~0.45 m)도 통과함.
#   demo_drift_period_frames: 드리프트 1주기 프레임 수.
#     340 → ~27s 클립(680프레임)에서 정확히 2주기. LEFT → SAFE → RIGHT → SAFE.
LDW = {
    # ── 물리 기반 차량/차선 상수 ──
    "lane_width_m":   3.7,   # 미국 고속도로 표준 차선폭 (m)
    "vehicle_width_m": 1.8,  # 승용차 대표 차폭 (m)
    # ── 휠-차선 거리 임계 (m) ──
    "danger_dist_m":       0.15,  # DANGER 진입: wheel_to_line ≤ 0.15 m
    "danger_exit_dist_m":  0.25,  # DANGER 해제: wheel_to_line > 0.25 m (히스테리시스)
    "caution_dist_m":      0.45,  # CAUTION 진입: wheel_to_line ≤ 0.45 m
    "caution_exit_dist_m": 0.55,  # CAUTION 해제: wheel_to_line > 0.55 m (히스테리시스; SAFE 복귀)
    # ── 레거시 호환 키 (읽기 전용, 상태 판정에 사용 안 함) ──
    "warn_on":       0.35,   # [레거시 — 물리 모델로 대체됨. 읽기 참조용 보존]
    "warn_off":      0.25,   # [레거시 — 물리 모델로 대체됨. 읽기 참조용 보존]
    "fill_alpha":    0.30,   # 주행영역 폴리곤 투명도 (addWeighted 에서 src1 가중치)
    "banner_height": 50,     # 경고 배너 높이 (px)
    "banner_alpha":  0.75,   # 경고 배너 배경 불투명도 (1−alpha = 배경 비침 정도)
                             # 0.75: 빨강 배너가 선명하면서도 하단 영상이 약간 비침
    # ── 3-state 위험 시각화 ──
    "caution_fill":        (0, 200, 255),   # 황색 주행영역 채움 (BGR: 순수 황색, 시안보다 선명)
    "caution_fill_alpha":  0.55,            # CAUTION 폴리곤 불투명도 — SAFE(0.30)보다 높여 선명하게
    "danger_fill":         (0, 0, 200),     # 적색 주행영역 채움 (BGR)
    "caution_strip_h":     44,              # CAUTION 상단 띠 높이 (px) — 텍스트 여유 증가
    "caution_strip_color": (0, 180, 255),   # 황색 상단 띠 배경 (BGR, 순수 황색)
    "caution_strip_alpha": 0.90,            # CAUTION 띠 불투명도 — 선명하게
    "danger_banner_bg":    (0, 0, 180),     # DANGER 대형 배너 배경 (BGR, 진한 적색)
    "danger_banner_h":     64,             # DANGER 배너 높이 (px) — 2줄 텍스트 여유
    "danger_border_color": (0, 0, 255),     # 플래시 테두리 색 (BGR, 순수 적색)
    "border_px":           16,             # DANGER 플래시 테두리 두께 (px)
    "danger_line_thick":   14,             # DANGER 이탈 차선선 두께 (px, 기본 8보다 두꺼움)
    # ── 오프셋 게이지 ──
    "gauge_height_px":     18,             # 게이지 바 높이 (px)
    "gauge_h_margin":      80,             # 게이지 좌우 여백 (px)
    "gauge_b_margin":      12,             # 게이지 하단 여백 (px, HUD 텍스트 위)
    "gauge_marker_r":      10,             # 차량 위치 마커 반경 (px)
    # ── LDW DEMO 드리프트 ──
    "demo_drift_amplitude":     0.55,      # 최대 드리프트 오프셋 (normalized)
    "demo_drift_period_frames": 340,       # 드리프트 1주기 프레임 수 (680프레임 클립 = 2주기)
}

# --- Bird-eye homography (9강) ---
# src: ROI 사다리꼴 4점 (tl, tr, br, bl) — ROI_TRAPEZOID_RATIO에서 직접 유도.
#   960×540: tl=(0.43×960, 0.62×540)=(412,334), tr=(0.57×960, 0.62×540)=(547,334),
#            br=(0.95×960, 1.00×540)=(912,540), bl=(0.10×960, 1.00×540)=(96,540)
#   1280×720: tl=(0.45×1280, 0.64×720)=(576,461), tr=(0.58×1280, 0.64×720)=(742,461),
#             br=(0.92×1280, 1.00×720)=(1178,720), bl=(0.12×1280, 1.00×720)=(154,720)
# dst: 25% 수평 여백을 둔 탑다운 직사각형 (차선이 수직·평행하게 펼쳐짐).
#   960×540: 좌=240(=0.25×960), 우=720(=0.75×960)
#   1280×720: 좌=320(=0.25×1280), 우=960(=0.75×1280)
BIRDEYE = {
    "solidYellowLeft": {
        "src": [(412, 334), (547, 334), (912, 540), (96, 540)],  # tl, tr, br, bl
        "dst": [(240, 0),   (720, 0),   (720, 540), (240, 540)],
    },
    "solidWhiteRight": {
        "src": [(412, 334), (547, 334), (912, 540), (96, 540)],  # same res/ROI
        "dst": [(240, 0),   (720, 0),   (720, 540), (240, 540)],
    },
    "project_video": {
        "src": [(576, 461), (742, 461), (1178, 720), (154, 720)],  # tl, tr, br, bl
        "dst": [(320, 0),   (960, 0),   (960, 720),  (320, 720)],
    },
}

# Bird-eye PiP (Picture-in-Picture) 설정
# PIP_WIDTH_RATIO: 원본 프레임 폭 대비 PiP 패널 폭 비율. 0.30 = 30%.
# PIP_BORDER: PiP 테두리 두께 (px).
# PIP_TOP_OFFSET: 배너와 겹침 방지를 위해 상단에서 띄우는 픽셀.
BIRDEYE_PIP = {
    "width_ratio":  0.30,   # 프레임 폭의 30% 크기로 PiP 축소
    "border_px":    2,      # 테두리 두께 (px)
    "top_offset":   55,     # LDW 배너 아래부터 시작 (banner_height=50보다 5px 아래)
    "border_color": (255, 255, 255),  # 테두리 색 BGR (흰색)
    "label_color":  (255, 255, 255),  # 레이블 텍스트 색 BGR
}

# --- 곡선 슬라이딩 윈도우 (M6, project_video) ---
SLIDING_WINDOW = {"n_windows": 9, "margin": 80, "minpix": 50}

# --- 곡선 유효성 검사 임계 (M6 fail-safe 게이트) ---
# 클립 이름 분기 없이 데이터 기반으로만 CURVE vs STRAIGHT(fallback) 결정.
#
# min_pixels_per_side: 폴리핏 최소 픽셀 수. 슬라이딩 윈도우가 이 수 미만이면
#   해당 측 None 반환 → is_valid() False → fallback.
#   50 = SLIDING_WINDOW["minpix"]와 같은 수준. 너무 낮으면 노이즈 폴리.
#
# min_lane_width_px: 탑다운 공간에서 최소 차선 폭 (픽셀).
#   두 폴리가 너무 가까우면 교차 직전이거나 오검 → fallback.
#   100px = 960×540 bird-eye 기준 약 20% 폭. 실측으로 확인.
#
# max_lane_width_px: 탑다운 공간에서 최대 차선 폭 (픽셀).
#   폭이 과도하게 넓으면 한쪽 폴리가 잘못된 픽셀에 끌린 것 → fallback.
#   600px = 960폭 기준 62.5%. 좌(240)~우(720)dst 범위 480px보다 여유.
#
# max_poly_a: 이차항 계수 |a| 최대값. 단위: x픽셀/y픽셀².
#   값이 클수록 심한 U자형. 실제 도로 곡선은 아주 완만함.
#   0.003 = 1280×720 클립에서 합리적 상한 (실측 도로 최대 곡률 참조).
#
# y_sample_count: 폴리 샘플링 포인트 수.
#   유효성 검사와 curved_lane_points에서 동일하게 사용.
#   50개면 y축 전체를 충분히 촘촘히 샘플링 (부드러운 곡선).
CURVE_VALIDITY = {
    "min_pixels_per_side": 500,    # 폴리핏 최소 픽셀 수
    "min_lane_width_px":   100,    # 탑다운 공간 최소 차선 폭 (px)
    "max_lane_width_px":   700,    # 탑다운 공간 최대 차선 폭 (px)
    "max_poly_a":          0.003,  # 이차항 계수 최대 절댓값 (과도한 곡률 거부)
    "y_sample_count":      50,     # 폴리 평가 샘플 수 (유효성 + 포인트 생성 공용)
}

# --- 차량 보너스 (M7, forward-ROI Haar) ---
# cascade_path: OpenCV 공식 Haar 자동차 cascade (전면/후면 뷰).
# scale_factor: 1.1 = 10% 축소 스텝. 느릴수록 정밀, 빠를수록 FP 증가.
#   → 1.05로 낮추면 더 많은 후보 나오나 FP 급증, 1.1이 FP/miss 균형점.
# min_neighbors: 3 = 같은 오브젝트에서 최소 겹침 횟수. 낮출수록 FP 증가.
#   → 2로 낮추면 ~41% 프레임에서 박스 뜨지만 대부분 road texture FP.
#   → 3이 실험 결과 균형점 (project_video에서 ~7% 프레임, 실제 차량 구간 포함).
# min_size: (40, 40) = 후보 박스 최소 크기 (px).
#   이동 카메라에서 먼 차량은 40×40 안팎. 20px 이하는 도로 균열 FP 증가.
# forward_roi_ratio: 전방 ROI [(좌상 x비율, y비율), (우하 x비율, y비율)].
#   x=0.30~0.70: 수평 중앙 40%만 탐색 (갓길·가드레일 FP 억제).
#   y=0.55~1.00: 수직 하단 45%만 탐색 (하늘·신호등 FP 제거).
#   1280×720 기준 → x=[384,896], y=[396,720] (512×324 px 서브이미지).
# BEFORE (초기값): cascade_path=None, scale_factor=1.1, min_neighbors=3, min_size 없음
# AFTER  (튜닝값): cascade_path="models/cars.xml", min_size=(40,40) 추가,
#                  forward_roi_ratio 동일 유지 (실측으로 차량 포함 확인)
VEHICLE = {
    "cascade_path":      "models/cars.xml",
    "scale_factor":      1.1,              # 기본 1.1 — FP/miss 균형
    "min_neighbors":     3,               # 3 → 7.1% 프레임 hit, 2 → FP 폭증
    "min_size":          (40, 40),        # 40px 이하 도로 균열 FP 억제
    "forward_roi_ratio": [(0.30, 0.55), (0.70, 1.00)],  # (좌상, 우하) 비율
}

# --- 출력 ---
OUTPUT_DIR = "output"
FRAMES_DIR = "frames"


def res_key(name: str) -> str:
    c = CLIPS[name]
    return f"{c['width']}x{c['height']}"
