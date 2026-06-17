# 계획명세서 — Classical CV 기반 대시캠 차선 인식 및 차선 이탈 경고 시스템

> 컴퓨터비전 기말 프로젝트 / 5분 발표(시연 포함) / **딥러닝 없이 classical CV 기법만 사용**
> 최상위 기준: **성능 안정성 · 완성도** (화려함보다 "무대에서 안 깨지는 완성품")

---

## 1. 프로젝트 개요

- **제목**: Classical CV 기반 대시캠 차선 인식 및 차선 이탈 경고(LDW) 시스템
- **한 줄 요약**: 딥러닝(YOLO 등) 없이 Canny·Hough·threshold·morphology·homography만으로 대시캠 영상에서 차선을 검출하고, 차선 중앙과 차량 중심을 비교해 차선 이탈을 실시간 경고하는 ADAS 기본 기능을 구현한다.
- **핵심 제약**: 모든 검출/추정은 classical CV로만. DL은 사용하지 않음(발표에서 "한계 인식" 논점으로 다룸).
- **정체성**: ADAS(차선유지 보조). **차선 검출이 주연**, 차량 검출은 보너스/한계분석.

## 2. 범위 (Scope)

> **성공 기준 2층 잠금** — 욕심(커브·차량)은 살리되 Core 붕괴를 막는다.
> **Core(=프로젝트 성공) = M0~M5.** M6=커브 심화 쇼케이스 게이트, M7=차량 실험, M8=발표자료. M6·M7 실패는 *전체 실패가 아님*.

### Core — 프로젝트 성공 판정 (M0~M5, 반드시)
1. 대시캠 영상 입력 → 프레임 처리 → **결과 영상은 반드시 생성**
2. 차선 검출 (흰색/노란색, **직선/약곡선** robust)
3. 차선 중앙 추정 + 차량(=화면) 중심과 비교
4. **차선 이탈 경고 (LDW)** — 오프셋 임계 초과 시 시각 경고
5. Bird-eye view (homography) 변환 + 차선 영역 overlay
6. 프레임 간 시간평활(temporal smoothing)로 깜빡임 제거 ← 완성도의 핵심

### Core-plus — 고급 쇼케이스 (M6, 넣되 게이트로 격리)
- **project_video 곡선 2차 폴리핏 쇼케이스**: bird-eye 워프 → 슬라이딩 윈도우 히스토그램 → 2차 폴리핏 → 곡률반경. 9강 homography 심화로 프레이밍.
- **Fail-safe(필수)**: 곡선 폴리핏이 실패/불안정해도 **기존 Hough/스무딩 차선 overlay로 결과 영상은 반드시 생성**. 즉 M6는 Core 위에 *덧붙는* 레이어이고, 깨져도 Core 산출물은 그대로.

### Bonus — 차량 후보 **시각화 실험** (M7, 성공정의 낮춤 + timebox)
- 명칭은 "차량 검출기"가 아니라 **"전방 차량 후보 시각화 실험"**.
- forward-ROI 한정 Haar 차량 캐스케이드 + optical flow 시각화.
- **완료 기준(낮춤)**: project_video 특정 구간에서 *일부 프레임*에 전방 차량 후보 박스가 뜨면 OK + **false positive/missed 케이스를 캡처해 한계로 기록**. 최종 점수 핵심은 차선/LDW, 차량은 보너스임을 명시.
- **Timebox**: 2시간 안에 Haar/flow 후보 박스가 안 나오면 **optical flow heatmap + 한계 캡처로 종료**(본 일정 잠식 금지).

### Out of scope (명시적으로 안 함)
- YOLO/딥러닝 차량·차선 검출
- 차량 카운팅 (대시캠=이동카메라라 배경차분 불가 → 불안정)
- 카메라 내부 캘리브레이션 정밀 보정(왜곡계수 추정)은 선택, 기본은 생략

## 3. 입력 데이터 (이미 확보, `~/dev/adas_lane/clips/`)

| 클립 | 해상도/길이 | 특성 | 용도 |
|---|---|---|---|
| `solidYellowLeft.mp4` | 960×540 / 27s | 왼쪽 노란 실선 + 오른쪽 흰 점선, 곧은 도로 | ⭐**메인 데모** |
| `solidWhiteRight.mp4` | 960×540 / 9s | 양쪽 흰 점선, 타 차선 차량 | 검증 #2 |
| `project_video.mp4` | 1280×720 / 50s | HD, 노란실선+흰점선, **커브 + 앞차** | 커브/버드아이/차량보너스 쇼케이스 |

- 출처: Udacity CarND (MIT, 본 과제용 표준 벤치마크 = 라이선스 깨끗, 흰+노랑 둘 다 포함). 모두 **가공 안 된 원본 입력**(육안+확대로 확인 완료).
- 한계: 정답(GT) 차선 라벨 없음 → 정량 정확도 산출 불가. 검증은 **검출률 프록시 + 육안 + FPS**로 대체(§10).

## 4. 수업 기법 매핑 (PPT 직결)

| 강의 | 기법 | 본 프로젝트 사용처 |
|---|---|---|
| **6강** 특징추출 | Canny edge, Hough line(`HoughLinesP`) | 차선 엣지 추출 + 차선 선분 검출 (핵심) |
| **7강** 이진화·모폴로지 | HSV/threshold 색상필터, open/close | 흰/노랑 차선 후보 마스크 + 노이즈 제거 |
| **9강** 매칭 | homography, `perspectiveTransform` | Bird-eye view 변환 + 차선영역 워핑 |
| **10강** 추적 | frame-to-frame smoothing, optical flow | 차선 파라미터 시간평활(핵심) / 차량 보너스 |
| **8강** 검출 | contour, connectedComponents | (보너스) 차량 후보 영역 |

> 핵심 4강(6·7·9·10)을 하나의 실시간 파이프라인으로 자연스럽게 엮음. 억지로 다 넣지 않음.

## 5. 시스템 파이프라인 (단계별)

입력 프레임 → 아래 순서로 처리 → 오버레이 프레임 출력.

| # | 단계 | 알고리즘 / 주요 cv2 | 초기 파라미터(영상 맞춰 튜닝) | 산출물 |
|---|---|---|---|---|
| 1 | 전처리 | grayscale, `GaussianBlur` | ksize (5,5) | blurred gray |
| 2 | 색상필터 | HSV `inRange` 흰색+노란색 OR | W: V>200,S<40 / Y: H 15–35 | color mask |
| 3 | 엣지 | `Canny` | low=50, high=150 | edge map |
| 4 | 마스크 결합 | **기본: ROI 적용 후 `color_mask ∩ Canny`(`bitwise_and`)**. 실패 시에만 폴백 실험: Hough 입력으로 `color_mask ∪ edge`(`bitwise_or`) | 기본=교집합 | lane candidate mask |
| 5 | ROI | 사다리꼴 다각형 `fillPoly`+`bitwise_and` | 화면 하단 ~55%, 소실점 향함 | ROI mask |
| 6 | 선분 검출 | `HoughLinesP` | rho1, θ=π/180, thr=20, minLen=20, maxGap=300 | line segments |
| 7 | 좌우 분리 | slope 부호/범위 필터 | |slope|∈[0.5,2.0] | left/right segs |
| 8 | 차선 피팅 | 가중 1차 폴리핏(`polyfit`), ROI상·하단 외삽 | — | left/right line |
| 9 | **시간평활** | 최근 N프레임 러닝평균 + 이상치 거부 | N=10, dev 임계 | stable lines |
| 10 | 차선중앙·이탈 | (왼+오)/2 vs 화면중심, 오프셋→경고 | thr≈ lane폭의 X% | offset, warn flag |
| 11 | Bird-eye | `getPerspectiveTransform`+`warpPerspective` | **클립별 src/dst 4점을 `config.py`에 고정**(감으로 찍지 않음) | top-down view + **필수 산출물 `frames/{clip}_birdeye_debug.png`** |
| 12 | 오버레이 | 초록 차선폴리곤 `fillPoly`+`addWeighted`, 텍스트 | α=0.3 | 결과 프레임 |

**커브 대응(project_video)**: 9·11단계 후 bird-eye 워프 이미지에서 2차 폴리핏(advanced)으로 곡선 차선 처리 — *라벨링 "심화"*. 곧은 두 클립은 8단계 직선 피팅으로 충분(=안전 기본경로).

## 6. 모듈 / 파일 구조

```
~/dev/adas_lane/
├── PLAN.md                 # 본 명세서
├── requirements.txt        # opencv-python, numpy
├── clips/                  # 입력 원본 (확보됨)
├── output/                 # 결과 영상/프레임
├── frames/                 # 디버그 프레임
└── src/
    ├── config.py           # 클립별 ROI·임계·homography 4점 파라미터
    ├── preprocess.py       # gray/blur/색상필터/Canny  (6·7강)
    ├── roi.py              # 사다리꼴 ROI 마스킹
    ├── lane_detect.py      # Hough + 좌우분리 + 폴리핏  (6강)
    ├── smoothing.py        # 시간평활 + 이상치 거부     (10강)
    ├── departure.py        # 차선중앙·오프셋·LDW 로직
    ├── birdeye.py          # homography 워프            (9강)
    ├── overlay.py          # 차선폴리곤·경고·HUD 렌더
    ├── vehicle_bonus.py    # (보너스) 차량 후보 + optical flow
    ├── metrics.py          # 검출상태 로깅(CSV)·C2/C5 집계·C4 합성이동 테스트
    └── main.py             # 파이프라인 오케스트레이션 + 영상 I/O
```

산출물(검증용): `output/{clip}_overlay.mp4`, `output/{clip}_detect_log.csv`(raw/held/rejected/missing), `output/{clip}_metrics.txt`(C2·C5), `frames/{clip}_birdeye_debug.png`, `output/ldw_shift_test.md`(C4 ±80/±140px 표).

## 7. 안정성 설계 ("잘 구현" = 여기서 점수 갈림)

1. **ROI 마스킹**: 도로 외(하늘·갓길·타 차량) 엣지 차단 → Hough 오검 급감.
2. **시간평활**: 차선 파라미터를 최근 N=10프레임 러닝평균. → 깜빡임/점프 제거.
3. **이상치 거부**: 현재 프레임 피팅이 직전 평균 대비 기울기/절편 임계 초과 시 **버리고 직전 값 유지**(검출 실패 프레임 방어).
4. **검출 유지(hold)**: 한 프레임 차선 놓쳐도 직전 안정값을 몇 프레임 유지(점선 구간 대응).
5. **경고 히스테리시스**: 위험 경고 진입/해제에 다른 거리 임계(DANGER 진입 바퀴–차선 ≤0.15m / 해제 >0.25m, §8) → 깜빡이는 경고 방지.
6. **클립별 config 허용 — 단 과적합으로 정직 보고**: 클립별 ROI/임계/4점을 `config.py`에 두되 (a)감으로 찍지 말고 실제 프레임 보고 설정, (b)**튜닝값을 발표/리포트에 공개**, (c)일반화 성능은 "클립 과적합 한계"로 명시. *권장 절차: `solidYellowLeft`에서만 튜닝하고 `solidWhiteRight`·`project_video`는 미세조정 없이 테스트셋으로 두어 일반화를 보여줌.*
7. **검출 상태 로깅(메트릭 조작 방지)**: 매 프레임 좌/우 차선을 4상태로 기록 — `raw_detected`(이번 프레임 실제 검출) / `held_from_previous`(놓쳐서 직전값 유지) / `rejected_as_outlier`(피팅했으나 이상치로 폐기) / `consecutive_missing`(연속 미검출 카운트). **검출률 지표는 `raw_detected`로만 계산**(hold/reject로 부풀리지 않음). `output/{clip}_detect_log.csv`로 산출.

## 8. 차선 이탈 경고(LDW) 로직 — 물리 기반 휠-차선 거리 모델

### 오프셋 계산
- 차선중앙 `lane_center_x = (left_x + right_x) / 2` (이미지 하단 기준선에서)
- 차량중심 `car_center_x = W/2` (카메라가 차량 횡방향 중심에 장착됐다 가정)
- 정규화 오프셋 `offset_norm = (car_center_x − lane_center_x) / (lane_pixel_width/2)` → 범위 ≈ [-1, 1]
  - 음수: 차량이 차선 중심 대비 왼쪽 → LEFT 이탈 방향
  - 양수: 차량이 차선 중심 대비 오른쪽 → RIGHT 이탈 방향

### 물리 변환 — 휠-차선 거리 (m)

**가정** (config.LDW에 상수로 저장):
- `lane_width_m = 3.7 m` — 미국 고속도로 표준 차선폭 (AASHTO/MUTCD)
- `vehicle_width_m = 1.8 m` — 승용차 대표 차폭
- 카메라 = 차량 횡방향 중심에 고정

**파생 상수**:
- `half_lane = 3.7 / 2 = 1.85 m`
- `half_car  = 1.8 / 2 = 0.90 m`

**공식**:
```
lateral_m       = |offset_norm| × half_lane        (차량 중심의 횡방향 편차, m)
wheel_to_line_m = (half_lane − half_car) − lateral_m
                = 0.95 − lateral_m
```
- 해석: 근접 휠에서 근접 차선까지의 여유 거리 (m).
  - 양수: 여유 있음.  0: 휠이 차선에 닿음.  음수: 휠이 이미 차선 침범.
- 검증: 차량이 차선 중앙(`offset_norm=0`)이면 `wheel_to_line = 0.95 m` (SAFE).
  `offset_norm=±1`(차선 끝)이면 `wheel_to_line = 0.95 − 1.85 = −0.90 m` (심각 침범).

### 3-state 판정 임계

| 상태 | 진입 조건 | 해제/전환 |
|------|-----------|-----------|
| **DANGER** | `wheel_to_line ≤ 0.15 m` | `wheel_to_line > 0.25 m` (히스테리시스 래치) |
| **CAUTION** | `0.15 m < wheel_to_line ≤ 0.45 m` (DANGER 아닐 때) | — |
| **SAFE** | `wheel_to_line > 0.45 m` | — |

- 히스테리시스: DANGER는 진입(≤0.15) 후 해제(>0.25)까지 래치 유지 → 경계 근처 깜빡임 방지.
- `offset_norm = None` (한쪽 차선 미검출): 직전 상태 유지 (오N/오FF 방지).
- 이탈 방향(LEFT/RIGHT): `offset_norm` 부호로 결정 (음수=LEFT, 양수=RIGHT).

### 가정·한계 (adversarial review 반영)

> **(a) 미터 수치는 근사 추정값** — `wheel_to_line_m`은 차선폭 3.7 m 가정 아래 정규화 오프셋을 스케일링한 값으로, 실제 카메라 캘리브레이션(내부 파라미터·호모그래피) 없이 계산됨. 절대 거리계측이 아닌 상대적 LDW 지표로만 사용해야 함. HUD·게이지에 `~approx(uncalibrated)` 표기.
>
> **(b) 카메라 중심 가정** — `car_center_x = W/2`는 카메라가 차량 횡방향 정중앙에 장착됐다는 가정. 카메라 위치가 치우쳐 있으면 오프셋에 상수 편향이 생기고 휠-차선 거리가 과대/과소 추정됨.

### 베이스라인 보정 (편향 영점조정 — 정직성)
- **문제**: 실측 결과 정상 주행에서도 `offset_norm` 평균이 0이 아님(solidYellowLeft −0.08, project_video −0.15). 차는 중앙인데 시스템이 "치우침"으로 봐 가짜 CAUTION 발생(project_video 2.3%).
- **보정**: `BaselineCalibrator`가 **초기 `BASELINE_FRAMES=30`프레임(양 차선 검출 시)의 median offset = 상수 편향**을 추정해 이후 모든 오프셋에서 차감 → "정상 차로유지 ≈ 0". 보정 후 3클립 CAUTION/DANGER = 0.
- **가정·한계**: "차량이 초기 구간에 차로 중앙을 주행한다"를 전제. 30프레임 윈도가 클립 전체 편향을 과대추정할 수 있음(project_video 보정후 평균 +0.08, 최악 휠-차선 0.50m로 SAFE 유지). 이는 *상수 편향 제거*일 뿐 **이탈을 주입/위조하지 않음** — `raw_detected`·스무딩·CSV는 보정 전 값 사용해 무영향.

### 시각화
- 오프셋 게이지 및 HUD에 `wheel->line: X.XX m  ~approx(uncalibrated)` 및 `offset_m` 표시.
- SAFE=초록 / CAUTION=황색 영역(선명)+상단 황색 띠 / DANGER=적색 영역+플래시 테두리+대형 배너+방향 화살표.

## 9. 보너스(M7) — 전방 차량 후보 **시각화 실험** + 한계분석

- **명칭 주의**: "차량 검출기"라 하면 망함 → **"전방 차량 후보 시각화 실험"**으로 표현.
- 방법: forward-ROI 한정 Haar 차량 캐스케이드 우선 + optical flow 시각화(보조 shadow+contour).
- **완료 기준(낮춤)**: project_video 특정 구간에서 *일부 프레임*에 전방 차량 후보 박스 표시 → OK. 동시에 **false positive / missed 케이스를 캡처해 한계로 기록**.
- **Timebox 2시간**: 그 안에 Haar/flow 박스가 안 나오면 **optical flow heatmap + 한계 캡처로 종료**(차선/LDW 일정 잠식 금지).
- 한계 정직 서술: 이동 카메라·조명·크기/가림 변화로 classical CV 검출 불안정 → DL(YOLO) 필요성으로 자연스럽게 연결. **최종 점수 핵심은 차선/LDW, 차량은 보너스**임을 발표에서 명시.

## 10. 성능·검증 기준 (verifiable success criteria)

- **C1 무결성**: 3개 클립 모두 끝까지 크래시 없이 처리 + 결과영상 생성.
- **C2 원시 검출률 (pass/fail = solidYellowLeft·solidWhiteRight 限定)**: 이 두 클립에서 좌우 차선 *둘 다* **`raw_detected`**(hold/reject 제외)인 프레임 비율 ≥ 90%. 함께 보고: hold율·reject율·최대 `consecutive_missing`. → smoothing/hold로 부풀리는 것 원천 차단(§7-7). **project_video는 곡선이라 C2 pass/fail에 포함하지 않고 raw 검출률을 별도 지표로만 보고**(M6 쇼케이스).
- **C3 안정성(육안)**: 시간평활 후 차선 오버레이가 프레임 간 점프/깜빡임 없이 부드러움 — 결과영상 직접 재생 확인.
- **C4 LDW 검증(합성 이동 테스트)**: Udacity 영상엔 실제 차선이탈 장면이 거의 없음 → **차선중앙을 인위적으로 좌우 평행이동시켜 경고 트리거를 검증**(`tools/ldw_shift_test.py`→`docs/ldw_shift_test.md`). 0px(경고 OFF)부터 이동량을 키우며 바퀴–차선 거리(§8 모델)가 0.15m 이하로 떨어질 때 경고 ON + 좌/우 방향이 맞는지 표로 기록. 거리 히스테리시스(진입 0.15m / 해제 0.25m)도 함께 확인.
- **C5 속도(성공기준 아님, 보고 지표)**: 데모는 사전 렌더이므로 실시간성은 필수 아님. 960×540 및 1280×720 각각 처리 FPS를 *측정·보고*만 함. 720p+optical flow에서 25fps 미달 가능 → 그대로 보고(숨기지 않음).
- **정직성**: GT 라벨 없음 → C2는 "원시 검출률 프록시"임을 명시, 정량 정확도는 한계로 보고. **메트릭 통과 ≠ 시각 완성**, 반드시 결과영상 육안 검증.

## 11. 마일스톤 (vibe-coding 연속 핸드오프, 각 단계 끝=결과 프레임 육안 검증)

**Core(=프로젝트 성공) = M0~M5 / Core-plus = M6 / Bonus = M7 / 발표 = M8.** M6·M7은 게이트(실패해도 Core 산출물 유지).

| M | 층위 | 내용 | 완료 기준(눈검증) |
|---|---|---|---|
| M0 | Core | venv+opencv, 영상 read/write 골격, config 스켈레톤 | 입력→무변환 출력 영상 생성 |
| M1 | Core | 전처리+색상필터+Canny+ROI | 차선만 하얗게 남은 마스크 프레임 확인 |
| M2 | Core | Hough+좌우분리+직선피팅 | 한 프레임에 좌우 차선 1쌍 그려짐 |
| M3 | Core | 시간평활+이상치거부 + 검출상태 로깅 | solidYellow 전 구간 깜빡임 없이 안정 + CSV 생성 |
| M4 | Core | 차선영역 오버레이 + LDW 오프셋·경고 | 초록 주행영역 + 이탈 경고 동작 |
| M5 | Core | Bird-eye(homography) 변환·뷰 | 정상 탑다운 차선 뷰 + birdeye_debug.png |
| — | — | **▲ 여기까지 = 프로젝트 성공. C1·C2(2클립)·C3·C4 충족 + 3클립 결과영상 생성.** | |
| M6 | Core-plus | **곡선 쇼케이스**: 버드아이 슬라이딩윈도우+2차 폴리핏(project_video). **Fail-safe: 실패 시 Hough 오버레이로 영상 생성** | project_video 커브 곡선 추종(되면). project_video raw검출률 *별도* 보고(C2 불포함) |
| M7 | Bonus | **차량 후보 시각화 실험**(2h timebox): forward-ROI Haar + optical flow | 일부 프레임 전방 차량 박스 + FP/missed 한계 캡처 |
| M8 | 발표 | HTML PPT + 대본(humanize) + 결과영상 데모컷 | 5분 발표 자료·데모 영상 완성(§12) |

## 12. 발표 산출물 (M8) — HTML PPT + 대본

### 형식·제약
- **PPT = HTML로 제작**(헤드리스 캡처/브라우저 발표 가능). 슬라이드 = 16:9 섹션. `presentation/index.html`.
- **분량**: 총 5분. 그중 **시연(데모 영상) 30초~1분** 포함.
- **핵심 요구**: *내가 어떤 기법을 어떻게 사용해서 만들었는지*를 슬라이드마다 구체적으로 설명. 각 단계 = "무슨 기법(6강 Canny 등) → 왜 → 어떻게 적용 → 중간결과 이미지". 추상적 나열 금지.
- **대본**(`presentation/script.md`): 슬라이드별 발화 대본 + 타이밍. 발표 그대로 읽을 수 있게.
- **말투**: 대본·슬라이드 텍스트는 **`/humanize`(humanize-korean) 스킬로 윤문** → AI 티 제거, 자연스러운 한국어. (먼저 내용 작성 → humanize 통과 → 최종)

### 디자인 (Corporate Blue 템플릿, 첨부 레퍼런스 기준)
- 메인 컬러 **로열 블루**(≈`#2F55A4`/`#2E5BBA`), 배경 화이트, 강조 솔리드 블루 박스.
- 좌측 세로 사이드바 라벨, 큰 볼드 산세리프 타이틀, 가는 화살표/구분선, 섹션 번호(01·02·03), 카드형 3열 레이아웃.
- 폰트: 산세리프(예: Pretendard/Noto Sans KR). 깔끔·여백 많은 코퍼릿 톤.
- 디자인 토큰을 `presentation/style.css`에 변수로 고정.

### 슬라이드 구성 (5분 배분)
1. (0:30) 타이틀 + 문제·목표 — 딥러닝 없이 classical CV로 ADAS 차선이탈경고
2. (0:40) 수업기법 매핑(§4) — "6·7·9·10강을 이렇게 썼다"
3. (1:30) **기법별 상세**: 단계마다 기법·적용법·중간결과 이미지(색상마스크→Canny→Hough→폴리핏→스무딩→LDW→버드아이→곡선)
4. (1:00) **시연** — 결과영상 재생(차선추적+이탈경고+버드아이+곡선)
5. (0:30) 안정성 설계(시간평활·이상치거부·검출상태로깅·히스테리시스) + 검증(C2 raw / C4 ±px)
6. (0:20) 차량 후보 실험 & classical CV **한계 정직 분석** → 마무리

## 12b. 협업 워크플로 (역할 분리 + 품질 게이트)

- **계획·설계·검수 = Opus(나)**. **구현 = Sonnet**(서브에이전트). 슬라이스마다: 내가 정밀 스펙 전달 → Sonnet 구현 → **내가 diff 검수**.
- **Sonnet 우회 방지 게이트(매 슬라이스 필수)**:
  1. 구현 후 **`git diff` 정독** — 명세 외 임계값 하드코딩·데이터 특화 분기·우회 없는지.
  2. **임계/상수 grep 대조**(before→after) — 합의 안 된 magic number 추가 금지([[feedback_sonnet_silent_tuning]]).
  3. **테스트 PASS ≠ 통과**. 반드시 **출력 영상/프레임 육안 검증**.
  4. 검출상태 로깅(§7-7)이 실제 raw 기준인지 확인(smoothing으로 못 부풀리게).
  5. 막히거나 설계 의심 시 **advisor 호출**.
- 단일 진실원 = 본 `PLAN.md`. 스펙 변경은 여기 먼저 반영 후 구현.

## 13. 리스크 & 대응

| 리스크 | 대응 |
|---|---|
| 커브에서 직선 Hough 부정확 | 곧은 클립을 메인, 커브는 bird-eye 심화로 분리 |
| 점선 구간 차선 끊김 | 검출 유지(hold) + 시간평활 |
| 라이브 데모 깨짐 | **결과영상 사전 렌더**해서 재생(영상 데모) |
| 그림자/노면색 변화 오검 | ROI + 색상필터 결합 + 이상치 거부 |
| 차량 보너스 욕심 | 보너스로 격리, 핵심 일정 우선 |

## 14. 가정 & 확정된 결정

- **가정**: 카메라는 차량 중앙·전방 고정. 주간·맑음. 차선폭 ≈3.7m(미터 환산 시).
- **확정된 결정** (2026-06-17):
  - D1. **데모 = 사전 렌더 결과영상 재생** (라이브 실행 안 함 → 무대 안정성 최우선).
  - D2. **커브 = 버드아이 2차 폴리핏 심화까지 구현** (§2-7). 곧은 클립 직선 Hough(M2~M4) 완성 후 곡선(M5~M6).
  - D3. **차량 검출 = 실제 구현** (forward-ROI Haar 우선). 발표에선 "구현 + classical CV 한계분석" 병기.
  - D4. 코드 주석·리포트 언어 = 한국어 통일.
