# M7 전방 차량 후보 시각화 실험 (vehicle_experiment.md)

> 명칭: "차량 검출기"가 아닌 **전방 차량 후보 시각화 실험**.  
> 목적: classical CV 한계를 정직하게 드러내어 "왜 DL(YOLO)이 필요한가"로 연결.

---

## 실험 개요

- 방법: OpenCV Haar cascade (cars.xml, 자동차 학습) + 전방 ROI 제한 + Farneback 광류 크기 히트맵.
- 대상 클립: `project_video.mp4` (1280×720, 50s=1260프레임, HD, 백색 앞차 있는 구간).
- 실행: `python -m src.main --clip project_video --vehicle`
- 출력: `output/project_video_vehicle.mp4` (lane-only `_overlay.mp4`와 별도 파일, 기존 파일 보존).

---

## 파라미터 (최종)

| 파라미터 | 값 | 비고 |
|---|---|---|
| cascade_path | `models/cars.xml` | OpenCV 공식 차량 Haar cascade |
| scale_factor | 1.1 | 초기값 그대로 — FP/miss 균형점 |
| min_neighbors | 3 | 초기값 그대로 |
| min_size | (40, 40) | **추가(신규)** — 도로 균열/텍스처 FP 억제 |
| forward_roi_ratio | [(0.30, 0.55), (0.70, 1.00)] | 초기값 그대로 |

**BEFORE → AFTER 변경 사항:**
- `cascade_path`: `None` → `"models/cars.xml"` (필수 초기화)
- `min_size`: 없음(미정) → `(40, 40)` (추가 튜닝)
- 나머지: 초기값 유지 (실험 결과 적절)

**forward ROI 픽셀 범위 (1280×720 기준):**
- x = [384, 896], y = [396, 720] — 512×324 px 서브이미지
- 수평 중앙 40%만 탐색 → 갓길·가드레일 FP 억제
- 수직 하단 45%만 탐색 → 하늘·신호등·오버패스 FP 제거

---

## 실험 결과 수치 (project_video, 1260 프레임)

| 지표 | 값 |
|---|---|
| 박스 ≥1 프레임 | 94 / 1260 (7.5%) |
| 누적 박스 수 | 114개 |
| 박스 있는 프레임 평균 박스 수 | 1.2개 |
| 처리 FPS (vehicle 모드) | 36.8 fps |
| 처리 FPS (lane-only 비교) | 71.1 fps |

---

## 파라미터 민감도 실험 (튜닝 단계)

| scale_factor | min_neighbors | min_size | hit 율 |
|---|---|---|---|
| 1.05 | 2 | (20,20) | 41.5% (33/42 샘플) — 대부분 FP |
| 1.1 | 3 | (40,40) | 7.5% (94/1260 전수) — 선택 |

min_neighbors=2로 낮추면 hit 율이 41%로 올라가지만 육안 확인 결과 대부분이 도로 질감/그림자 FP이므로 기각.

---

## 육안 검증 (캡처 프레임 직접 확인)

### 검출 샘플 (vehicle_*.png)

**vehicle_6.png** (frame 6), **vehicle_83.png** (frame 83), **vehicle_119.png** (frame 119):
- 3개 모두 박스가 **도로 소실점 근처 (y≈420, ROI 상단부)**에 위치.
- 실제 차량이 해당 위치에 있는지 원본 프레임 대조 결과: 모두 **도로 표면 질감/차선 마킹 교차점**에 반응한 FP 또는 매우 먼 거리의 차량이어서 식별 불가.
- "vehicle?" 라벨이 정확: 후보일 뿐 확정이 아님.

### FP(오검) 사례 — frames/vehicle_fp.png

- **frame 247**: 박스가 도로 소실점 중앙에 위치. 동시에 실제 백색 차량이 오른쪽 차선에 명확히 보임.
- **cascade가 실제 차 있는 위치를 무시하고 도로 질감에 반응한 전형적 FP.**
- 빨간 원이 실제 차량 위치를 표시 (annotation 추가됨).

### missed(미검출) 사례 — frames/vehicle_missed.png

- **frame 360**: 오른쪽 차선에 백색 차량이 명확히 보이나 `candidates:0` — 완전 미검출.
- 원인: 차량이 ROI의 오른쪽 끝(x=896)에 걸쳐 있어 forward_roi x=[384,896]의 경계에 걸리거나, 측면 외관이어서 cars.xml (전면/후면 학습)과 불일치.

---

## 정직한 한계 분석

### 1. 박스 위치가 실제 차량과 일치하지 않음 (핵심 한계)
검출된 94 프레임의 박스 대부분이 **도로 소실점 근처 y≈420** 에 집중.
원인:
- Haar cascade는 학습 데이터 외관과 비슷한 패턴에 반응 → 도로 표면의 반복 패턴이 차량 후면과 유사.
- cars.xml은 주로 **후면 뷰** 자동차로 학습 → **전방 카메라** 영상의 실제 전방 차량 외관(윗면/유리)과 불일치.

### 2. 이동 카메라 문제
배경이 고정되지 않아 배경차분 기반 접근 불가.
Farneback 광류도 도로 전체가 움직이는 것으로 계산 → "접근하는 차량"만 분리 불가.

### 3. 파라미터 민감도
min_neighbors 1 낮추는 것(3→2)만으로 hit 율이 7.5%→41.5%로 급변.
robust한 검출기가 아님 — 실제 배포 불가.

### 4. 프레임 처리 속도 저하
lane-only 71fps → vehicle mode 36.8fps (약 48% 감소).
Haar cascade + Farneback의 연산 부하.

---

## 결론 → DL(YOLO)이 필요한 이유

classical CV (Haar cascade)의 전방 차량 검출은:
- 전체 프레임의 **7.5%에서만** 후보 박스 생성 (대부분 구간 미검출)
- **박스 위치가 실제 차량과 일치하지 않는 경우가 다수** (도로 질감 FP)
- **파라미터에 극도로 민감** — 실제 운용 환경에서 튜닝 불가

반면 YOLO 계열 DL 검출기는:
- 수천만 장 이미지로 학습 → 관점/크기/조명 변화에 강건
- 전방/측면/후면 모두 동일 모델로 처리 (학습 데이터에 포함)
- 1-stage detector로 실시간(>30fps) 가능

**결론: M7은 classical CV 한계를 정직하게 보여주는 실험이다. "전방 차량 후보"라는 이름 그대로 — 박스가 곧 차량을 의미하지 않음. 차선/LDW(M0~M5)가 핵심이며, 이 실험은 "한계 인식 + DL 동기 부여" 목적의 보너스 레이어이다.**

---

## 차선 파이프라인 무영향 확인

vehicle 플래그 ON/OFF 후 동일 프레임(frame 600) 픽셀 비교:
- Max pixel diff: **0** (완전 동일)
- Mean pixel diff: **0.0000**
- 출력 파일: `project_video_vehicle.mp4` (lane-only `project_video_overlay.mp4`와 별도)
- 결론: `--vehicle` 플래그 없이 실행하면 기존 차선/LDW/curve 파이프라인과 출력이 비트 단위로 동일. 차선 검출 통계(raw_detected 98.0%/95.7%)도 동일.
