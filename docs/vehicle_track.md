# L10 차량 추적 데모 — CSRT 추적기 정직 보고서

## 요약

`--vehicle-track` 플래그로 실행하면 CSRT(Discriminative Correlation Filter with Channel
and Spatial Reliability) 상관 필터 추적기가 seed 박스로 초기화된 차량을 이후 프레임에서
추적한다. 이 데모는 **Lecture 10: 추적(Tracking)** 에 해당한다.

---

## 구현 개요

| 항목 | 내용 |
|------|------|
| 추적기 | `cv2.TrackerCSRT_create()` (opencv-contrib 4.13) |
| seed 클립 | `project_video.mp4` — 1280×720, 50s, 25fps, 총 1260프레임 |
| seed 프레임 | frame 200 (t=8s) |
| seed 박스 | (x=855, y=355, w=380, h=165) — 흰색 승용차, 우측 차선 |
| 추적 프레임 수 | 1060프레임 (frame 201~1260, 클립 끝까지) |
| LOST 발생 | **없음** — 전 구간 추적 성공 (100%) |

---

## seed 박스 선정 과정

1. `project_video.mp4`에서 frame 125(t=5s)·175(t=7s)·200(t=8s)·960(t=38s) 등 여러 프레임을
   시각적으로 스캔해 차량을 확인.
2. frame 200에서 흰색 승용차(우측 차선)가 뚜렷하게 보이고, 클립 잔여 프레임이 1060개로
   가장 길어 이 프레임을 선택.
3. **Haar cascade 자동 탐색 실패**: `detect_candidates()`의 `forward_roi_ratio`
   (x=0.30~0.70, y=0.55~1.00) 바깥에 차량이 위치해 자동 seed 탐색 불가.
4. 따라서 `config.VEHICLE["track_seed"]["project_video"]`에 하드코딩 —
   PLAN.md 규칙대로 "데이터, 로직 아님" 처리.
5. 박스 좌표는 프레임 저장 후 격자 오버레이로 육안 측정하여 결정.

---

## 추적 결과 (정량)

| 항목 | 수치 |
|------|------|
| 추적 성공 프레임 | 1060 / 1060 (100%) |
| 첫 LOST 발생 | 없음 |
| 연속 최대 추적 | 1060 프레임 (끝까지) |
| 검증 저장 프레임 | frames/track_1.png (+10), track_2.png (+100), track_3.png (+300) |

---

## 검증 프레임 설명 (육안)

- **track_1.png** (frame 210, seed+10): 흰 승용차 위에 초록 박스가 정확히 얹힘. 차선 overlay
  + CURVE 모드 동시 표시. 궤적 점 아직 적음.
- **track_2.png** (frame 300, seed+100): 차량이 카메라에 가까워져 박스가 커지고 위치 이동.
  CSRT가 스케일 변화를 흡수하며 동일 차량 추적.
- **track_3.png** (frame 500, seed+300): 차량이 더 가까워져 우측 화면 가장자리에 위치.
  박스가 차체에 맞게 조정됨. 궤적이 왼쪽→오른쪽 방향으로 표시.

---

## Classical CV 한계 (정직 분석)

### 1. Seed 위치 결정이 취약점
- Haar cascade로 자동 seed를 잡지 못했다. 이동 카메라 특성상 전방 ROI 밖에 있는
  차량은 탐색되지 않는다.
- 수동 하드코딩 박스에 의존 → 새 영상이나 다른 카메라 위치에 적용 시 재측정 필요.

### 2. LOST 후 재검출 불가
- CSRT는 순수 추적기이므로 LOST 이후 자동으로 타깃을 다시 찾지 못한다.
- 이 클립에서는 LOST 없이 성공했지만, 차량이 프레임 밖으로 나가거나 다른 차량에
  완전히 가려지면 LOST 후 추적 중단.

### 3. 클립 특화 성능
- `project_video`의 우측 차선 흰 차량은 특수 조건(직선+완만한 이동+밝은 차체)이라
  CSRT가 잘 동작한다.
- 야간·악천후·고속 가로지르기·부분 가림 등 현실 조건에서는 조기 LOST 가능.

### 4. DL(YOLO)과의 비교
- YOLO 등 딥러닝 검출기는 매 프레임 독립 검출 → LOST 개념 없음.
- Classical CV 추적은 **재검출 없는 연속성**을 강점으로 가지나,
  초기 seed 획득에 검출기가 필요하다. 즉 완전한 시스템은 "DL 검출 + classical 추적" 조합.

---

## 의존성 변경 사항

`opencv-contrib-python 4.13.0.92`를 설치해야 `cv2.TrackerCSRT_create()`를 사용할 수 있다.
`opencv-python`(contrib 제외 빌드)과 동시 설치 시 충돌하므로, `requirements.txt`를
`opencv-contrib-python>=4.13`으로 교체했다. 상위집합이므로 기존 lane 파이프라인 기능은
완전 유지된다.

---

## 정상 모드 영향 없음 확인

`--vehicle-track` 플래그 없는 일반 실행에서 solidYellowLeft의 `raw_detected` 비율이
opencv-contrib 교체 전후 모두 **100.0% (681/681 프레임)** 로 동일함을 실측 확인.
