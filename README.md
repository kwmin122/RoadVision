# RoadVision — Classical CV 기반 대시캠 차선 인식 및 차선 이탈 경고(LDW)

컴퓨터비전 기말 프로젝트. **딥러닝 없이 classical CV 기법(Canny·Hough·threshold·morphology·homography·optical flow)만으로** 대시캠 영상에서 차선을 검출하고, 차선 중앙과 차량 중심을 비교해 **차선 이탈을 경고**하는 ADAS 기본 기능을 구현한다.

> 최상위 기준: **성능 안정성 · 완성도**. 단일 진실원 = [`PLAN.md`](./PLAN.md).

## 범위 (성공 기준 2층)
- **Core (=프로젝트 성공, M0~M5)**: 차선 검출 + 차선중앙 추정 + LDW 경고 + bird-eye + 시간평활. 결과 영상은 반드시 생성.
- **Core-plus (M6)**: project_video 곡선 2차 폴리핏 쇼케이스(실패 시 Hough 오버레이로 fail-safe).
- **Bonus (M7)**: 전방 차량 후보 시각화 실험(2h timebox) + 한계분석.
- **발표 (M8)**: HTML PPT + 대본.
- **Out**: YOLO/DL, 차량 카운팅.

## 수업 기법 매핑
| 강의 | 기법 | 사용처 |
|---|---|---|
| 6강 | Canny, HoughLinesP | 차선 엣지·선분 검출 |
| 7강 | HSV 색상필터, morphology | 흰/노랑 차선 마스크·노이즈 제거 |
| 9강 | homography, perspectiveTransform | bird-eye view |
| 10강 | 시간평활, optical flow | 차선 안정화 / 차량 보너스 |
| 8강 | contour, connectedComponents | (보너스) 차량 후보 |

## 입력 데이터
Udacity CarND 표준 클립 3종(MIT, 가공 안 된 원본). 대용량이라 git 미포함 — 클론 후:
```bash
bash fetch_clips.sh
```

## 셋업 & 실행
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash fetch_clips.sh
python -m src.main --clip solidYellowLeft   # (구현 진행에 따라)
```

## 개발 방식
- **계획·설계·검수 = Opus / 구현 = Sonnet**. 매 슬라이스마다 diff 검수 + 임계값 grep 대조 + 출력 영상 육안 검증(PLAN §12b).
- **Vertical slice**: 매 슬라이스가 `입력 mp4 → 출력 mp4`로 관통. 모듈 따로 만들고 마지막에 합치기 금지.

자세한 내용은 [`PLAN.md`](./PLAN.md) 참조.
