# RoadVision 발표 슬라이드 빌드 명세 (HTML PPT)

5분 발표 / 시연 30~60초 포함 / 11슬라이드. 각 슬라이드는 "무슨 기법을 어떻게 썼는지"를 구체적으로 보여준다.

## 아키텍처
- `presentation/style.css` — 공유 디자인 시스템(코퍼릿 블루).
- `presentation/slides/slide-01.html` ~ `slide-11.html` — 각 1280×720 독립 페이지, `../style.css`·`../assets/` 참조.
- `presentation/index.html` — 발표용. 키보드(←/→)로 슬라이드 전환(iframe 또는 섹션 스택). 풀스크린.
- 헤드리스 Chrome으로 각 slide-NN.html → PNG 렌더(검증·PDF용).

## 디자인 토큰 (첨부 Corporate Blue 레퍼런스 기준)
- `--blue: #2E5BBA;` (메인 로열 블루) / `--blue-dark:#21407f;` / `--ink:#1f2430;` / `--muted:#5b6472;` / `--bg:#ffffff;` / `--line:#e3e8f0;`
- 폰트: 산세리프 — `Pretendard, "Noto Sans KR", system-ui, sans-serif`. 제목은 굵게(800), 본문 400/500.
- 모티프: 좌측 세로 사이드바 라벨, 큰 볼드 타이틀, 솔리드 블루 박스 강조, 섹션 번호 배지(01·02·03), 가는 화살표/구분선, 카드형 레이아웃, 여백 넉넉.
- 16:9, 1280×720 고정. 이미지는 라운드 코너(8px)+옅은 그림자.

## 슬라이드별 내용 (제목 / 레이아웃 / 자산 / 핵심 불릿)
발화 대본은 `presentation/script.md`(humanize 윤문본)에서 슬라이드별로 가져와 발표자 노트로 사용.

### slide-01 — 타이틀
- 풀 블루 배경, 좌측 세로 라벨 "COMPUTER VISION · FINAL". 큰 흰색 타이틀 "RoadVision".
- 부제: "Classical CV 기반 대시캠 차선 인식 및 차선 이탈 경고(LDW)". 우측 가는 화살표.

### slide-02 — 문제와 목표
- 좌: 문구. 우: `assets/st01_input.png`(원본 대시캠).
- 불릿: ① 딥러닝(YOLO) 없이 고전 CV만 ② 목표=안정성·완성도 ③ ADAS 기본기능(차선·이탈경고)

### slide-03 — 수업 기법 매핑 (표)
- 4행 표(블루 헤더): 6강 Canny·Hough→차선 선분 / 7강 HSV·모폴로지→마스크 / 9강 호모그래피→버드아이 / 10강 시간평활·옵티컬플로우→안정화·차량.
- 캡션: "핵심 4강의를 하나의 실시간 파이프라인으로."

### slide-04 — 파이프라인 개요 (흐름도)
- 가로 플로우: 입력 → 색상필터+Canny → ROI → Hough → 좌우분리·폴리핏 → 시간평활 → LDW → 버드아이/곡선. 블루 화살표 체인.

### slide-05 — 검출: 색상필터→Canny→Hough→폴리핏 (4컷)
- 4장 그리드: `st02_colormask.png`(HSV 흰/노랑) · `st03_canny.png`(Canny) · `st05_hough.png`(Hough 선분+ROI) · `st07_ldw_area.png`(폴리핏 결과).
- 불릿(기법): HSV inRange / Canny / HoughLinesP / 길이가중 1차 폴리핏.

### slide-06 — 핵심 문제 해결: 점선 + 모폴로지 (Before/After)
- 좌: 문제 설명(교집합이 점선 끊음, 64%). 우: `st04_lanemask.png`.
- 강조 박스: "모폴로지 클로징(7강) → 우측 검출 64% → **100%**. 보정 아닌 실제 검출 증가."

### slide-07 — 안정성: 시간평활 + 정직 로깅
- 좌: 시간평활(N=10, 떨림 79%↓)·이상치거부. 우: 4상태 로깅 다이어그램(raw/held/rejected/missing).
- 강조: "검출률을 평활로 부풀릴 수 없게 raw만 집계."

### slide-08 — LDW + 버드아이
- 좌: `st08_ldw_warn.png`(경고 배너). 우: `st09_birdeye.png`(원근↔탑다운).
- 불릿: 정규화 오프셋·히스테리시스(0.35/0.25) / 호모그래피 버드아이.

### slide-09 — 곡선 차선 (Core-plus)
- 좌: 슬라이딩 윈도우+2차 폴리핏 설명 + fail-safe. 우: `st10_curve.png`(커브 추종).
- 강조: "곡선 실패 시 직선으로 안전 복귀 — 영상은 항상 생성."

### slide-10 — 시연 (데모)
- 중앙 큰 영상 `assets/demo_cut.mp4`(autoplay/loop/muted, controls). 하단 캡션: 직선→커브, 버드아이·곡률 표시.

### slide-11 — 결과·한계·결론
- 좌 지표 카드 3개: C2 직선 100% / C4 합성이동 검증 / C5 115fps·70fps.
- 우: 차량 한계(`st11_vehicle.png`, 7% FP) → "DL이 필요한 이유".
- 마무리 한 줄: "고전 CV만으로 ADAS 핵심을 안정적으로 완성."

## 렌더/검증
- 각 slide-NN.html을 헤드리스 Chrome `--headless --screenshot --window-size=1280,720`로 PNG 저장 → 육안 확인.
- 글자 잘림·이미지 깨짐·정렬 점검. 깨지면 수정 후 재렌더.
