# RoadVision Metrics Report

> 생성: `python -m src.metrics` (Slice 8, M6 verify)
> 모든 수치는 이 세션 실제 실행 결과. 수작업 수치 없음.

## C1 — 무결성 (Integrity)

| 클립 | 해상도 | 입력 프레임 | 출력 프레임 | 일치 | 결과 |
|---|---|---:|---:|:---:|:---:|
| solidYellowLeft | 960x540 | 681 | 681 | YES | **PASS** |
| solidWhiteRight | 960x540 | 221 | 221 | YES | **PASS** |
| project_video | 1280x720 | 1260 | 1260 | YES | **PASS** |

C1 판정 기준: 출력 mp4가 존재하고 프레임 수가 입력과 일치, 실행 중 예외 없음.

## C2 — 원시 검출률 (Raw Detection Rate)

**판정 클립**: `solidYellowLeft`, `solidWhiteRight` (직선, C2 pass/fail 대상)

`project_video` 는 곡선 쇼케이스 클립이므로 C2 pass/fail 에서 **제외**되고 참고 지표로만 보고한다.

### C2 통계 상세

| 클립 | 방향 | raw_detected | held | rejected | cmissing | 합계 | raw% | max_streak |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| solidYellowLeft | left | 681 | 0 | 0 | 0 | 681/681 | 100.0% | 0 |
| solidYellowLeft | right | 681 | 0 | 0 | 0 | 681/681 | 100.0% | 0 |
| solidWhiteRight | left | 221 | 0 | 0 | 0 | 221/221 | 100.0% | 0 |
| solidWhiteRight | right | 221 | 0 | 0 | 0 | 221/221 | 100.0% | 0 |
| project_video | left | 1235 | 22 | 3 | 0 | 1260/1260 | 98.0% | 4 |
| project_video | right | 1206 | 53 | 0 | 1 | 1260/1260 | 95.7% | 4 |

- **raw%** = raw_detected 행 수 / 전체 프레임 (hold·rejected 로 부풀리지 않음)
- **max_streak** = raw_detected 가 아닌 연속 프레임의 최장 구간 (단순 레이블 카운트 ≠)

### C2 PASS/FAIL 판정 (solidYellowLeft · solidWhiteRight)

| 클립 | 왼쪽 raw% | 오른쪽 raw% | 기준(≥90%) | 판정 |
|---|---:|---:|:---:|:---:|
| solidYellowLeft | 100.0% | 100.0% | 90% | **PASS** |
| solidWhiteRight | 100.0% | 100.0% | 90% | **PASS** |

### project_video C2 참고 (C2 pass/fail 미포함)

| 방향 | raw% | max_streak |
|---|---:|---:|
| left  | 98.0% | 4 |
| right | 95.7% | 4 |

오른쪽 95.7% 는 프로젝트 전체 최저치 (그러나 여전히 ≥90%).

## C5 — 처리 속도 (Processing FPS)

데모는 사전 렌더 재생 방식이므로 실시간성은 필수 기준이 아님 (PLAN §10 C5).
아래 수치는 `src.main` 내부 루프-only FPS (`frames_written / loop_elapsed`).

| 클립 | 해상도 | 처리 FPS |
|---|---|---:|
| solidYellowLeft | 960x540 | 115.2 |
| solidWhiteRight | 960x540 | 117.8 |
| project_video | 1280x720 | 69.9 |

참고: 소요 벽시계 시간(subprocess 포함 startup overhead)은 보고 지표에서 제외.

## M6 — 곡선 vs Fallback 프레임 집계

슬라이딩 윈도우 + 2차 폴리핏이 유효성 기준을 통과(CURVE) vs 직선 폴백(STRAIGHT).
is_valid() 는 클립 이름 분기 없이 데이터 기반 판단 — 직선 클립도 통과 가능.

| 클립 | CURVE 프레임 | STRAIGHT(fallback) 프레임 | 전체 | CURVE% |
|---|---:|---:|---:|---:|
| solidYellowLeft | 681 | 0 | 681 | 100.0% |
| solidWhiteRight | 221 | 0 | 221 | 100.0% |
| project_video | 1194 | 66 | 1260 | 94.8% |

- **CURVE** = `curve.is_valid()` 통과 → 슬라이딩 윈도우 곡선 오버레이 적용
- **STRAIGHT(fallback)** = 폴리핏 불충분/유효성 미달 → 기존 Hough 직선 오버레이 적용
- 직선 클립도 CURVE 100%: `is_valid()` 가 클립 이름 분기 없이 데이터 기반으로만 판단하므로
  직선 도로에서도 슬라이딩 윈도우 폴리핏이 유효성 검사를 통과함 (PLAN §2).
- project_video 에서 66프레임 STRAIGHT(fallback): 심한 곡률 구간 픽셀 불충분 또는 유효성 미달.
  fail-safe 동작으로 Core 파이프라인(직선 오버레이)이 보호됨.

## C4 — LDW 합성 이동 테스트

C4 상세 결과(케이스별 표·히스테리시스 스윕)는 [`docs/ldw_shift_test.md`](ldw_shift_test.md) 참조.

요약:
- warn_on = 0.35, warn_off = 0.25 (히스테리시스)
- shift=0px: 경고 OFF (실제 오프셋 ≈ −0.057)
- shift=±80px: 경고 OFF (|offset| ≈ 0.17~0.29)
- shift=+140px: LEFT 경고 ON (|offset| = 0.459)
- shift=−160px: RIGHT 경고 ON (|offset| = 0.401)
- 히스테리시스 동작 확인: ON 임계 >0.35, OFF 임계 <0.25

## 한계 및 정직 보고 (Honest Limitations)

1. **GT 라벨 없음**: C2 검출률은 raw_detected 프록시이며 정량 정확도(precision/recall)가 아님.
   실제 차선과 비교할 ground-truth 어노테이션이 없어 위음성/위양성을 분리 계산 불가.

2. **project_video 오른쪽 95.7%**: 전체 3클립 중 최저 수치.
   여전히 ≥90% 이지만 곡선 구간에서 Hough 기반 오른쪽 흰 점선 검출이 가장 취약함.
   이 클립은 C2 pass/fail 대상에서 제외되므로 합격/불합격 판정에 영향 없음.

3. **클립별 ROI·파라미터 조정**: config.py 에서 클립별 ROI·homography 4점을 설정했으며
   `solidYellowLeft` 에서 주로 튜닝 후 나머지를 테스트셋으로 운영.
   범용성(실제 대시캠 영상)은 미검증 — 클립 과적합 가능성 인정.

4. **M6 fallback 비율**: project_video 에서도 일부 프레임이 STRAIGHT(fallback)으로 처리됨.
   곡선 유효성 임계(CURVE_VALIDITY)가 엄격해 오검출 안전성은 높지만
   곡률이 심한 구간에서 CURVE 모드 활성 비율이 낮아질 수 있음.

5. **FPS 변동성**: subprocess 기반 측정이므로 시스템 부하에 따라 ±10% 내외 변동 가능.
   데모는 사전 렌더 방식이므로 실시간 FPS 미달이 발표 품질에 영향 없음.
