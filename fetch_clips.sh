#!/usr/bin/env bash
# RoadVision 입력 클립 다운로드 (재현성). 출처: Udacity CarND (MIT), 가공 안 된 원본.
# 대용량 mp4는 git에 안 올라가므로 클론 후 이 스크립트로 받는다.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p clips

base_p1="https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_videos"
base_adv="https://raw.githubusercontent.com/udacity/CarND-Advanced-Lane-Lines/master"

echo "[1/3] solidYellowLeft.mp4 (메인, 960x540)"
curl -fSL -o clips/solidYellowLeft.mp4  "$base_p1/solidYellowLeft.mp4"
echo "[2/3] solidWhiteRight.mp4 (검증, 960x540)"
curl -fSL -o clips/solidWhiteRight.mp4  "$base_p1/solidWhiteRight.mp4"
echo "[3/3] project_video.mp4 (HD 커브+앞차, 1280x720)"
curl -fSL -o clips/project_video.mp4    "$base_adv/project_video.mp4"

echo "완료. 받은 파일:"
ls -lh clips/*.mp4
