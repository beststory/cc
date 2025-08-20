#!/bin/bash

# 전체 코드 품질 검사 스크립트
set -e

echo "🚀 전체 코드 품질 검사를 시작합니다..."

# 1. 포맷팅
echo "1️⃣ 코드 포맷팅..."
./scripts/format.sh

# 2. 품질 검사
echo "2️⃣ 코드 품질 검사..."
./scripts/lint.sh

# 3. 테스트 실행
echo "3️⃣ 테스트 실행..."
./scripts/test.sh

echo "🎉 모든 검사가 성공적으로 완료되었습니다!"
echo "📈 프로젝트가 높은 품질 기준을 충족합니다."