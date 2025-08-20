#!/bin/bash

# 테스트 실행 스크립트
set -e

echo "🧪 테스트를 시작합니다..."

# 단위 테스트 실행
echo "🔬 pytest로 테스트 실행 중..."
uv run pytest backend/tests/ -v --cov=backend --cov-report=term-missing --cov-report=html

echo "✅ 모든 테스트가 완료되었습니다!"
echo "📊 HTML 커버리지 리포트가 htmlcov/index.html에 생성되었습니다."