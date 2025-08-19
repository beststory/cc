#!/bin/bash

# 코드 품질 검사 스크립트
set -e

echo "🔍 코드 품질 검사를 시작합니다..."

# 코드 스타일 검사
echo "📏 flake8으로 코드 스타일 검사 중..."
uv run flake8 backend/ main.py --max-line-length=88 --extend-ignore=E203,W503

# 타입 검사
echo "🔬 mypy로 타입 검사 중..."
uv run mypy backend/ main.py

# Import 순서 검사
echo "🔄 isort로 import 순서 검사 중..."
uv run isort --check-only --diff backend/ main.py

# Black 포맷팅 검사
echo "📝 Black으로 포맷팅 검사 중..."
uv run black --check --diff backend/ main.py

echo "✅ 모든 품질 검사가 통과되었습니다!"