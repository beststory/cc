#!/bin/bash

# 코드 포맷팅 스크립트
set -e

echo "🎨 코드 포맷팅을 시작합니다..."

# Black 포맷팅
echo "📝 Black으로 포맷팅 중..."
uv run black backend/ main.py

# Import 정렬
echo "🔄 isort로 import 정렬 중..."
uv run isort backend/ main.py

echo "✅ 코드 포맷팅이 완료되었습니다!"