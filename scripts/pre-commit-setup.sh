#!/bin/bash

# Pre-commit hooks 설정 스크립트
set -e

echo "🔧 Pre-commit hooks를 설정합니다..."

# .pre-commit-config.yaml 생성
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 24.2.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203,W503"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        exclude: ^(docs/|tests/)
EOF

# Pre-commit 설치
echo "📦 Pre-commit hooks 설치 중..."
uv run pre-commit install

echo "✅ Pre-commit hooks가 성공적으로 설정되었습니다!"
echo "🔄 이제 커밋할 때마다 자동으로 코드 품질 검사가 실행됩니다."