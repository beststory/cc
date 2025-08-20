.PHONY: install format lint test check clean help

help:
	@echo "🚀 Course Materials RAG System - 개발 명령어"
	@echo ""
	@echo "install     - 개발 의존성 설치"
	@echo "format      - 코드 포맷팅 (Black + isort)"
	@echo "lint        - 코드 품질 검사 (flake8 + mypy)"
	@echo "test        - 테스트 실행"
	@echo "check       - 전체 코드 품질 검사"
	@echo "clean       - 생성된 파일 정리"
	@echo "dev         - 개발 서버 실행"

install:
	@echo "📦 개발 의존성을 설치합니다..."
	uv sync --extra dev

format:
	@echo "🎨 코드 포맷팅을 시작합니다..."
	uv run black backend/ main.py
	uv run isort backend/ main.py
	@echo "✅ 코드 포맷팅이 완료되었습니다!"

lint:
	@echo "🔍 코드 품질 검사를 시작합니다..."
	uv run flake8 backend/ main.py --max-line-length=88 --extend-ignore=E203,W503
	uv run mypy backend/ main.py --ignore-missing-imports
	uv run isort --check-only --diff backend/ main.py
	uv run black --check --diff backend/ main.py
	@echo "✅ 모든 품질 검사가 통과되었습니다!"

test:
	@echo "🧪 테스트를 시작합니다..."
	uv run pytest backend/tests/ -v --cov=backend --cov-report=term-missing --cov-report=html
	@echo "✅ 모든 테스트가 완료되었습니다!"

check: format lint test
	@echo "🎉 모든 검사가 성공적으로 완료되었습니다!"

dev:
	@echo "🚀 개발 서버를 시작합니다..."
	./run.sh

clean:
	@echo "🧹 생성된 파일을 정리합니다..."
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf backend/__pycache__/
	rm -rf backend/tests/__pycache__/
	rm -rf backend/chroma_db/
	@echo "✅ 정리가 완료되었습니다!"