---
allowed-tools: [Read, Write, Edit, MultiEdit, Bash, Glob, TodoWrite, Task]
description: "Frontend feature implementation with automatic change tracking"
---

# /implement-feature - Frontend Feature Implementation

## Purpose
Frontend 특화 기능 구현 및 변경사항 자동 추적

## Usage
```
/implement-feature [feature-description] [--framework react|vue|angular] [--safe]
```

## Arguments
- `feature-description` - 구현할 기능 설명
- `--framework` - 사용할 프론트엔드 프레임워크
- `--safe` - 보수적 구현 접근법 사용
- `--with-tests` - 테스트 코드 포함
- `--iterative` - 반복적 개발 활성화

## Execution Flow
1. Frontend 요구사항 분석 및 기술 스택 감지
2. Frontend 페르소나 자동 활성화
3. Magic MCP 서버를 통한 UI 컴포넌트 생성
4. 기능 구현 및 품질 검증
5. **frontend-changes.md 파일에 변경사항 자동 기록**
6. 테스트 권장사항 제공

## Auto-Features
- Frontend 페르소나 자동 활성화
- Magic MCP 서버 연동 (UI 컴포넌트 생성)
- 접근성 및 반응형 디자인 자동 적용
- **변경사항 자동 추적 및 문서화**

$ARGUMENTS