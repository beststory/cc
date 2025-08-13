# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Install dependencies
uv sync

# Set up environment
# Create .env file with: ANTHROPIC_API_KEY=your_key_here
```

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8001

# For network access (external connections)
cd backend && uv run uvicorn app:app --reload --host 0.0.0.0 --port 8001
```

### Data Management
```bash
# Clear vector database (fresh start)
rm -rf backend/chroma_db/

# Add new course documents
# Place .txt files in docs/ directory, restart server to auto-load
```

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** built with FastAPI that enables intelligent question-answering about course materials using vector search and AI generation.

### Core Architecture Pattern
The system follows a **layered orchestration pattern** where `rag_system.py` acts as the central coordinator:

```
Frontend (script.js) → FastAPI (app.py) → RAG System (rag_system.py) → [AI + Vector + Session] Components
```

### Data Flow Architecture
1. **Document Ingestion**: `document_processor.py` parses structured course files → `vector_store.py` creates embeddings → ChromaDB storage
2. **Query Processing**: User question → AI determines if search needed → `search_tools.py` performs vector search → AI generates contextual response
3. **Session Management**: `session_manager.py` maintains conversation context for multi-turn dialogs

### Key Architectural Components

**`rag_system.py`** - Central orchestrator that coordinates all system components. This is the primary entry point for understanding the system flow.

**`document_processor.py`** - Handles structured parsing of course documents with specific format:
- Line 1: Course Title: [title]  
- Line 2: Course Link: [url]
- Line 3: Course Instructor: [instructor]
- Subsequent lines: Lesson markers (`Lesson N: title`) and content

**`vector_store.py`** - Manages ChromaDB with two collections:
- `course_catalog`: Course metadata (titles, instructors, links)
- `course_content`: Chunked content for semantic search
- Implements intelligent course name resolution for fuzzy matching

**`ai_generator.py`** - Claude API integration with tool-calling support. Uses a specialized system prompt optimized for educational content and implements conversation context management.

**`search_tools.py`** - Provides AI-accessible tools using Anthropic's tool-calling interface. The `CourseSearchTool` enables the AI to autonomously search course content when needed.

### Vector Database Structure
ChromaDB stores data in `/backend/chroma_db/` with:
- Sentence-transformer embeddings (all-MiniLM-L6-v2)
- Structured metadata (course_title, lesson_number, chunk_index)
- Semantic search with filtering capabilities

### Frontend Integration
The frontend uses vanilla JavaScript with real-time updates, Markdown rendering, and source attribution. It communicates via REST API (`/api/query`, `/api/courses`) and maintains session state.

## Important Configuration

### Network Settings
- Default: `127.0.0.1:8001` (localhost only)
- Network access: `0.0.0.0:8001` (external connections)
- Update `run.sh` for persistent network configuration changes

### Environment Variables
- `ANTHROPIC_API_KEY`: Required for Claude AI functionality
- `CHROMA_PATH`: Vector database location (default: `./chroma_db`)
- Key settings in `config.py`: chunk size (800), overlap (100), max results (5)

### Document Format Requirements
Course documents must follow the structured format in `/docs/`. The document processor expects:
- Metadata in first 3 lines
- Lesson markers: `Lesson N: title`  
- Optional lesson links: `Lesson Link: url`

## Development Context

- **No testing framework** is currently implemented
- **No linting configuration** is present  
- Uses `uv` as package manager (not pip/poetry)
- **Auto-reloading** enabled in development mode
- Documents auto-load on server startup from `/docs/` directory
- Vector database is created automatically on first run
- 항상 uv 를 사용하고 pip 은 사용하지 말아