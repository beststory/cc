import os
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch
from fastapi.testclient import TestClient

import pytest

# Add backend to sys.path for importing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock chromadb before importing vector_store
sys.modules["chromadb"] = MagicMock()
sys.modules["chromadb.config"] = MagicMock()
sys.modules["chromadb.utils"] = MagicMock()
sys.modules["chromadb.utils.embedding_functions"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults


@pytest.fixture
def sample_course():
    """Sample course for testing"""
    return Course(
        title="MCP 서버 개발 가이드",
        course_link="https://example.com/mcp-course",
        instructor="Claude Assistant",
        lessons=[
            Lesson(
                lesson_number=1,
                title="MCP 소개",
                lesson_link="https://example.com/lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="서버 설정",
                lesson_link="https://example.com/lesson2",
            ),
            Lesson(
                lesson_number=3,
                title="도구 구현",
                lesson_link="https://example.com/lesson3",
            ),
        ],
    )


@pytest.fixture
def sample_course_chunks():
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="MCP는 Model Context Protocol의 줄임말입니다.",
            course_title="MCP 서버 개발 가이드",
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="서버 설정은 config.json 파일을 통해 관리됩니다.",
            course_title="MCP 서버 개발 가이드",
            lesson_number=2,
            chunk_index=1,
        ),
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return SearchResults(
        documents=[
            "MCP는 Model Context Protocol의 줄임말입니다.",
            "서버 설정은 config.json 파일을 통해 관리됩니다.",
        ],
        metadata=[
            {
                "course_title": "MCP 서버 개발 가이드",
                "lesson_number": 1,
                "chunk_index": 0,
            },
            {
                "course_title": "MCP 서버 개발 가이드",
                "lesson_number": 2,
                "chunk_index": 1,
            },
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def empty_search_results():
    """Empty search results for testing"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    """Error search results for testing"""
    return SearchResults.empty("Database connection error")


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = Mock()

    # Default behavior - return empty results
    mock_store.search.return_value = SearchResults(
        documents=[], metadata=[], distances=[]
    )
    mock_store._resolve_course_name.return_value = None
    mock_store.get_lesson_link.return_value = None
    mock_store.get_all_courses_metadata.return_value = []

    return mock_store


@pytest.fixture
def mock_vector_store_with_data(
    mock_vector_store, sample_search_results, sample_course
):
    """Mock vector store with sample data"""
    mock_vector_store.search.return_value = sample_search_results
    mock_vector_store._resolve_course_name.return_value = "MCP 서버 개발 가이드"
    mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
    mock_vector_store.get_all_courses_metadata.return_value = [
        {
            "title": sample_course.title,
            "instructor": sample_course.instructor,
            "course_link": sample_course.course_link,
            "lessons": [
                {
                    "lesson_number": lesson.lesson_number,
                    "lesson_title": lesson.title,
                    "lesson_link": lesson.lesson_link,
                }
                for lesson in sample_course.lessons
            ],
        }
    ]

    return mock_vector_store


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    mock_client = Mock()

    # Mock response without tool use
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [Mock(text="This is a test response")]

    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_anthropic_client_with_tool_use(mock_anthropic_client):
    """Mock Anthropic client that uses tools"""

    # First response with tool use
    tool_use_response = Mock()
    tool_use_response.stop_reason = "tool_use"
    # Create tool use content with proper attributes
    tool_use_content = Mock()
    tool_use_content.type = "tool_use"
    tool_use_content.name = "search_course_content"
    tool_use_content.id = "tool_123"
    tool_use_content.input = {"query": "MCP", "course_name": "MCP 서버 개발 가이드"}

    tool_use_response.content = [tool_use_content]

    # Final response after tool execution
    final_response = Mock()
    final_response.stop_reason = "end_turn"
    final_response.content = [Mock(text="MCP는 Model Context Protocol입니다.")]

    # Set up call sequence
    mock_anthropic_client.messages.create.side_effect = [
        tool_use_response,
        final_response,
    ]

    return mock_anthropic_client


@pytest.fixture
def mock_tool_manager():
    """Mock tool manager for testing"""
    mock_manager = Mock()
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        }
    ]
    mock_manager.execute_tool.return_value = "MCP는 Model Context Protocol입니다."
    mock_manager.get_last_sources.return_value = []
    mock_manager.reset_sources.return_value = None

    return mock_manager


@pytest.fixture
def test_config():
    """Test configuration"""
    from config import Config

    config = Config()
    config.ANTHROPIC_API_KEY = "test_key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config


@pytest.fixture
def mock_rag_system():
    """Mock RAG system for API testing"""
    mock_rag = Mock()
    
    # Mock query method
    mock_rag.query.return_value = (
        "이것은 테스트 응답입니다.",
        [
            {
                "content": "테스트 내용입니다.",
                "course_title": "테스트 코스",
                "lesson_number": 1,
                "lesson_link": "https://example.com/lesson1"
            }
        ]
    )
    
    # Mock session manager
    mock_rag.session_manager = Mock()
    mock_rag.session_manager.create_session.return_value = "test_session_123"
    
    # Mock get_course_analytics
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["테스트 코스 1", "테스트 코스 2"]
    }
    
    # Mock add_course_folder
    mock_rag.add_course_folder.return_value = (2, 10)  # 2 courses, 10 chunks
    
    return mock_rag


@pytest.fixture
def test_client(mock_rag_system):
    """FastAPI test client with mocked dependencies"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Dict, Any
    
    # Create test app without static file mounting
    test_app = FastAPI(title="Test Course Materials RAG System")
    
    # Add CORS middleware
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request/Response models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
        
    class QueryResponse(BaseModel):
        answer: str
        sources: List[Dict[str, Any]]
        session_id: str
        
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
        
    class NewSessionResponse(BaseModel):
        session_id: str
    
    # Mock the global rag_system
    rag_system = mock_rag_system
    
    # API endpoints
    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()
        
        answer, sources = rag_system.query(request.query, session_id)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )
    
    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"]
        )
    
    @test_app.post("/api/new-session", response_model=NewSessionResponse)
    async def create_new_session():
        session_id = rag_system.session_manager.create_session()
        return NewSessionResponse(session_id=session_id)
    
    @test_app.get("/")
    async def root():
        return {"message": "Course Materials RAG System API"}
    
    # Create test client
    return TestClient(test_app)


@pytest.fixture
def api_query_payload():
    """Sample API query payload"""
    return {
        "query": "MCP란 무엇인가요?",
        "session_id": "test_session_123"
    }


@pytest.fixture
def expected_query_response():
    """Expected API query response structure"""
    return {
        "answer": str,
        "sources": list,
        "session_id": str
    }


@pytest.fixture
def expected_courses_response():
    """Expected API courses response structure"""
    return {
        "total_courses": int,
        "course_titles": list
    }
