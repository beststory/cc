import os
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

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
