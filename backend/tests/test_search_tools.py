import pytest
from unittest.mock import Mock
from search_tools import CourseSearchTool, CourseSyllabulTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool"""
    
    def test_execute_successful_search(self, mock_vector_store_with_data, sample_search_results):
        """Test successful search with results"""
        tool = CourseSearchTool(mock_vector_store_with_data)
        
        result = tool.execute(query="MCP", course_name="MCP 서버 개발 가이드")
        
        # Verify search was called with correct parameters
        mock_vector_store_with_data.search.assert_called_once_with(
            query="MCP",
            course_name="MCP 서버 개발 가이드", 
            lesson_number=None
        )
        
        # Verify result format
        assert "[MCP 서버 개발 가이드 - Lesson 1]" in result
        assert "MCP는 Model Context Protocol의 줄임말입니다." in result
        assert "[MCP 서버 개발 가이드 - Lesson 2]" in result
        assert "서버 설정은 config.json 파일을 통해 관리됩니다." in result
        
        # Verify sources are tracked
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "MCP 서버 개발 가이드 - Lesson 1"
        assert tool.last_sources[0]["link"] == "https://example.com/lesson1"
    
    def test_execute_empty_search_results(self, mock_vector_store):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
        tool = CourseSearchTool(mock_vector_store)
        
        result = tool.execute(query="nonexistent", course_name="Unknown Course")
        
        assert "No relevant content found in course 'Unknown Course'." == result
    
    def test_execute_search_error(self, mock_vector_store):
        """Test handling of search errors"""
        error_result = SearchResults.empty("Database connection error")
        mock_vector_store.search.return_value = error_result
        tool = CourseSearchTool(mock_vector_store)
        
        result = tool.execute(query="test")
        
        assert result == "Database connection error"
    
    def test_execute_with_lesson_filter(self, mock_vector_store_with_data):
        """Test search with lesson number filter"""
        tool = CourseSearchTool(mock_vector_store_with_data)
        
        result = tool.execute(query="설정", lesson_number=2)
        
        mock_vector_store_with_data.search.assert_called_once_with(
            query="설정",
            course_name=None,
            lesson_number=2
        )
    
    def test_get_tool_definition(self, mock_vector_store):
        """Test tool definition structure"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]
    
    def test_format_results_without_lesson_number(self, mock_vector_store):
        """Test formatting results without lesson numbers"""
        tool = CourseSearchTool(mock_vector_store)
        
        # Create search results without lesson numbers
        results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course"}],
            distances=[0.1]
        )
        
        formatted = tool._format_results(results)
        
        assert "[Test Course]" in formatted
        assert "Test content" in formatted
        assert "Lesson" not in formatted


class TestCourseSyllabulTool:
    """Test cases for CourseSyllabulTool"""
    
    def test_execute_successful_syllabus_retrieval(self, mock_vector_store_with_data, sample_course):
        """Test successful syllabus retrieval"""
        tool = CourseSyllabulTool(mock_vector_store_with_data)
        
        result = tool.execute(course_name="MCP")
        
        # Verify course name resolution was called
        mock_vector_store_with_data._resolve_course_name.assert_called_once_with("MCP")
        
        # Verify result format
        assert "# MCP 서버 개발 가이드" in result
        assert "**강사**: Claude Assistant" in result
        assert "**강의 링크**: https://example.com/mcp-course" in result
        assert "## 강의 목록" in result
        assert "1. MCP 소개" in result
        assert "2. 서버 설정" in result
        assert "3. 도구 구현" in result
    
    def test_execute_course_not_found(self, mock_vector_store):
        """Test handling when course is not found"""
        mock_vector_store._resolve_course_name.return_value = None
        tool = CourseSyllabulTool(mock_vector_store)
        
        result = tool.execute(course_name="NonExistent")
        
        assert result == "강의를 찾을 수 없습니다: 'NonExistent'"
    
    def test_execute_metadata_not_found(self, mock_vector_store):
        """Test handling when course metadata is not found"""
        mock_vector_store._resolve_course_name.return_value = "Test Course"
        mock_vector_store.get_all_courses_metadata.return_value = []
        tool = CourseSyllabulTool(mock_vector_store)
        
        result = tool.execute(course_name="Test")
        
        assert result == "강의 메타데이터를 찾을 수 없습니다: 'Test Course'"
    
    def test_execute_exception_handling(self, mock_vector_store):
        """Test exception handling"""
        mock_vector_store._resolve_course_name.return_value = "Test Course"
        mock_vector_store.get_all_courses_metadata.side_effect = Exception("Database error")
        tool = CourseSyllabulTool(mock_vector_store)
        
        result = tool.execute(course_name="Test")
        
        assert "강의 계획서 조회 중 오류 발생: Database error" in result
    
    def test_get_tool_definition(self, mock_vector_store):
        """Test tool definition structure"""
        tool = CourseSyllabulTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "get_course_syllabus"
        assert "description" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "course_name" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["course_name"]
    
    def test_format_syllabus_empty_lessons(self, mock_vector_store):
        """Test formatting syllabus with no lessons"""
        tool = CourseSyllabulTool(mock_vector_store)
        
        course_data = {
            "title": "Empty Course",
            "instructor": "Test Instructor", 
            "course_link": "https://example.com",
            "lessons": []
        }
        
        result = tool._format_syllabus(course_data)
        
        assert "# Empty Course" in result
        assert "강의 목록이 없습니다." in result


class TestToolManager:
    """Test cases for ToolManager"""
    
    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        
        manager.register_tool(tool)
        
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == tool
    
    def test_register_tool_without_name(self, mock_vector_store):
        """Test tool registration error handling"""
        manager = ToolManager()
        
        # Create a mock tool with invalid definition
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"description": "No name"}
        
        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(mock_tool)
    
    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting tool definitions"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        syllabus_tool = CourseSyllabulTool(mock_vector_store)
        
        manager.register_tool(search_tool)
        manager.register_tool(syllabus_tool)
        
        definitions = manager.get_tool_definitions()
        
        assert len(definitions) == 2
        tool_names = [def_["name"] for def_ in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_syllabus" in tool_names
    
    def test_execute_tool(self, mock_vector_store_with_data):
        """Test tool execution"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store_with_data)
        manager.register_tool(tool)
        
        result = manager.execute_tool("search_course_content", query="test")
        
        assert isinstance(result, str)
        # Should contain formatted search results
        assert "[MCP 서버 개발 가이드" in result
    
    def test_execute_nonexistent_tool(self, mock_vector_store):
        """Test executing non-existent tool"""
        manager = ToolManager()
        
        result = manager.execute_tool("nonexistent_tool", query="test")
        
        assert result == "Tool 'nonexistent_tool' not found"
    
    def test_get_last_sources(self, mock_vector_store_with_data):
        """Test getting last sources from tools"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store_with_data)
        manager.register_tool(tool)
        
        # Execute search to populate sources
        manager.execute_tool("search_course_content", query="test")
        
        sources = manager.get_last_sources()
        
        assert len(sources) > 0
        assert sources[0]["text"] == "MCP 서버 개발 가이드 - Lesson 1"
    
    def test_reset_sources(self, mock_vector_store_with_data):
        """Test resetting sources"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store_with_data)
        manager.register_tool(tool)
        
        # Execute search and verify sources exist
        manager.execute_tool("search_course_content", query="test")
        assert len(manager.get_last_sources()) > 0
        
        # Reset sources and verify they are cleared
        manager.reset_sources()
        assert len(manager.get_last_sources()) == 0