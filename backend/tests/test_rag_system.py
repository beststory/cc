from unittest.mock import MagicMock, Mock, patch

import pytest
from rag_system import RAGSystem


class TestRAGSystem:
    """Test cases for RAG System integration"""

    @pytest.fixture
    def mock_components(self):
        """Mock all RAG system components"""
        components = {}

        # Mock DocumentProcessor
        components["document_processor"] = Mock()
        components["document_processor"].process_course_document.return_value = (
            Mock(),
            [Mock()],
        )

        # Mock VectorStore
        components["vector_store"] = Mock()
        components["vector_store"].add_course_metadata.return_value = None
        components["vector_store"].add_course_content.return_value = None
        components["vector_store"].get_existing_course_titles.return_value = []
        components["vector_store"].get_course_count.return_value = 0

        # Mock AIGenerator
        components["ai_generator"] = Mock()
        components["ai_generator"].generate_response.return_value = "Test AI response"

        # Mock SessionManager
        components["session_manager"] = Mock()
        components["session_manager"].get_conversation_history.return_value = None
        components["session_manager"].add_exchange.return_value = None

        # Mock ToolManager
        components["tool_manager"] = Mock()
        components["tool_manager"].get_tool_definitions.return_value = []
        components["tool_manager"].get_last_sources.return_value = []
        components["tool_manager"].reset_sources.return_value = None

        return components

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.ToolManager")
    @patch("rag_system.CourseSearchTool")
    @patch("rag_system.CourseSyllabulTool")
    def test_rag_system_initialization(
        self,
        mock_syllabus_tool,
        mock_search_tool,
        mock_tool_manager,
        mock_doc_processor,
        mock_vector_store,
        mock_ai_generator,
        mock_session_manager,
        test_config,
    ):
        """Test RAG system initialization"""

        # Create RAG system instance
        rag_system = RAGSystem(test_config)

        # Verify all components were initialized with correct parameters
        mock_doc_processor.assert_called_once_with(
            test_config.CHUNK_SIZE, test_config.CHUNK_OVERLAP
        )
        mock_vector_store.assert_called_once_with(
            test_config.CHROMA_PATH,
            test_config.EMBEDDING_MODEL,
            test_config.MAX_RESULTS,
        )
        mock_ai_generator.assert_called_once_with(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        mock_session_manager.assert_called_once_with(test_config.MAX_HISTORY)

        # Verify tools were registered
        mock_tool_manager.return_value.register_tool.assert_called()
        assert (
            mock_tool_manager.return_value.register_tool.call_count == 2
        )  # search_tool and syllabus_tool

    def test_query_content_question_workflow(self, mock_components, test_config):
        """Test complete workflow for content-related questions"""

        with patch.multiple(
            "rag_system",
            DocumentProcessor=lambda *args: mock_components["document_processor"],
            VectorStore=lambda *args: mock_components["vector_store"],
            AIGenerator=lambda *args: mock_components["ai_generator"],
            SessionManager=lambda *args: mock_components["session_manager"],
            ToolManager=lambda: mock_components["tool_manager"],
            CourseSearchTool=lambda *args: Mock(),
            CourseSyllabulTool=lambda *args: Mock(),
        ):

            rag_system = RAGSystem(test_config)

            # Mock tool manager behavior for content search
            mock_components["tool_manager"].get_last_sources.return_value = [
                {
                    "text": "MCP 서버 개발 가이드 - Lesson 1",
                    "link": "https://example.com/lesson1",
                }
            ]

            # Execute query
            response, sources = rag_system.query("MCP란 무엇인가요?")

            # Verify AI generator was called with correct parameters
            mock_components["ai_generator"].generate_response.assert_called_once()
            call_args = mock_components["ai_generator"].generate_response.call_args

            assert "MCP란 무엇인가요?" in call_args[1]["query"]
            assert (
                call_args[1]["tools"]
                == mock_components["tool_manager"].get_tool_definitions()
            )
            assert call_args[1]["tool_manager"] == mock_components["tool_manager"]

            # Verify sources were retrieved and reset
            mock_components["tool_manager"].get_last_sources.assert_called_once()
            mock_components["tool_manager"].reset_sources.assert_called_once()

            assert response == "Test AI response"
            assert len(sources) == 1
            assert sources[0]["text"] == "MCP 서버 개발 가이드 - Lesson 1"

    def test_query_with_session_management(self, mock_components, test_config):
        """Test query with session management"""

        with patch.multiple(
            "rag_system",
            DocumentProcessor=lambda *args: mock_components["document_processor"],
            VectorStore=lambda *args: mock_components["vector_store"],
            AIGenerator=lambda *args: mock_components["ai_generator"],
            SessionManager=lambda *args: mock_components["session_manager"],
            ToolManager=lambda: mock_components["tool_manager"],
            CourseSearchTool=lambda *args: Mock(),
            CourseSyllabulTool=lambda *args: Mock(),
        ):

            rag_system = RAGSystem(test_config)

            # Mock conversation history
            mock_components["session_manager"].get_conversation_history.return_value = (
                "Previous: Hello"
            )

            # Execute query with session ID
            response, sources = rag_system.query(
                "후속 질문입니다", session_id="test_session"
            )

            # Verify session management calls
            mock_components[
                "session_manager"
            ].get_conversation_history.assert_called_once_with("test_session")
            mock_components["session_manager"].add_exchange.assert_called_once_with(
                "test_session", "후속 질문입니다", "Test AI response"
            )

            # Verify AI generator received conversation history
            call_args = mock_components["ai_generator"].generate_response.call_args
            assert call_args[1]["conversation_history"] == "Previous: Hello"

    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("os.path.join")
    @patch("os.path.isfile")
    def test_add_course_folder_success(
        self,
        mock_isfile,
        mock_join,
        mock_listdir,
        mock_exists,
        mock_components,
        test_config,
        sample_course,
        sample_course_chunks,
    ):
        """Test successful course folder processing"""

        # Mock file system
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.pdf", "ignore.jpg"]
        mock_isfile.side_effect = lambda path: path.endswith((".txt", ".pdf"))
        mock_join.side_effect = lambda *args: "/".join(args)

        # Mock document processing
        mock_components["document_processor"].process_course_document.return_value = (
            sample_course,
            sample_course_chunks,
        )
        mock_components["vector_store"].get_existing_course_titles.return_value = set()

        with patch.multiple(
            "rag_system",
            DocumentProcessor=lambda *args: mock_components["document_processor"],
            VectorStore=lambda *args: mock_components["vector_store"],
            AIGenerator=lambda *args: mock_components["ai_generator"],
            SessionManager=lambda *args: mock_components["session_manager"],
            ToolManager=lambda: mock_components["tool_manager"],
            CourseSearchTool=lambda *args: Mock(),
            CourseSyllabulTool=lambda *args: Mock(),
        ):

            rag_system = RAGSystem(test_config)

            # Execute
            total_courses, total_chunks = rag_system.add_course_folder("/test/folder")

            # Verify results
            assert total_courses == 2
            assert total_chunks == len(sample_course_chunks) * 2

            # Verify document processing was called for valid files
            assert (
                mock_components["document_processor"].process_course_document.call_count
                == 2
            )

            # Verify vector store operations
            assert mock_components["vector_store"].add_course_metadata.call_count == 2
            assert mock_components["vector_store"].add_course_content.call_count == 2

    def test_add_course_folder_duplicate_prevention(
        self, mock_components, test_config, sample_course
    ):
        """Test prevention of duplicate course addition"""

        with (
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["course1.txt"]),
            patch("os.path.isfile", return_value=True),
            patch("os.path.join", side_effect=lambda *args: "/".join(args)),
        ):

            # Mock existing course
            mock_components["vector_store"].get_existing_course_titles.return_value = {
                sample_course.title
            }
            mock_components[
                "document_processor"
            ].process_course_document.return_value = (sample_course, [])

            with patch.multiple(
                "rag_system",
                DocumentProcessor=lambda *args: mock_components["document_processor"],
                VectorStore=lambda *args: mock_components["vector_store"],
                AIGenerator=lambda *args: mock_components["ai_generator"],
                SessionManager=lambda *args: mock_components["session_manager"],
                ToolManager=lambda: mock_components["tool_manager"],
                CourseSearchTool=lambda *args: Mock(),
                CourseSyllabulTool=lambda *args: Mock(),
            ):

                rag_system = RAGSystem(test_config)

                total_courses, total_chunks = rag_system.add_course_folder(
                    "/test/folder"
                )

                # Verify no new courses were added
                assert total_courses == 0
                assert total_chunks == 0

                # Verify vector store operations were not called
                mock_components["vector_store"].add_course_metadata.assert_not_called()
                mock_components["vector_store"].add_course_content.assert_not_called()

    def test_add_course_folder_clear_existing(self, mock_components, test_config):
        """Test clearing existing data when requested"""

        with (
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=[]),
            patch.multiple(
                "rag_system",
                DocumentProcessor=lambda *args: mock_components["document_processor"],
                VectorStore=lambda *args: mock_components["vector_store"],
                AIGenerator=lambda *args: mock_components["ai_generator"],
                SessionManager=lambda *args: mock_components["session_manager"],
                ToolManager=lambda: mock_components["tool_manager"],
                CourseSearchTool=lambda *args: Mock(),
                CourseSyllabulTool=lambda *args: Mock(),
            ),
        ):

            rag_system = RAGSystem(test_config)

            rag_system.add_course_folder("/test/folder", clear_existing=True)

            # Verify clear operation was called
            mock_components["vector_store"].clear_all_data.assert_called_once()

    def test_add_course_folder_nonexistent_folder(self, mock_components, test_config):
        """Test handling of non-existent folder"""

        with (
            patch("os.path.exists", return_value=False),
            patch.multiple(
                "rag_system",
                DocumentProcessor=lambda *args: mock_components["document_processor"],
                VectorStore=lambda *args: mock_components["vector_store"],
                AIGenerator=lambda *args: mock_components["ai_generator"],
                SessionManager=lambda *args: mock_components["session_manager"],
                ToolManager=lambda: mock_components["tool_manager"],
                CourseSearchTool=lambda *args: Mock(),
                CourseSyllabulTool=lambda *args: Mock(),
            ),
        ):

            rag_system = RAGSystem(test_config)

            total_courses, total_chunks = rag_system.add_course_folder(
                "/nonexistent/folder"
            )

            assert total_courses == 0
            assert total_chunks == 0

    def test_add_single_course_document_success(
        self, mock_components, test_config, sample_course, sample_course_chunks
    ):
        """Test successful single document addition"""

        mock_components["document_processor"].process_course_document.return_value = (
            sample_course,
            sample_course_chunks,
        )

        with patch.multiple(
            "rag_system",
            DocumentProcessor=lambda *args: mock_components["document_processor"],
            VectorStore=lambda *args: mock_components["vector_store"],
            AIGenerator=lambda *args: mock_components["ai_generator"],
            SessionManager=lambda *args: mock_components["session_manager"],
            ToolManager=lambda: mock_components["tool_manager"],
            CourseSearchTool=lambda *args: Mock(),
            CourseSyllabulTool=lambda *args: Mock(),
        ):

            rag_system = RAGSystem(test_config)

            course, chunk_count = rag_system.add_course_document("/test/course.txt")

            assert course == sample_course
            assert chunk_count == len(sample_course_chunks)

            # Verify processing and storage calls
            mock_components[
                "document_processor"
            ].process_course_document.assert_called_once_with("/test/course.txt")
            mock_components["vector_store"].add_course_metadata.assert_called_once_with(
                sample_course
            )
            mock_components["vector_store"].add_course_content.assert_called_once_with(
                sample_course_chunks
            )

    def test_add_single_course_document_error(self, mock_components, test_config):
        """Test error handling in single document addition"""

        mock_components["document_processor"].process_course_document.side_effect = (
            Exception("Processing error")
        )

        with patch.multiple(
            "rag_system",
            DocumentProcessor=lambda *args: mock_components["document_processor"],
            VectorStore=lambda *args: mock_components["vector_store"],
            AIGenerator=lambda *args: mock_components["ai_generator"],
            SessionManager=lambda *args: mock_components["session_manager"],
            ToolManager=lambda: mock_components["tool_manager"],
            CourseSearchTool=lambda *args: Mock(),
            CourseSyllabulTool=lambda *args: Mock(),
        ):

            rag_system = RAGSystem(test_config)

            course, chunk_count = rag_system.add_course_document("/test/course.txt")

            assert course is None
            assert chunk_count == 0

    def test_get_course_analytics(self, mock_components, test_config):
        """Test course analytics retrieval"""

        mock_components["vector_store"].get_course_count.return_value = 3
        mock_components["vector_store"].get_existing_course_titles.return_value = [
            "Course 1",
            "Course 2",
            "Course 3",
        ]

        with patch.multiple(
            "rag_system",
            DocumentProcessor=lambda *args: mock_components["document_processor"],
            VectorStore=lambda *args: mock_components["vector_store"],
            AIGenerator=lambda *args: mock_components["ai_generator"],
            SessionManager=lambda *args: mock_components["session_manager"],
            ToolManager=lambda: mock_components["tool_manager"],
            CourseSearchTool=lambda *args: Mock(),
            CourseSyllabulTool=lambda *args: Mock(),
        ):

            rag_system = RAGSystem(test_config)

            analytics = rag_system.get_course_analytics()

            assert analytics["total_courses"] == 3
            assert len(analytics["course_titles"]) == 3
            assert "Course 1" in analytics["course_titles"]

    def test_tool_registration_and_integration(self, test_config):
        """Test that tools are properly registered and integrated"""

        with patch.multiple(
            "rag_system",
            DocumentProcessor=Mock,
            VectorStore=Mock,
            AIGenerator=Mock,
            SessionManager=Mock,
        ) as mocks:

            # Create actual ToolManager to test real integration
            rag_system = RAGSystem(test_config)

            # Verify tools are registered
            tool_definitions = rag_system.tool_manager.get_tool_definitions()

            assert len(tool_definitions) == 2
            tool_names = [tool_def["name"] for tool_def in tool_definitions]
            assert "search_course_content" in tool_names
            assert "get_course_syllabus" in tool_names

            # Verify tools can be executed (with mocked dependencies)
            assert "search_course_content" in rag_system.tool_manager.tools
            assert "get_course_syllabus" in rag_system.tool_manager.tools
