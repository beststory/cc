from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials 
and educational content with access to two comprehensive tools for course 
information.

Available Tools:
1. **search_course_content**: For searching specific course content and 
   detailed materials
2. **get_course_syllabus**: For getting course syllabus (title, instructor, 
   link, complete lesson list)

Tool Usage Guidelines:
- **Course syllabus/overview questions**: Use get_course_syllabus tool to 
  show course title, instructor, course link, and complete lesson list with 
  numbers and titles
- **Course content questions**: Use search_course_content tool for specific 
  lesson content or detailed materials
- **최대 2라운드의 순차적 도구 호출 지원**: 첫 번째 도구 결과를 바탕으로 
  추가 도구 호출 가능
- 복잡한 쿼리의 경우 단계별 도구 사용 권장 (예: 검색 → 강의계획서 조회)
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course syllabus questions** (강의 계획서, 강의 목록, 전체 강의): Use get_course_syllabus tool
- **Course content questions**: Use search_course_content tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or tool usage


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution_with_rounds(
                response, api_params, tool_manager, tools
            )

        # Return direct response
        return response.content[0].text

    def _handle_tool_execution_with_rounds(
        self,
        initial_response,
        base_params: Dict[str, Any],
        tool_manager,
        tools: Optional[List],
        max_rounds: int = 2,
    ):
        """
        최대 2라운드 순차적 도구 호출을 지원하는 도구 실행 처리

        Args:
            initial_response: 초기 도구 호출 요청이 포함된 응답
            base_params: 기본 API 파라미터
            tool_manager: 도구 실행 관리자
            tools: 사용 가능한 도구 목록
            max_rounds: 최대 라운드 수 (기본값: 2)

        Returns:
            최종 AI 응답 텍스트
        """
        # 메시지 체인 초기화
        messages = base_params["messages"].copy()
        current_response = initial_response

        # 최대 라운드 수만큼 반복
        for round_num in range(1, max_rounds + 1):
            # 현재 라운드의 도구 실행 처리
            messages = self._process_tool_round(
                current_response, messages, tool_manager
            )

            # 최대 라운드에 도달한 경우 최종 라운드 실행
            if round_num == max_rounds:
                return self._execute_final_round(messages, base_params)

            # 다음 라운드 실행 (도구 사용 가능)
            current_response = self._execute_round_with_tools(
                messages, base_params, tools
            )

            # 도구 사용이 없으면 응답 반환
            if current_response.stop_reason != "tool_use":
                return current_response.content[0].text

        # 예외 상황 - 최종 라운드 실행
        return self._execute_final_round(messages, base_params)

    def _process_tool_round(self, response, messages: List, tool_manager):
        """
        개별 라운드에서 도구 실행 및 메시지 체인 업데이트

        Args:
            response: Claude의 도구 호출 응답
            messages: 현재 메시지 체인
            tool_manager: 도구 실행 관리자

        Returns:
            업데이트된 메시지 체인
        """
        # AI의 도구 사용 응답 추가
        messages.append({"role": "assistant", "content": response.content})

        # 모든 도구 호출 실행 및 결과 수집
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )
                except Exception as e:
                    tool_result = str(e)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        # 도구 결과를 메시지에 추가
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        return messages

    def _execute_round_with_tools(
        self, messages: List, base_params: Dict[str, Any], tools: Optional[List]
    ):
        """
        도구 사용 가능한 상태로 다음 라운드 실행

        Args:
            messages: 현재 메시지 체인
            base_params: 기본 API 파라미터
            tools: 사용 가능한 도구 목록

        Returns:
            Claude의 응답
        """
        # 도구를 포함한 API 파라미터 준비
        round_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # 도구가 있으면 추가
        if tools:
            round_params["tools"] = tools
            round_params["tool_choice"] = {"type": "auto"}

        # API 호출
        return self.client.messages.create(**round_params)

    def _execute_final_round(self, messages: List, base_params: Dict[str, Any]):
        """
        도구 없이 최종 응답 생성

        Args:
            messages: 완전한 메시지 체인
            base_params: 기본 API 파라미터

        Returns:
            최종 AI 응답 텍스트
        """
        # 도구 없이 최종 API 호출
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls and get follow-up response.
        (기존 테스트와의 하위 호환성을 위해 유지)

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )
                except Exception as e:
                    tool_result = str(e)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
