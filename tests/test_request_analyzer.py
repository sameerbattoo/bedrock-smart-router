"""Tests for the request analyzer."""

from bedrock_smart_router.models import Complexity
from bedrock_smart_router.request_analyzer import RequestAnalyzer


def _msg(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"text": text}]}]


class TestRequestAnalyzer:
    def setup_method(self):
        self.analyzer = RequestAnalyzer()

    def test_simple_greeting(self):
        result = self.analyzer.analyze(_msg("Hello, how are you?"))
        assert result.complexity == Complexity.SIMPLE

    def test_moderate_question(self):
        result = self.analyzer.analyze(
            _msg("Summarize the key differences between S3 and EBS storage.")
        )
        assert result.complexity in (Complexity.SIMPLE, Complexity.MODERATE)

    def test_complex_code_task(self):
        result = self.analyzer.analyze(
            _msg(
                "Write a Python function that implements a binary search tree "
                "with insert, delete, and search operations. Include type hints "
                "and handle edge cases. Then analyze the time complexity of each "
                "operation step by step."
            )
        )
        assert result.complexity in (Complexity.MODERATE, Complexity.COMPLEX, Complexity.REASONING)
        assert result.is_code_task

    def test_reasoning_task(self):
        result = self.analyzer.analyze(
            _msg(
                "Analyze and evaluate the trade-offs between microservices and "
                "monolithic architecture. Compare and contrast their scalability, "
                "maintainability, and deployment complexity. Reason through each "
                "dimension step by step and prove which is better for a startup."
            )
        )
        assert result.complexity == Complexity.REASONING

    def test_vision_detection(self):
        msgs = [{"role": "user", "content": [
            {"text": "What is in this image?"},
            {"image": {"format": "png", "source": {"bytes": b"fake"}}},
        ]}]
        result = self.analyzer.analyze(msgs)
        assert result.requires_vision

    def test_tool_use_detection(self):
        result = self.analyzer.analyze(
            _msg("Call the function to get weather data"),
            tool_config={"tools": [{"name": "get_weather"}]},
        )
        assert result.requires_tool_use

    def test_multi_turn(self):
        msgs = [
            {"role": "user", "content": [{"text": "Hi"}]},
            {"role": "assistant", "content": [{"text": "Hello!"}]},
            {"role": "user", "content": [{"text": "Tell me about AWS"}]},
        ]
        result = self.analyzer.analyze(msgs)
        assert result.is_multi_turn
        assert result.conversation_turn_count == 2

    def test_token_estimation(self):
        text = "a" * 4000  # ~1000 tokens
        result = self.analyzer.analyze(_msg(text))
        assert 800 <= result.estimated_input_tokens <= 1200
