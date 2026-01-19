"""Tests for the Agent class."""

from unittest.mock import MagicMock, patch

import pytest

from babel_ai.agent import Agent
from models import AgentConfig
from models.api import LLMResponse
from api.interp_inference import DEFAULT_MODEL


@pytest.fixture
def sample_agent_config():
    """Create a sample AgentConfig for testing."""
    return AgentConfig(
        model=DEFAULT_MODEL,
        device="cpu",
        temperature=0.8,
        max_new_tokens=150,
        top_p=0.9,
    )


@pytest.fixture
def sample_messages():
    """Create sample conversation messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What's the weather like?"},
    ]


class TestAgent:
    """Test the Agent class."""

    @patch("babel_ai.agent.InterpInference")
    def test_agent_initialization(
        self, mock_interp_inference, sample_agent_config
    ):
        """Test that Agent initializes correctly with AgentConfig."""
        agent = Agent(sample_agent_config)

        # Config attributes
        assert agent.config == sample_agent_config
        assert agent.config.model == DEFAULT_MODEL
        assert agent.config.device == "cpu"
        assert agent.config.system_prompt is None
        assert agent.config.temperature == 0.8
        assert agent.config.max_new_tokens == 150
        assert agent.config.top_p == 0.9

        # Explicit attributes
        assert agent.id is not None
        assert agent.model == DEFAULT_MODEL
        assert agent.system_prompt is None

        # Verify InterpInference was initialized correctly
        mock_interp_inference.assert_called_once_with(
            model=DEFAULT_MODEL,
            device="cpu",
        )

    @patch("babel_ai.agent.InterpInference")
    def test_agent_initialization_with_system_prompt(
        self, mock_interp_inference
    ):
        """Test that Agent initializes correctly with system prompt."""
        config = AgentConfig(
            model=DEFAULT_MODEL,
            device="cpu",
            system_prompt="You are a helpful assistant.",
            temperature=0.0,
        )

        agent = Agent(config)

        assert agent.system_prompt == "You are a helpful assistant."
        assert agent.config.system_prompt == "You are a helpful assistant."

    @patch("babel_ai.agent.InterpInference")
    def test_generate_response_calls_model(
        self, mock_interp_class, sample_agent_config, sample_messages
    ):
        """Test that generate_response calls
        the model with correct parameters."""
        # Setup mock
        mock_interp_instance = MagicMock()
        mock_interp_class.return_value = mock_interp_instance
        mock_interp_instance.generate_from_messages.return_value = LLMResponse(
            content="Test response from model",
            input_token_count=10,
            output_token_count=5,
        )

        agent = Agent(sample_agent_config)
        response = agent.generate_response(sample_messages)

        # Verify the response
        assert response == "Test response from model"

        # Verify generate_from_messages was called with correct parameters
        mock_interp_instance.generate_from_messages.assert_called_once_with(
            messages=sample_messages,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.9,
        )

    @patch("babel_ai.agent.InterpInference")
    def test_generate_response_with_system_prompt(self, mock_interp_class):
        """Test generate_response with system prompt."""
        # Setup mock
        mock_interp_instance = MagicMock()
        mock_interp_class.return_value = mock_interp_instance
        mock_interp_instance.generate_from_messages.return_value = LLMResponse(
            content="Test response",
            input_token_count=10,
            output_token_count=5,
        )

        config = AgentConfig(
            model=DEFAULT_MODEL,
            device="cpu",
            system_prompt="You are a helpful assistant.",
            temperature=0.7,
        )
        agent = Agent(config)
        messages = [{"role": "user", "content": "Hello"}]

        response = agent.generate_response(messages)

        # Verify response
        assert response == "Test response"

        # Verify generate_from_messages was called with system prompt prepended
        expected_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        mock_interp_instance.generate_from_messages.assert_called_once_with(
            messages=expected_messages,
            max_new_tokens=256,
            temperature=0.7,
            top_p=1.0,
        )

    @patch("babel_ai.agent.InterpInference")
    def test_generate_response_without_system_prompt(self, mock_interp_class):
        """Test generate_response without system prompt."""
        # Setup mock
        mock_interp_instance = MagicMock()
        mock_interp_class.return_value = mock_interp_instance
        mock_interp_instance.generate_from_messages.return_value = LLMResponse(
            content="Test response",
            input_token_count=10,
            output_token_count=5,
        )

        config = AgentConfig(
            model=DEFAULT_MODEL,
            device="cpu",
            # No system_prompt
            temperature=0.7,
        )
        agent = Agent(config)
        messages = [{"role": "user", "content": "Hello"}]

        response = agent.generate_response(messages)

        # Verify response
        assert response == "Test response"

        # Verify generate_from_messages was called with original messages
        mock_interp_instance.generate_from_messages.assert_called_once_with(
            messages=messages,  # No system prompt added
            max_new_tokens=256,
            temperature=0.7,
            top_p=1.0,
        )

    def test_define_msg_tree_empty_list(self):
        """Test _define_msg_tree with empty list input."""
        result = Agent._define_msg_tree([])
        result_list = list(result)  # Convert iterator to list

        assert result_list == []

    def test_define_msg_tree_single_message(self):
        """Test _define_msg_tree with single message."""
        messages = [{"content": "Hello world"}]
        result = Agent._define_msg_tree(messages)
        result_list = list(result)  # Convert iterator to list

        expected = [{"role": "user", "content": "Hello world"}]
        assert result_list == expected

    def test_define_msg_tree_four_messages(self):
        """Test _define_msg_tree with four messages."""
        messages = [
            {"content": "Message one"},
            {"content": "Message two"},
            {"content": "Message three"},
            {"content": "Message four"},
        ]
        result = Agent._define_msg_tree(messages)
        result_list = list(result)  # Convert iterator to list

        expected = [
            {"role": "assistant", "content": "Message one"},
            {"role": "user", "content": "Message two"},
            {"role": "assistant", "content": "Message three"},
            {"role": "user", "content": "Message four"},
        ]
        assert result_list == expected

    def test_define_msg_tree_ignores_original_roles(self):
        """Test that _define_msg_tree ignores original role keys."""
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
            {"role": "system", "content": "Third message"},
        ]
        result = Agent._define_msg_tree(messages)
        result_list = list(result)  # Convert iterator to list

        # Original roles should be ignored, new roles assigned based on
        # position
        expected = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
            {"role": "user", "content": "Third message"},
        ]
        assert result_list == expected

    def test_define_msg_tree_with_extra_keys(self):
        """Test _define_msg_tree with messages containing extra keys."""
        messages = [
            {"content": "First", "timestamp": "2023-01-01", "id": 1},
            {"content": "Second", "metadata": {"key": "value"}},
        ]
        result = Agent._define_msg_tree(messages)
        result_list = list(result)  # Convert iterator to list

        # Should only have role and content keys
        for msg in result_list:
            assert set(msg.keys()) == {"role", "content"}

        assert result_list[0]["content"] == "First"
        assert result_list[1]["content"] == "Second"
