"""Tests for the InterpInference class."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from api.interp_inference import DEFAULT_MODEL, InterpInference
from models.api import LLMResponse


class TestInterpInference:
    """Test the InterpInference class."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted prompt"
        tokenizer.decode.return_value = "Generated response text"
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
        return tokenizer

    @pytest.fixture
    def sample_messages(self):
        """Create sample conversation messages for testing."""
        return [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well!"},
            {"role": "user", "content": "What is 2+2?"},
        ]

    @patch("api.interp_inference.LanguageModel")
    def test_initialization_with_defaults(self, mock_language_model):
        """Test that InterpInference initializes
        correctly with default parameters."""
        _ = InterpInference(model=DEFAULT_MODEL)

        mock_language_model.assert_called_once()
        call_kwargs = mock_language_model.call_args
        assert call_kwargs[0][0] == DEFAULT_MODEL
        assert call_kwargs[1]["torch_dtype"] == torch.float16

    @patch("api.interp_inference.LanguageModel")
    def test_initialization_with_custom_device(self, mock_language_model):
        """Test initialization with custom device."""
        _ = InterpInference(model=DEFAULT_MODEL, device="cuda:1")

        mock_language_model.assert_called_once()
        call_kwargs = mock_language_model.call_args
        assert call_kwargs[1]["device_map"] == "cuda:1"

    @patch("api.interp_inference.LanguageModel")
    def test_initialization_with_cpu_device(self, mock_language_model):
        """Test initialization with CPU device sets device_map to None."""
        _ = InterpInference(model=DEFAULT_MODEL, device="cpu")

        mock_language_model.assert_called_once()
        call_kwargs = mock_language_model.call_args
        assert call_kwargs[1]["device_map"] is None

    @patch("api.interp_inference.LanguageModel")
    def test_generate_calls_model_correctly(self, mock_language_model):
        """Test that generate calls the underlying
        model with correct parameters."""
        # Setup mock
        mock_model_instance = MagicMock()
        mock_language_model.return_value = mock_model_instance

        # Mock the context manager for generate
        mock_tracer = MagicMock()
        mock_model_instance.generate.return_value.__enter__ = MagicMock(
            return_value=mock_tracer
        )
        mock_model_instance.generate.return_value.__exit__ = MagicMock(
            return_value=False
        )
        mock_model_instance.generator.output.save.return_value = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8]]
        )
        mock_model_instance.tokenizer.decode.return_value = "Generated text"

        interp = InterpInference(model=DEFAULT_MODEL)
        result = interp.generate(
            "Test prompt", max_new_tokens=100, input_len=3
        )

        mock_model_instance.generate.assert_called_once()
        assert result == "Generated text"

    @patch("api.interp_inference.LanguageModel")
    def test_generate_from_messages_with_chat_template(
        self, mock_language_model, mock_tokenizer, sample_messages
    ):
        """Test generate_from_messages uses chat template when available."""
        # Setup mock
        mock_model_instance = MagicMock()
        mock_language_model.return_value = mock_model_instance
        mock_model_instance.tokenizer = mock_tokenizer
        mock_model_instance.tokenizer.apply_chat_template.return_value = (
            "formatted prompt"
        )
        mock_model_instance.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]])
        }

        # Mock generate context manager
        mock_model_instance.generate.return_value.__enter__ = MagicMock()
        mock_model_instance.generate.return_value.__exit__ = MagicMock(
            return_value=False
        )
        mock_model_instance.generator.output.save.return_value = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8]]
        )
        mock_model_instance.tokenizer.decode.return_value = "Response text"

        interp = InterpInference(model=DEFAULT_MODEL)
        response = interp.generate_from_messages(
            sample_messages, max_new_tokens=50
        )

        # Verify chat template was called
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            sample_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Verify response is LLMResponse
        assert isinstance(response, LLMResponse)
        assert response.content == "Response text"
        assert response.input_token_count == 5

    @patch("api.interp_inference.LanguageModel")
    def test_generate_from_messages_fallback_without_chat_template(
        self, mock_language_model, sample_messages
    ):
        """Test generate_from_messages falls back
        to concatenation without chat template."""
        # Setup mock without apply_chat_template
        mock_model_instance = MagicMock()
        mock_language_model.return_value = mock_model_instance
        mock_model_instance.tokenizer = MagicMock(spec=["decode", "__call__"])
        mock_model_instance.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]])
        }

        # Mock generate context manager
        mock_model_instance.generate.return_value.__enter__ = MagicMock()
        mock_model_instance.generate.return_value.__exit__ = MagicMock(
            return_value=False
        )
        mock_model_instance.generator.output.save.return_value = torch.tensor(
            [[1, 2, 3, 4, 5]]
        )
        mock_model_instance.tokenizer.decode.return_value = "Fallback response"

        interp = InterpInference(model=DEFAULT_MODEL)
        response = interp.generate_from_messages(sample_messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Fallback response"

    @patch("api.interp_inference.LanguageModel")
    def test_tokenizer_property(self, mock_language_model, mock_tokenizer):
        """Test that tokenizer property returns the underlying tokenizer."""
        mock_model_instance = MagicMock()
        mock_language_model.return_value = mock_model_instance
        mock_model_instance.tokenizer = mock_tokenizer

        interp = InterpInference(model=DEFAULT_MODEL)

        assert interp.tokenizer == mock_tokenizer

    @patch("api.interp_inference.LanguageModel")
    def test_generate_with_custom_parameters(self, mock_language_model):
        """Test generate passes custom parameters to the model."""
        mock_model_instance = MagicMock()
        mock_language_model.return_value = mock_model_instance

        # Mock generate context manager
        mock_model_instance.generate.return_value.__enter__ = MagicMock()
        mock_model_instance.generate.return_value.__exit__ = MagicMock(
            return_value=False
        )
        mock_model_instance.generator.output.save.return_value = torch.tensor(
            [[1, 2, 3, 4]]
        )
        mock_model_instance.tokenizer.decode.return_value = "Response"

        interp = InterpInference(model=DEFAULT_MODEL)
        interp.generate(
            "Test prompt",
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=False,
            input_len=2,
        )

        mock_model_instance.generate.assert_called_once_with(
            "Test prompt",
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=False,
        )
