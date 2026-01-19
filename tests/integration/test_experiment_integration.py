"""Integration tests for the Experiment class.

These tests verify the complete end-to-end functionality of the Experiment
class by running real experiments with mocked API calls only. All other
components (analyzers, fetchers, agents, etc.) run normally.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from babel_ai.enums import AgentSelectionMethod, AnalyzerType, FetcherType
from babel_ai.experiment import Experiment
from models import (
    AgentConfig,
    AgentMetric,
    AnalyzerConfig,
    ExperimentConfig,
    FetcherConfig,
    FetcherMetric,
)
from models.api import LLMResponse
from api.interp_inference import DEFAULT_MODEL


@pytest.fixture
def test_sharegpt_data():
    """Create test ShareGPT data file."""
    # ... existing code (unchanged) ...
    test_data = [
        {
            "items": [
                {"from": "human", "value": "Hello, how are you today?"},
                {
                    "from": "assistant",
                    "value": "I'm doing well, thank you for asking!",
                },
            ]
        },
        {
            "items": [
                {"from": "human", "value": "What's the weather like?"},
                {
                    "from": "assistant",
                    "value": "I don't have access to current weather data.",
                },
                {"from": "human", "value": "That's okay, thanks anyway."},
                {
                    "from": "assistant",
                    "value": "No problem! Is there anything else I can help?",
                },
            ]
        },
    ]

    # Create temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(test_data, f)
        return f.name


@pytest.fixture
def fetcher_config(test_sharegpt_data):
    return FetcherConfig(
        fetcher=FetcherType.SHAREGPT,
        data_path=test_sharegpt_data,
        min_messages=2,
        max_messages=4,
    )


@pytest.fixture
def agent_config():
    return AgentConfig(
        model=DEFAULT_MODEL,
        device="cpu",  # Use CPU for testing
        temperature=0.7,
    )


@pytest.fixture
def integration_experiment_config(fetcher_config, agent_config):
    """Create a complete experiment config for integration testing."""

    analyzer_config = AnalyzerConfig(
        analyzer=AnalyzerType.SIMILARITY, analyze_window=3
    )

    return ExperimentConfig(
        fetcher_config=fetcher_config,
        analyzer_config=analyzer_config,
        agent_configs=[agent_config],
        agent_selection_method=AgentSelectionMethod.ROUND_ROBIN,
        max_iterations=5,  # Keep small for testing
        max_total_characters=2000,
        output_dir=None,  # Will be set by test
    )


class TestExperimentIntegration:
    """Integration tests for the Experiment class."""

    @patch("babel_ai.agent.InterpInference")
    def test_full_experiment_workflow(
        self,
        mock_interp_class,
        integration_experiment_config,
        fetcher_config,
        agent_config,
        tmp_path,
    ):
        """Test the complete experiment workflow end-to-end."""
        # Setup mock InterpInference to return realistic responses
        mock_interp_instance = MagicMock()
        mock_interp_class.return_value = mock_interp_instance
        
        test_response_content = (
            "This is a test response from the NNsight model. "
            "It contains some realistic content for testing purposes."
        )
        mock_interp_instance.generate_from_messages.return_value = LLMResponse(
            content=test_response_content,
            input_token_count=50,
            output_token_count=20,
        )

        # Set output directory to temp path
        integration_experiment_config.output_dir = str(tmp_path)

        # Create and run experiment
        experiment = Experiment(integration_experiment_config)
        results = experiment.run()

        # Verify experiment ran successfully
        assert isinstance(results, list)
        assert len(results) > 0

        # Verify we have both fetcher and agent metrics
        fetcher_metrics = [r for r in results if isinstance(r, FetcherMetric)]
        agent_metrics = [r for r in results if isinstance(r, AgentMetric)]

        assert len(fetcher_metrics) >= 2  # From test data
        assert len(agent_metrics) >= 1  # At least one agent response

        # Verify fetcher metrics have correct data
        for metric in fetcher_metrics:
            assert metric.iteration >= 0
            assert isinstance(metric.timestamp, datetime)
            assert metric.role in ["human", "assistant", "user"]
            assert metric.content in [
                "Hello, how are you today?",
                "I'm doing well, thank you for asking!",
                "What's the weather like?",
                "I don't have access to current weather data.",
                "That's okay, thanks anyway.",
                "No problem! Is there anything else I can help?",
            ]
            assert metric.fetcher_config == fetcher_config

        # Verify agent metrics have correct data
        for metric in agent_metrics:
            assert metric.iteration >= len(fetcher_metrics)
            assert isinstance(metric.timestamp, datetime)
            assert metric.agent_id is not None
            assert metric.content == test_response_content
            assert len(metric.content) > 0
            assert metric.agent_config == agent_config

        # Verify InterpInference was initialized correctly
        mock_interp_class.assert_called_with(
            model=DEFAULT_MODEL,
            device="cpu",
        )
        
        # Verify generate_from_messages was called
        mock_interp_instance.generate_from_messages.assert_called()

        # Cleanup
        Path(fetcher_config.data_path).unlink()