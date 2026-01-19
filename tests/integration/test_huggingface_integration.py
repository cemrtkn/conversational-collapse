"""Integration tests for HuggingFace Hub external dependencies.

These tests verify that external models and resources required by the
project are available and accessible on the HuggingFace Hub.
"""

import pytest
from huggingface_hub import HfApi, repo_exists
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from api.interp_inference import DEFAULT_MODEL


class TestHuggingFaceIntegration:
    """Tests for HuggingFace Hub external dependencies."""

    def test_default_model_exists_on_hub(self):
        """Test that DEFAULT_MODEL repository exists on HuggingFace Hub."""
        assert repo_exists(DEFAULT_MODEL), (
            f"DEFAULT_MODEL '{DEFAULT_MODEL}' does not exist on "
            "HuggingFace Hub"
        )

    def test_default_model_is_accessible(self):
        """Test that DEFAULT_MODEL metadata is
        fetchable (checks access permissions)."""
        api = HfApi()
        try:
            info = api.model_info(DEFAULT_MODEL)
            assert info.id == DEFAULT_MODEL
        except RepositoryNotFoundError:
            pytest.fail(
                f"DEFAULT_MODEL '{DEFAULT_MODEL}' not found on "
                "HuggingFace Hub"
            )
        except GatedRepoError:
            pytest.skip(
                f"DEFAULT_MODEL '{DEFAULT_MODEL}' is gated - "
                "requires authentication"
            )
