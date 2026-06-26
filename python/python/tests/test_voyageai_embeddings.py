# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

"""Unit tests for VoyageAI embedding function.

These tests verify model registration and configuration without requiring API calls.
"""

import pyarrow as pa
import pytest
from unittest.mock import MagicMock, patch

from lancedb.embeddings import get_registry


@pytest.fixture(autouse=True)
def reset_voyageai_client():
    """Reset VoyageAI client before and after each test to avoid state pollution."""
    from lancedb.embeddings.voyageai import VoyageAIEmbeddingFunction

    VoyageAIEmbeddingFunction.client = None
    yield
    VoyageAIEmbeddingFunction.client = None


class TestVoyageAIModelRegistration:
    """Tests for VoyageAI model registration and configuration."""

    @pytest.fixture
    def mock_voyageai_client(self):
        """Mock VoyageAI client to avoid API calls."""
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            with patch("lancedb.embeddings.voyageai.attempt_import_or_raise") as mock:
                mock_client = MagicMock()
                mock_voyageai = MagicMock()
                mock_voyageai.Client.return_value = mock_client
                mock.return_value = mock_voyageai
                yield mock_client

    def test_voyageai_registered(self):
        """Test that VoyageAI is registered in the embedding function registry."""
        registry = get_registry()
        assert registry.get("voyageai") is not None

    @pytest.mark.parametrize(
        "model_name,expected_dims",
        [
            # Voyage-4 series (all 1024 dims)
            ("voyage-4", 1024),
            ("voyage-4-lite", 1024),
            ("voyage-4-large", 1024),
            # Voyage-3 series
            ("voyage-3", 1024),
            ("voyage-3-lite", 512),
            # Domain-specific models
            ("voyage-finance-2", 1024),
            ("voyage-multilingual-2", 1024),
            ("voyage-law-2", 1024),
            ("voyage-code-2", 1536),
            # Multimodal
            ("voyage-multimodal-3", 1024),
        ],
    )
    def test_model_dimensions(self, model_name, expected_dims, mock_voyageai_client):
        """Test that each model returns the correct dimensions."""
        registry = get_registry()
        func = registry.get("voyageai").create(name=model_name)
        assert func.ndims() == expected_dims, (
            f"Model {model_name} should have {expected_dims} dimensions"
        )

    def test_unsupported_model_raises_error(self, mock_voyageai_client):
        """Test that unsupported models raise ValueError."""
        registry = get_registry()
        func = registry.get("voyageai").create(name="unsupported-model")
        with pytest.raises(ValueError, match="not supported"):
            func.ndims()

    @pytest.mark.parametrize(
        "model_name",
        [
            "voyage-4",
            "voyage-4-lite",
            "voyage-4-large",
        ],
    )
    def test_voyage4_models_are_text_models(self, model_name, mock_voyageai_client):
        """Test that voyage-4 models are classified as text models (not multimodal)."""
        registry = get_registry()
        func = registry.get("voyageai").create(name=model_name)
        assert not func._is_multimodal_model(model_name), (
            f"{model_name} should be a text model, not multimodal"
        )

    def test_voyage4_models_in_text_embedding_list(self, mock_voyageai_client):
        """Test that voyage-4 models are in the text_embedding_models list."""
        registry = get_registry()
        func = registry.get("voyageai").create(name="voyage-4")
        assert "voyage-4" in func.text_embedding_models
        assert "voyage-4-lite" in func.text_embedding_models
        assert "voyage-4-large" in func.text_embedding_models

    def test_voyage4_models_not_in_multimodal_list(self, mock_voyageai_client):
        """Test that voyage-4 models are NOT in the multimodal_embedding_models list."""
        registry = get_registry()
        func = registry.get("voyageai").create(name="voyage-4")
        assert "voyage-4" not in func.multimodal_embedding_models
        assert "voyage-4-lite" not in func.multimodal_embedding_models
        assert "voyage-4-large" not in func.multimodal_embedding_models


class TestVoyageContext4Model:
    """Tests for the voyage-context-4 contextualized embedding model."""

    @pytest.fixture
    def func(self):
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            yield get_registry().get("voyageai").create(name="voyage-context-4")

    def test_registered_as_contextual(self, func):
        """voyage-context-4 should be recognized as a contextualized model."""
        assert func._is_contextual_model("voyage-context-4")
        assert "voyage-context-4" in func.contextual_embedding_models

    def test_dimensions(self, func):
        """voyage-context-4 has 1024 dimensions by default."""
        assert func.ndims() == 1024

    def test_source_embeddings_use_flat_input_and_chunk_size(self, func):
        """Document embeddings must pass a flat List[str] and chunk_size=32000."""
        from lancedb.embeddings.voyageai import VoyageAIEmbeddingFunction

        mock_client = MagicMock()
        mock_client.tokenize.return_value = [[1, 2], [3, 4]]
        result_aa = MagicMock()
        result_aa.embeddings = [[0.1] * 1024]
        result_bb = MagicMock()
        result_bb.embeddings = [[0.2] * 1024]
        response = MagicMock()
        response.results = [result_aa, result_bb]
        mock_client.contextualized_embed.return_value = response
        VoyageAIEmbeddingFunction.client = mock_client

        embeddings = func.compute_source_embeddings(pa.array(["aa", "bb"]))

        assert len(embeddings) == 2
        call_kwargs = mock_client.contextualized_embed.call_args.kwargs
        # Input must be a flat List[str], NOT nested as [["aa", "bb"]].
        assert call_kwargs["inputs"] == ["aa", "bb"]
        # chunk_size must be the maximum (32000) with auto-chunking enabled.
        assert call_kwargs["chunk_size"] == 32000
        assert call_kwargs["enable_auto_chunking"] is True
        assert call_kwargs["input_type"] == "document"
        assert call_kwargs["model"] == "voyage-context-4"

    def test_query_embeddings_use_flat_input(self, func):
        """Query embeddings must pass a flat List[str], not a nested list."""
        from lancedb.embeddings.voyageai import VoyageAIEmbeddingFunction

        mock_client = MagicMock()
        result = MagicMock()
        result.embeddings = [[0.3] * 1024]
        response = MagicMock()
        response.results = [result]
        mock_client.contextualized_embed.return_value = response
        VoyageAIEmbeddingFunction.client = mock_client

        embeddings = func.compute_query_embeddings("hello")

        assert len(embeddings) == 1
        call_kwargs = mock_client.contextualized_embed.call_args.kwargs
        # Flat single-element list, NOT [["hello"]].
        assert call_kwargs["inputs"] == ["hello"]
        assert call_kwargs["input_type"] == "query"
