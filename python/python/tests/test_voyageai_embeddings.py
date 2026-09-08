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
            ("voyage-4-large", 1024),
            ("voyage-4", 1024),
            ("voyage-4-lite", 1024),
            ("voyage-4-nano", 1024),
            ("voyage-code-4", 1024),
            # Contextualized chunk embeddings
            ("voyage-context-4", 1024),
            ("voyage-context-3", 1024),
            # Voyage-3 series
            ("voyage-3-large", 1024),
            ("voyage-3.5", 1024),
            ("voyage-3.5-lite", 1024),
            ("voyage-3", 1024),
            ("voyage-3-lite", 512),
            ("voyage-code-3", 1024),
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
        assert (
            func.ndims() == expected_dims
        ), f"Model {model_name} should have {expected_dims} dimensions"

    def test_unsupported_model_raises_error(self, mock_voyageai_client):
        """Test that unsupported models raise ValueError."""
        registry = get_registry()
        func = registry.get("voyageai").create(name="unsupported-model")
        with pytest.raises(ValueError, match="not supported"):
            func.ndims()

    def test_voyage3_source_embeddings_use_text_api(self, mock_voyageai_client):
        """Regression test for text table data being sent to the multimodal API."""
        mock_voyageai_client.tokenize.return_value = [["hello", "world"]]
        mock_voyageai_client.embed.return_value.embeddings = [[0.1] * 1024]

        registry = get_registry()
        func = registry.get("voyageai").create(name="voyage-3")

        embeddings = func.compute_source_embeddings("hello world")

        assert embeddings == [[0.1] * 1024]
        mock_voyageai_client.embed.assert_called_once_with(
            texts=["hello world"], model="voyage-3", input_type="document"
        )
        mock_voyageai_client.multimodal_embed.assert_not_called()

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
        assert not func._is_multimodal_model(
            model_name
        ), f"{model_name} should be a text model, not multimodal"

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

    @pytest.mark.parametrize("model_name", ["voyage-context-4", "voyage-context-3"])
    def test_contextual_models_classified(self, model_name, mock_voyageai_client):
        """Contextual models are recognized and listed as contextual, not text."""
        registry = get_registry()
        func = registry.get("voyageai").create(name=model_name)
        assert func._is_contextual_model(model_name)
        assert not func._is_multimodal_model(model_name)
        assert model_name in func.contextual_embedding_models


class TestVoyageAIContextualInput:
    """Tests for the contextualized_embed input handling (both spec formats)."""

    def test_sanitize_flat_list_becomes_single_document(self):
        from lancedb.embeddings.voyageai import sanitize_contextual_input

        # Flat List[str] -> one document whose chunks are the inputs.
        assert sanitize_contextual_input(["a", "b", "c"]) == [["a", "b", "c"]]
        # A single str is wrapped the same way.
        assert sanitize_contextual_input("a") == [["a"]]

    def test_sanitize_nested_list_passthrough(self):
        from lancedb.embeddings.voyageai import sanitize_contextual_input

        nested = [["a", "b"], ["c"]]
        assert sanitize_contextual_input(nested) == nested

    def test_sanitize_rejects_bad_input(self):
        from lancedb.embeddings.voyageai import sanitize_contextual_input

        with pytest.raises(
            ValueError, match="List\\[str\\] or List\\[List\\[str\\]\\]"
        ):
            sanitize_contextual_input([1, 2, 3])

    @pytest.fixture
    def mock_voyageai_client(self):
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            with patch("lancedb.embeddings.voyageai.attempt_import_or_raise") as mock:
                mock_client = MagicMock()
                mock_voyageai = MagicMock()
                mock_voyageai.Client.return_value = mock_client
                mock.return_value = mock_voyageai
                yield mock_client

    def test_source_embeddings_use_nested_document_format(self, mock_voyageai_client):
        """compute_source_embeddings sends chunks as List[List[str]] and flattens."""
        mock_voyageai_client.tokenize.return_value = [["a"], ["b"]]
        doc_result = MagicMock()
        doc_result.embeddings = [[0.1] * 1024, [0.2] * 1024]
        mock_voyageai_client.contextualized_embed.return_value.results = [doc_result]

        func = get_registry().get("voyageai").create(name="voyage-context-4")
        embeddings = func.compute_source_embeddings(pa.array(["a", "b"]))

        assert embeddings == [[0.1] * 1024, [0.2] * 1024]
        _, kwargs = mock_voyageai_client.contextualized_embed.call_args
        # Flat inputs are grouped into one document (List[List[str]]).
        assert kwargs["inputs"] == [["a", "b"]]
        assert kwargs["input_type"] == "document"

    def test_query_embeddings_use_contextualized_embed(self, mock_voyageai_client):
        """compute_query_embeddings routes through contextualized_embed."""
        doc_result = MagicMock()
        doc_result.embeddings = [[0.3] * 1024]
        mock_voyageai_client.contextualized_embed.return_value.results = [doc_result]

        func = get_registry().get("voyageai").create(name="voyage-context-4")
        embeddings = func.compute_query_embeddings("hello")

        assert embeddings == [[0.3] * 1024]
        _, kwargs = mock_voyageai_client.contextualized_embed.call_args
        assert kwargs["inputs"] == [["hello"]]
        assert kwargs["input_type"] == "query"
