# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

"""Unit tests for VoyageAI embedding function.

These tests verify model registration and configuration without requiring API calls.
"""

import numpy as np
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
            ("voyage-code-4", 1024),
            # Contextualized
            ("voyage-context-4", 1024),
            ("voyage-context-3", 1024),
            # Voyage-3 series
            ("voyage-3-large", 1024),
            ("voyage-3.5", 1024),
            ("voyage-3.5-lite", 1024),
            ("voyage-code-3", 1024),
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


@pytest.mark.parametrize(
    "inputs",
    [
        ["first", "second"],
        np.array(["first", "second"]),
        pa.array(["first", "second"]),
        pa.chunked_array([["first"], ["second"]]),
    ],
    ids=["list", "numpy", "arrow", "chunked-arrow"],
)
def test_text_source_embeddings_accept_batch_containers(inputs):
    from lancedb.embeddings.voyageai import VoyageAIEmbeddingFunction

    client = MagicMock()
    client.tokenize.return_value = [[1], [2]]
    client.embed.return_value.embeddings = [[0.1, 0.2], [0.3, 0.4]]
    VoyageAIEmbeddingFunction.client = client
    func = get_registry().get("voyageai").create(name="voyage-3")

    assert func.compute_source_embeddings(inputs) == [[0.1, 0.2], [0.3, 0.4]]
    client.embed.assert_called_once_with(
        texts=["first", "second"], model="voyage-3", input_type="document"
    )


@pytest.mark.parametrize("inputs", [[], np.array([], dtype=str)])
def test_text_source_embeddings_accept_empty_batch(inputs):
    from lancedb.embeddings.voyageai import VoyageAIEmbeddingFunction

    client = MagicMock()
    VoyageAIEmbeddingFunction.client = client
    func = get_registry().get("voyageai").create(name="voyage-3")

    assert func.compute_source_embeddings(inputs) == []
    client.tokenize.assert_not_called()
    client.embed.assert_not_called()


@pytest.mark.parametrize(
    "inputs",
    [["text", 42], np.array(["text", None]), np.array([["first", "second"]])],
)
def test_text_source_embeddings_reject_non_text_batch_entries(inputs):
    from lancedb.embeddings.voyageai import VoyageAIEmbeddingFunction

    client = MagicMock()
    VoyageAIEmbeddingFunction.client = client
    func = get_registry().get("voyageai").create(name="voyage-3")

    with pytest.raises(ValueError, match="Each input should be str"):
        func.compute_source_embeddings(inputs)
    client.tokenize.assert_not_called()
    client.embed.assert_not_called()


@pytest.mark.parametrize("model_name", ["voyage-code-4", "voyage-context-4"])
def test_output_dimension_is_sent_only_when_requested(model_name):
    from lancedb.embeddings.voyageai import VoyageAIEmbeddingFunction

    client = MagicMock()
    client.tokenize.return_value = [[1]]
    client.embed.return_value.embeddings = [[0.1] * 256]
    client.contextualized_embed.return_value.results = [MagicMock(embeddings=[[0.1]])]
    VoyageAIEmbeddingFunction.client = client

    func = get_registry().get("voyageai").create(name=model_name)
    func.compute_source_embeddings(["text"])
    for call in (client.embed.call_args, client.contextualized_embed.call_args):
        if call is not None:
            assert "output_dimension" not in call.kwargs

    func = get_registry().get("voyageai").create(name=model_name, output_dimension=256)
    assert func.ndims() == 256
    func.compute_source_embeddings(["text"])
    call = client.embed.call_args or client.contextualized_embed.call_args
    assert call.kwargs["output_dimension"] == 256


@pytest.mark.parametrize(
    "model_name,output_dimension",
    [("voyage-3", 256), ("voyage-code-4", 128)],
    ids=["model-without-flexible-dimensions", "invalid-dimension"],
)
def test_invalid_output_dimension_raises(model_name, output_dimension):
    func = (
        get_registry()
        .get("voyageai")
        .create(name=model_name, output_dimension=output_dimension)
    )
    with pytest.raises(ValueError, match="output_dimension"):
        func.ndims()


def _contextualized_result(chunks_per_document):
    result = MagicMock()
    result.results = [
        MagicMock(embeddings=[[float(i)] * 2 for _ in range(n)])
        for i, n in enumerate(chunks_per_document)
    ]
    return result


def test_contextualized_documents_use_flat_inputs_with_auto_chunking():
    from lancedb.embeddings.voyageai import VoyageAIEmbeddingFunction

    client = MagicMock()
    client.tokenize.return_value = [[1], [2]]
    client.contextualized_embed.return_value = _contextualized_result([1, 1])
    VoyageAIEmbeddingFunction.client = client
    func = get_registry().get("voyageai").create(name="voyage-context-4")

    assert func.compute_source_embeddings(["first", "second"]) == [
        [0.0, 0.0],
        [1.0, 1.0],
    ]
    client.contextualized_embed.assert_called_once_with(
        inputs=["first", "second"],
        model="voyage-context-4",
        input_type="document",
        enable_auto_chunking=True,
        chunk_size=32_000,
    )


def test_contextualized_query_does_not_use_auto_chunking():
    from lancedb.embeddings.voyageai import VoyageAIEmbeddingFunction

    client = MagicMock()
    client.contextualized_embed.return_value = _contextualized_result([1])
    VoyageAIEmbeddingFunction.client = client
    func = get_registry().get("voyageai").create(name="voyage-context-4")

    func.compute_query_embeddings("question")
    client.contextualized_embed.assert_called_once_with(
        inputs=[["question"]], model="voyage-context-4", input_type="query"
    )


def test_contextualized_document_split_into_chunks_raises():
    from lancedb.embeddings.voyageai import VoyageAIEmbeddingFunction

    client = MagicMock()
    client.tokenize.return_value = [[1], [2]]
    client.contextualized_embed.return_value = _contextualized_result([1, 2])
    VoyageAIEmbeddingFunction.client = client
    func = get_registry().get("voyageai").create(name="voyage-context-3")

    with pytest.raises(ValueError, match="Input 1 was split into 2 chunks"):
        func.compute_source_embeddings(["first", "second"])


def _batches(model_name, token_counts):

    client = MagicMock()
    client.tokenize.side_effect = lambda texts, model: [[0] * n for n in token_counts]
    func = get_registry().get("voyageai").create(name=model_name)
    texts = [str(i) for i in range(len(token_counts))]
    return client, texts, list(func._build_batches(client, texts))


def test_batches_split_at_the_model_token_limit_and_keep_order():
    # voyage-3 allows 120K tokens per request
    client, texts, batches = _batches("voyage-3", [60_000, 60_000, 1, 119_999, 1])
    assert batches == [["0", "1"], ["2", "3"], ["4"]]
    assert [t for batch in batches for t in batch] == texts
    client.tokenize.assert_called_once_with(texts, model="voyage-3")


def test_batches_use_the_limit_of_each_model():
    assert len(_batches("voyage-4", [200_000, 100_000])[2]) == 1  # 320K
    assert len(_batches("voyage-4", [200_000, 130_000])[2]) == 2
    assert len(_batches("voyage-4-lite", [600_000, 400_000])[2]) == 1  # 1M
    assert len(_batches("voyage-3-large", [100_000, 30_000])[2]) == 2  # 120K


def test_batches_are_limited_to_1000_inputs():
    _, _, batches = _batches("voyage-4-lite", [1] * 2001)
    assert [len(batch) for batch in batches] == [1000, 1000, 1]


def test_batches_without_a_documented_token_limit_only_split_by_input_count():
    client, _, batches = _batches("voyage-code-4", [1_000_000] * 1001)
    assert [len(batch) for batch in batches] == [1000, 1]
    client.tokenize.assert_not_called()


def test_batches_reject_a_model_without_a_token_limit_entry():

    func = get_registry().get("voyageai").create(name="unsupported-model")
    with pytest.raises(ValueError, match="not supported"):
        list(func._build_batches(MagicMock(), ["text"]))


def test_every_supported_model_has_a_token_limit_entry():
    from lancedb.embeddings.voyageai import VOYAGE_TOTAL_TOKEN_LIMITS

    func = get_registry().get("voyageai").create(name="voyage-3")
    supported = (
        set(func.text_embedding_models)
        | set(func.multimodal_embedding_models)
        | set(func.contextual_embedding_models)
    )
    assert supported == set(VOYAGE_TOTAL_TOKEN_LIMITS)


def test_input_errors_are_not_retried():
    from lancedb.embeddings.voyageai import VoyageAIEmbeddingFunction

    client = MagicMock()
    client.tokenize.return_value = [[1]]
    client.contextualized_embed.return_value = _contextualized_result([2])
    VoyageAIEmbeddingFunction.client = client
    func = get_registry().get("voyageai").create(name="voyage-context-4")

    with pytest.raises(ValueError, match="split into 2 chunks"):
        func.compute_source_embeddings_with_retry(["long document"])
    client.contextualized_embed.assert_called_once()
