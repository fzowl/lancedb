# Voyage AI by MongoDB Embeddings

Voyage AI by MongoDB provides cutting-edge embedding and rerankers.


Using the Voyage AI by MongoDB API requires the `voyageai` package, version 0.4.0 or later, which can be installed using `pip install "voyageai>=0.4.0"`. Voyage AI by MongoDB embeddings are used to generate embeddings for text data. The embeddings can be used for various tasks like semantic search, clustering, and classification.
You also need to set the `VOYAGE_API_KEY` environment variable to use the Voyage AI by MongoDB API.

Supported models are:

**Voyage-4 Series (Latest)**

- voyage-4 (1024 dims, general-purpose and multilingual retrieval, 320K batch tokens)
- voyage-4-lite (1024 dims, optimized for latency and cost, 1M batch tokens)
- voyage-4-large (1024 dims, best retrieval quality, 120K batch tokens)
- voyage-code-4 (1024 dims, code retrieval, no documented batch token limit)

**Contextualized Models**

- voyage-context-4 (1024 dims, 120K batch tokens)
- voyage-context-3 (1024 dims, 120K batch tokens)

**Voyage-3 Series**

- voyage-3-large
- voyage-3.5
- voyage-3.5-lite
- voyage-code-3
- voyage-3
- voyage-3-lite

**Domain-Specific Models**

- voyage-finance-2
- voyage-multilingual-2
- voyage-law-2
- voyage-code-2


Supported parameters (to be passed in `create` method) are:

| Parameter | Type | Default Value | Description |
|---|---|--------|---------|
| `name` | `str` | `None` | The model ID of the model to use. Supported base models for Text Embeddings: voyage-4, voyage-4-lite, voyage-4-large, voyage-code-4, voyage-context-4, voyage-context-3, voyage-3-large, voyage-3.5, voyage-3.5-lite, voyage-code-3, voyage-3, voyage-3-lite, voyage-finance-2, voyage-multilingual-2, voyage-law-2, voyage-code-2 |
| `input_type` | `str` | `None` | Type of the input text. Default to None. Other options: query, document. |
| `truncation` | `bool` | `True` | Whether to truncate the input texts to fit within the context length. |
| `output_dimension` | `int` | `None` | Output embedding dimension, sent to the API only when set. Supported by the voyage-4 family, voyage-code-4, voyage-context-4, voyage-context-3, voyage-3-large, voyage-3.5, voyage-3.5-lite and voyage-code-3; any other model raises a `ValueError`. Valid options: 256, 512, 1024 (default), 2048. |


Usage Example:
    
```python
    import lancedb
    from lancedb.pydantic import LanceModel, Vector
    from lancedb.embeddings import EmbeddingFunctionRegistry

    voyageai = EmbeddingFunctionRegistry
        .get_instance()
        .get("voyageai")
        .create(name="voyage-3")

    class TextModel(LanceModel):
        text: str = voyageai.SourceField()
        vector: Vector(voyageai.ndims()) =  voyageai.VectorField()

    data = [ { "text": "hello world" },
            { "text": "goodbye world" }]

    db = lancedb.connect("~/.lancedb")
    tbl = db.create_table("test", schema=TextModel, mode="overwrite")

    tbl.add(data)
```