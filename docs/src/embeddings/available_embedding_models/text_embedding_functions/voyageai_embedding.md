# VoyageAI by MongoDB Embeddings

VoyageAI by MongoDB provides cutting-edge embedding and rerankers.


Using voyageai API requires voyageai package, which can be installed using `pip install voyageai`. VoyageAI by MongoDB embeddings are used to generate embeddings for text data. The embeddings can be used for various tasks like semantic search, clustering, and classification.
You also need to set the `VOYAGE_API_KEY` environment variable to use the VoyageAI by MongoDB API.

Supported models (see [docs.voyageai.com/docs/embeddings](https://docs.voyageai.com/docs/embeddings) for the current lineup) are:

**Voyage-4 Series (Latest)**

- voyage-4-large (1024 dims, best retrieval quality)
- voyage-4 (1024 dims, general-purpose and multilingual retrieval)
- voyage-4-lite (1024 dims, optimized for latency and cost)
- voyage-4-nano (1024 dims, open-weight)
- voyage-code-4 (1024 dims, code retrieval and coding agents)

**Voyage-3 Series**

- voyage-3-large
- voyage-3.5
- voyage-3.5-lite
- voyage-3
- voyage-3-lite
- voyage-code-3

**Contextualized Chunk Embeddings**

- voyage-context-4 (latest)
- voyage-context-3

**Domain-Specific Models**

- voyage-finance-2
- voyage-multilingual-2
- voyage-law-2
- voyage-code-2


Supported parameters (to be passed in `create` method) are:

| Parameter | Type | Default Value | Description |
|---|---|--------|---------|
| `name` | `str` | `None` | The model ID of the model to use. Supported base models for Text Embeddings: voyage-4-large, voyage-4, voyage-4-lite, voyage-4-nano, voyage-code-4, voyage-3-large, voyage-3.5, voyage-3.5-lite, voyage-3, voyage-3-lite, voyage-code-3, voyage-finance-2, voyage-multilingual-2, voyage-law-2, voyage-code-2 |
| `input_type` | `str` | `None` | Type of the input text. Default to None. Other options: query, document. |
| `truncation` | `bool` | `True` | Whether to truncate the input texts to fit within the context length. |


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