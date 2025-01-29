from typing import Any, Dict

from FlagEmbedding import BGEM3FlagModel
from langchain_core.embeddings import Embeddings
from langchain_qdrant.sparse_embeddings import SparseVector
from pydantic import BaseModel, Field


class BGEM3Embedding(BaseModel, Embeddings):
    model: Any = None
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the model."""

    def __init__(self, model: BGEM3FlagModel, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, **self.encode_kwargs)

        if self.encode_kwargs.get("return_sparse", False):
            return [
                SparseVector(indices=i.keys(), values=i.values())
                for i in embeddings["lexical_weights"]
            ]

        if self.encode_kwargs.get("return_dense", False):
            return embeddings["dense_vecs"]

    def embed_query(self, text):
        return self.embed_documents([text])[0]
