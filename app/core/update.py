from uuid import uuid4

from langchain.text_splitter import MarkdownTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
)

from core.embeddings import bgem3
from core.settings import settings

qdrant = QdrantClient(str(settings.qdrant_host))
md_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)


def delete_old_document(document_path: str):
    qdrant.delete(
        collection_name=settings.collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="metadata.source", match=MatchValue(value=document_path)
                ),
            ]
        ),
    )


def upload_new_document(document_path: str, document_text: str):
    vectors = bgem3.encode(document_text, return_dense=True, return_sparse=True)
    dense_vector = vectors["dense_vecs"]
    sparse_vector = SparseVector(
        indices=vectors["lexical_weights"].keys(),
        values=vectors["lexical_weights"].values(),
    )

    qdrant.upload_points(
        settings.collection_name,
        [
            PointStruct(
                id=str(uuid4()),
                vector={"text-dense": dense_vector, "text-sparse": sparse_vector},
                payload={
                    "metadata": {"source": document_path},
                    "page_content": document_text,
                },
            )
        ],
    )


def update_collection(document_path: str, document_text: str):
    delete_old_document(document_path)
    splits = md_splitter.split_text(document_text)
    for split in splits:
        upload_new_document(document_path, split)
