from langchain_qdrant.qdrant import QdrantVectorStore, RetrievalMode

from core.embeddings import bgem3_dense, bgem3_sparse
from core.settings import settings

qvs = QdrantVectorStore.from_existing_collection(
    embedding=bgem3_dense,
    vector_name="text-dense",
    sparse_embedding=bgem3_sparse,  # type: ignore
    sparse_vector_name="text-sparse",
    location=str(settings.qdrant_host),
    collection_name=settings.collection_name,
    retrieval_mode=RetrievalMode.HYBRID,
)
