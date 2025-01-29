from FlagEmbedding import BGEM3FlagModel

from core.bge import BGEM3Embedding

bgem3 = BGEM3FlagModel("BAAI/bge-m3", device="cuda", normalize_embeddings=True)

bgem3_sparse = BGEM3Embedding(
    bgem3, encode_kwargs={"return_sparse": True, "return_dense": False, "batch_size": 1}
)

bgem3_dense = BGEM3Embedding(
    bgem3, encode_kwargs={"return_sparse": False, "return_dense": True, "batch_size": 1}
)
