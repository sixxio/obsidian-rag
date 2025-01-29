import pickle
import time
from pathlib import Path

from core.settings import settings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

data = pickle.load(Path("../data/dump.pickle").open("rb"))

time.sleep(15)

qc = QdrantClient(str(settings.qdrant_host), timeout=60)

if not qc.collection_exists(settings.collection_name):
    qc.create_collection(
        collection_name=settings.collection_name,
        vectors_config={
            "text-dense": VectorParams(size=1024, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "text-sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )

    points = [PointStruct(id=i.id, vector=i.vector, payload=i.payload) for i in data]

    qc.upload_points(settings.collection_name, points)

    points_count = qc.count(settings.collection_name)

    print(f"Loaded {points_count} points into {settings.collection_name}")

else:
    print("Collection already exists.")
