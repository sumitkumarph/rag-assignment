from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np


class FAISSVectorStore:

    def __init__(self, dimension: int):

        self.dimension = dimension

        # Inner product works as cosine similarity
        # because our embeddings are normalized.
        self.index = faiss.IndexFlatIP(dimension)

        # Maps FAISS position -> original chunk
        self.documents: List[Dict] = []

    def add(
        self,
        embeddings: np.ndarray,
        documents: List[Dict]
    ):

        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings must match "
                "number of documents."
            )

        self.index.add(
            embeddings.astype("float32")
        )

        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ):

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            document = self.documents[index].copy()

            document["score"] = float(score)

            results.append(document)

        return results

    def save(
        self,
        index_path: Path,
        metadata_path: Path
    ):

        faiss.write_index(
            self.index,
            str(index_path)
        )

        import pickle

        with open(
            metadata_path,
            "wb"
        ) as file:

            pickle.dump(
                self.documents,
                file
            )

    @classmethod
    def load(
        cls,
        index_path: Path,
        metadata_path: Path
    ):

        index = faiss.read_index(
            str(index_path)
        )

        import pickle

        with open(
            metadata_path,
            "rb"
        ) as file:

            documents = pickle.load(file)

        store = cls(index.d)

        store.index = index
        store.documents = documents

        return store