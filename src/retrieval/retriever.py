from typing import List, Dict

from src.embeddings.embedding_service import EmbeddingService
from src.vector_store.faiss_store import FAISSVectorStore


class Retriever:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore
    ):

        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[Dict]:

        # Convert user question into vector
        query_embedding = (
            self.embedding_service
            .generate_embedding(query)
        )

        # Search FAISS
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        # Remove low-quality results
        filtered_results = [
            result
            for result in results
            if result["score"] >= similarity_threshold
        ]

        return filtered_results