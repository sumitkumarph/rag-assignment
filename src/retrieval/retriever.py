import re
import math
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

    # ---------------------------------------------------------
    # Tokenize
    # ---------------------------------------------------------

    def _tokenize(self, text: str):

        return set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                text.lower()
            )
        )

    # ---------------------------------------------------------
    # Keyword score
    # ---------------------------------------------------------

    def _keyword_score(
        self,
        query: str,
        document: str
    ):

        query_words = self._tokenize(query)
        document_words = self._tokenize(document)

        if not query_words or not document_words:
            return 0.0

        common_words = (
            query_words.intersection(document_words)
        )

        return len(common_words) / len(query_words)

    # ---------------------------------------------------------
    # Retrieve
    # ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.20
    ) -> List[Dict]:

        # -----------------------------------------------------
        # 1. Semantic retrieval
        # -----------------------------------------------------

        query_embedding = (
            self.embedding_service
            .generate_embedding(query)
        )

        semantic_results = self.vector_store.search(
            query_embedding,
            top_k=12
        )

        # -----------------------------------------------------
        # 2. Keyword retrieval over all documents
        # -----------------------------------------------------

        keyword_candidates = []

        for document in self.vector_store.documents:

            score = self._keyword_score(
                query,
                document["text"]
            )

            if score > 0:

                item = document.copy()

                item["keyword_score"] = score

                keyword_candidates.append(item)

        keyword_candidates.sort(
            key=lambda x: x["keyword_score"],
            reverse=True
        )

        keyword_candidates = keyword_candidates[:12]

        # -----------------------------------------------------
        # 3. Combine candidates
        # -----------------------------------------------------

        candidates = {}

        for item in semantic_results:

            key = (
                item["source"],
                item["chunk_id"]
            )

            candidates[key] = item.copy()

            candidates[key]["keyword_score"] = (
                self._keyword_score(
                    query,
                    item["text"]
                )
            )

        for item in keyword_candidates:

            key = (
                item["source"],
                item["chunk_id"]
            )

            if key not in candidates:

                candidates[key] = item.copy()

                candidates[key]["score"] = 0.0

        # -----------------------------------------------------
        # 4. Combined reranking
        # -----------------------------------------------------

        reranked = []

        for item in candidates.values():

            semantic_score = max(
                0.0,
                min(1.0, item.get("score", 0.0))
            )

            keyword_score = item.get(
                "keyword_score",
                0.0
            )

            # Semantic similarity is more important,
            # but keyword matching helps exact concepts.
            combined_score = (
                0.65 * semantic_score +
                0.35 * keyword_score
            )

            item["combined_score"] = combined_score

            reranked.append(item)

        reranked.sort(
            key=lambda x: x["combined_score"],
            reverse=True
        )

        # -----------------------------------------------------
        # 5. Final filtering
        # -----------------------------------------------------

        final_results = []

        for item in reranked:

            if item["combined_score"] >= similarity_threshold:

                item["score"] = item["combined_score"]

                final_results.append(item)

            if len(final_results) >= top_k:
                break

        return final_results