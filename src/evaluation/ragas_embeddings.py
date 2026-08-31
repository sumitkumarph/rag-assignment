from typing import List

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class RagasEmbeddings:

    def __init__(
        self
    ):
        print(
            f"Loading RAGAS embedding model: {EMBEDDING_MODEL}"
        )

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        print(
            "RAGAS embedding model loaded."
        )

    def embed_query(
        self,
        text: str
    ) -> List[float]:

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def embed_documents(
        self,
        texts: List[str]
    ) -> List[List[float]]:

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings.tolist()