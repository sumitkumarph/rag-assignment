from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:

    def __init__(self, model_name: str):
        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print("Embedding model loaded successfully.")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single piece of text.
        """

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def generate_embeddings(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """
        Generate embeddings for multiple text chunks.
        """

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return np.asarray(embeddings)