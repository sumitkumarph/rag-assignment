from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    LLAMA_MODEL,
    LLAMA_BASE_URL
)

from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.retriever import Retriever
from src.generation.prompt_builder import (build_rag_prompt)
from src.generation.llama_service import (LlamaService)
from src.vector_store.faiss_store import FAISSVectorStore
from conversation.memory import ConversationMemory


def main():
    print("===================================")
    print("RAG Assignment")
    print("Step 4 - Load Embedding model & Load existing FAISS index")
    print("===================================")

    # -----------------------------
    # 1. Load embedding model
    # -----------------------------

    embedding_service = EmbeddingService(
        EMBEDDING_MODEL
    )

    # -----------------------------
    # 2. Load existing FAISS index
    # -----------------------------

    if not FAISS_INDEX_PATH.exists():
        print(
            "\nFAISS index not found."
        )

        print(
            "Run: python index_documents.py"
        )

        return

    vector_store = (
        FAISSVectorStore.load(
            FAISS_INDEX_PATH,
            FAISS_METADATA_PATH
        )
    )

    print(
        f"\nLoaded FAISS vectors: "
        f"{vector_store.index.ntotal}"
    )

    print("===================================")
    print("RAG Assignment")
    print("Step 6 - Retriever")
    print("===================================")

    retriever = Retriever(
        embedding_service,
        vector_store
    )

    # -----------------------------
    # 4. Create Llama service
    # -----------------------------

    llama_service = LlamaService(
        LLAMA_BASE_URL,
        LLAMA_MODEL
    )

    # -----------------------------
    # 5. Ask questions
    # -----------------------------

    #Creating the conersation memory
    memory = ConversationMemory(max_conversations=4)

    while True:

        query = input(
            "\nEnter your question "
            "(type 'exit' to quit): "
        )

        if query.lower() == "exit":
            break

        if not query.strip():
            continue

        # -------------------------
        # Retrieval
        # -------------------------

        results = retriever.retrieve(
            query=query,
            top_k=5,
            similarity_threshold=0.3
        )

        if not results:

            print(
                "No relevant information found."
            )

            continue

        # -------------------------
        # Build RAG prompt
        # -------------------------

        prompt = build_rag_prompt(
            query,
            results,
            memory
        )

        # -------------------------
        # Generate answer
        # -------------------------

        answer = llama_service.generate(
            prompt
        )

        memory.add(query, answer)

        print(
            "\n==================================="
        )

        print("Generated Answer")

        print(
            "==================================="
        )

        print(answer)

        # -------------------------
        # Sources
        # -------------------------

        print(
            "\n==================================="
        )

        print("Sources")

        print(
            "==================================="
        )

        for result in results:

            print(
                f"- {result['source']} "
                f"(Chunk {result['chunk_id']}, "
                f"Score: {result['score']:.4f})"
            )

if __name__ == "__main__":
    main()