from config import (
    create_directories,
    PDF_DIRECTORY,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    LLAMA_MODEL,
    LLAMA_BASE_URL
)

from config import create_directories, PDF_DIRECTORY
from src.ingestion.pdf_loader import extract_all_pdfs
from src.chunking.document_chunker import process_document
from src.embeddings.embedding_service import EmbeddingService
from src.vector_store.faiss_store import (FAISSVectorStore)
from src.retrieval.retriever import Retriever
from src.generation.prompt_builder import (build_rag_prompt)
from src.generation.llama_service import (LlamaService)


def main():
    print("===================================")
    print("RAG Assignment")
    print("Step 1 - Project Setup")
    print("===================================")

    # Create required directories
    create_directories()

    print(f"PDF directory: {PDF_DIRECTORY}")
    print("Project setup completed successfully.")

    print("===================================")
    print("RAG Assignment")
    print("Step 2 - PDF Ingestion")
    print("===================================")

    print("Processing all documents & extracting texts")

    documents = extract_all_pdfs(PDF_DIRECTORY)
    
    all_chunks = []

    print("===================================")
    print("RAG Assignment")
    print("Step 3 - Cleaning & Chunking")
    print("===================================")

    for document in documents:
    
            chunks = process_document(
                document["filename"],
                document["text"]
            )
    
            all_chunks.extend(chunks)
    
    print(
        f"\nTotal chunks: {len(all_chunks)}"
    )

    print("===================================")
    print("RAG Assignment")
    print("Step 4 - Embeddings")
    print("===================================")

    embedding_service = EmbeddingService(
        EMBEDDING_MODEL
    )

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = embedding_service.generate_embeddings(
        texts
    )

    print("\n===================================")
    print("Embedding Results")
    print("===================================")

    print(
        f"Number of embeddings: {len(embeddings)}"
    )

    print(
        f"Embedding dimensions: {embeddings.shape[1]}"
    )

    # Display first embedding
    print("\nFirst embedding:")

    print(
        embeddings[0][:10]
    )

    print("===================================")
    print("RAG Assignment")
    print("Step 5 - FAISS Vector Store")
    print("===================================")

    dimension = embeddings.shape[1]

    vector_store = FAISSVectorStore(
        dimension
    )

    vector_store.add(
        embeddings,
        all_chunks
    )

    print(
        f"FAISS vectors: "
        f"{vector_store.index.ntotal}"
    )


    # -------------------------
    # Save FAISS
    # -------------------------

    vector_store.save(
        FAISS_INDEX_PATH,
        FAISS_METADATA_PATH
    )

    print("\nFAISS index saved:")
    print(FAISS_INDEX_PATH)

    print("\nMetadata saved:")
    print(FAISS_METADATA_PATH)

    print("===================================")
    print("RAG Assignment")
    print("Step 6 - Retriever")
    print("===================================")

    retriever = Retriever(
        embedding_service,
        vector_store
    )

    query = input(
        "\nEnter your question: "
    )

    results = retriever.retrieve(
        query=query,
        top_k=5,
        similarity_threshold=0.3
    )

    print("\n===================================")
    print("Retrieved Results")
    print("===================================")

    if not results:

        print(
            "No relevant information found."
        )

    print("===================================")
    print("RAG Assignment")
    print("Step 7 - Generation")
    print("===================================")

    prompt = build_rag_prompt(
        query,
        results
    )

    llama_service = LlamaService(
         LLAMA_BASE_URL,
         LLAMA_MODEL
    )

    answer = llama_service.generate(
        prompt
    )

    print("\n===================================")
    print("Generated Answer")
    print("===================================")

    print(answer)

    print("\n===================================")
    print("Sources")
    print("===================================")

    for result in results:

        print(
            f"- {result['source']} "
            f"(Chunk {result['chunk_id']}, "
            f"Score: {result['score']:.4f})"
        )

if __name__ == "__main__":
    main()