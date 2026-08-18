from config import (
    create_directories,
    PDF_DIRECTORY,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH
)

from config import create_directories, PDF_DIRECTORY
from src.ingestion.pdf_loader import extract_all_pdfs
from src.chunking.document_chunker import process_document
from src.embeddings.embedding_service import EmbeddingService
from src.vector_store.faiss_store import (FAISSVectorStore)


def main():
    print("===================================")
    print("RAG Assignment")
    print("Phase 1 - Project Setup")
    print("===================================")

    # Create required directories
    create_directories()

    print(f"PDF directory: {PDF_DIRECTORY}")
    print("Project setup completed successfully.")

    print("===================================")
    print("RAG Assignment")
    print("Phase 2 - PDF Ingestion")
    print("===================================")

    create_directories()

    documents = extract_all_pdfs(PDF_DIRECTORY)

    print(f"\nTotal PDFs processed: {len(documents)}")

    for document in documents:
        print(
            f"{document['filename']} "
            f"-> {len(document['text'])} characters"
        )

    print("===================================")
    print("RAG Assignment")
    print("Phase 3 - Cleaning & Chunking")
    print("===================================")

    create_directories()

    documents = extract_all_pdfs(PDF_DIRECTORY)

    total_chunks = 0

    for document in documents:

        chunks = process_document(
            document["filename"],
            document["text"]
        )

        print(
            f"\n{document['filename']}"
        )

        print(
            f"Characters: {len(document['text'])}"
        )

        print(
            f"Chunks: {len(chunks)}"
        )

        total_chunks += len(chunks)

        # Show first chunk for verification
        if chunks:

            print("\nFirst chunk:")
            print("--------------------")

            print(chunks[0]["text"][:500])

            print("--------------------")

    print(
        f"\nTotal chunks created: {total_chunks}"
    )

    print("===================================")
    print("RAG Assignment")
    print("Phase 4 - Embeddings")
    print("===================================")

    create_directories()

    # -----------------------------
    # Phase 2: PDF ingestion
    # -----------------------------

    documents = extract_all_pdfs(PDF_DIRECTORY)

    all_chunks = []

    # -----------------------------
    # Phase 3: Cleaning + Chunking
    # -----------------------------

    for document in documents:

        chunks = process_document(
            document["filename"],
            document["text"]
        )

        all_chunks.extend(chunks)

    print(
        f"\nTotal chunks: {len(all_chunks)}"
    )

    # -----------------------------
    # Phase 4: Embeddings
    # -----------------------------

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
    print("Phase 5 - FAISS Vector Store")
    print("===================================")

    create_directories()

    # -------------------------
    # Phase 2
    # PDF ingestion
    # -------------------------

    documents = extract_all_pdfs(
        PDF_DIRECTORY
    )

    # -------------------------
    # Phase 3
    # Chunking
    # -------------------------

    all_chunks = []

    for document in documents:

        chunks = process_document(
            document["filename"],
            document["text"]
        )

        all_chunks.extend(chunks)

    print(
        f"\nTotal chunks: {len(all_chunks)}"
    )

    # -------------------------
    # Phase 4
    # Embeddings
    # -------------------------

    embedding_service = (
        EmbeddingService(
            EMBEDDING_MODEL
        )
    )

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = (
        embedding_service
        .generate_embeddings(texts)
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    # -------------------------
    # Phase 5
    # FAISS
    # -------------------------

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








    query = (
        "What does this document say "
        "about artificial intelligence?"
    )

    query_embedding = (
        embedding_service
        .generate_embedding(query)
    )

    results = vector_store.search(
        query_embedding,
        top_k=3
    )

    print("\n===================================")
    print("FAISS Search Test")
    print("===================================")

    for result in results:

        print(
            f"\nScore: {result['score']:.4f}"
        )

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Chunk: {result['chunk_id']}"
        )

        print(
            f"Text: {result['text'][:300]}"
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


if __name__ == "__main__":
    main()