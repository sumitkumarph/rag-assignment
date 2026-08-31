import json
import sys
from pathlib import Path

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    LLAMA_BASE_URL,
    LLAMA_MODEL
)

from src.embeddings.embedding_service import EmbeddingService
from src.vector_store.faiss_store import FAISSVectorStore
from src.retrieval.retriever import Retriever
from src.generation.llama_service import LlamaService
from src.rag.rag_service import RAGService


# ---------------------------------------------------------
# Files
# ---------------------------------------------------------

REFERENCES_FILE = PROJECT_ROOT / "src" / "evaluation" / "references.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "ragas_dataset.json"


# ---------------------------------------------------------
# Load evaluation questions
# ---------------------------------------------------------

def load_questions():

    with open(REFERENCES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------
# Create RAG service
# ---------------------------------------------------------

def create_rag_service():

    print()
    print("Initializing RAG components...")

    # -----------------------------------------------------
    # Embedding service
    # -----------------------------------------------------

    embedding_service = EmbeddingService(
        EMBEDDING_MODEL
    )

    print("Embedding service initialized.")

    # -----------------------------------------------------
    # Load existing FAISS index
    # -----------------------------------------------------

    vector_store = FAISSVectorStore.load(
        FAISS_INDEX_PATH,
        FAISS_METADATA_PATH
    )

    print(
        f"FAISS index loaded successfully. "
        f"Documents: {len(vector_store.documents)}"
    )

    # -----------------------------------------------------
    # Retriever
    # -----------------------------------------------------

    retriever = Retriever(
        embedding_service,
        vector_store
    )

    print("Retriever initialized.")

    # -----------------------------------------------------
    # Llama service
    # -----------------------------------------------------

    llama_service = LlamaService(
        LLAMA_BASE_URL,
        LLAMA_MODEL
    )

    print("Llama service initialized.")

    # -----------------------------------------------------
    # RAG service
    # -----------------------------------------------------

    rag_service = RAGService(
        retriever,
        llama_service
    )

    print("RAG service initialized.")

    return rag_service


# ---------------------------------------------------------
# Convert retrieved results into RAGAS contexts
# ---------------------------------------------------------

def extract_contexts(results):

    contexts = []

    for result in results:

        # Your FAISS results are dictionaries.
        # Try the common text fields.

        if isinstance(result, dict):

            text = (
                result.get("text")
                or result.get("content")
                or result.get("chunk")
                or result.get("document")
            )

            if text:
                contexts.append(str(text))

        elif isinstance(result, str):

            contexts.append(result)

    return contexts


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("Building RAGAS Evaluation Dataset")
    print("=" * 60)

    # Load questions
    questions = load_questions()

    print(f"Total questions: {len(questions)}")

    # Create RAG
    rag = create_rag_service()

    dataset = []

    # -----------------------------------------------------
    # Process every question
    # -----------------------------------------------------

    for index, item in enumerate(questions, start=1):

        question = item["question"]

        # Clear conversation memory for independent evaluation
        rag.clear_memory()

        print()
        print("-" * 60)
        print(f"[{index}/{len(questions)}]")
        print(f"Question: {question}")

        try:
            answer, results = rag.ask(question)

            # Convert FAISS results to RAGAS contexts
            contexts = extract_contexts(results)

            print(f"Retrieved chunks: {len(contexts)}")

            print("Answer:")
            print(answer)

            # -------------------------------------------------
            # Reference answer
            # -------------------------------------------------

            reference = (
                item.get("reference")
                or item.get("answer")
                or item.get("expected_answer")
                or ""
            )

            # -------------------------------------------------
            # Dataset record
            # -------------------------------------------------

            dataset_item = {
                "id": item.get(
                    "id",
                    f"Q{index:02d}"
                ),

                "question": question,

                "answer": answer,

                "contexts": contexts,

                "reference": reference,

                "document": item.get(
                    "document",
                    ""
                ),

                "topic": item.get(
                    "topic",
                    ""
                ),

                "type": item.get(
                    "type",
                    ""
                )
            }

            dataset.append(dataset_item)

            print("✓ Question processed successfully.")

        except Exception as ex:

            print()
            print(f"✗ ERROR processing question {index}")
            print(str(ex))

            dataset.append({
                "id": item.get(
                    "id",
                    f"Q{index:02d}"
                ),

                "question": question,

                "answer": "",

                "contexts": [],

                "reference": (
                    item.get("reference")
                    or item.get("answer")
                    or ""
                ),

                "document": item.get(
                    "document",
                    ""
                ),

                "topic": item.get(
                    "topic",
                    ""
                ),

                "type": item.get(
                    "type",
                    ""
                ),

                "error": str(ex)
            })

    # ---------------------------------------------------------
    # Save dataset
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            indent=4,
            ensure_ascii=False
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    successful = sum(
        1
        for item in dataset
        if item.get("answer")
    )

    failed = len(dataset) - successful

    print()
    print("=" * 60)
    print("RAGAS Dataset Creation Completed")
    print("=" * 60)

    print(f"Total questions : {len(dataset)}")
    print(f"Successful      : {successful}")
    print(f"Failed          : {failed}")

    print()
    print(f"Dataset saved to:")
    print(OUTPUT_FILE)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()