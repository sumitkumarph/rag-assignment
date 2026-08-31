import streamlit as st

from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    LLAMA_MODEL,
    LLAMA_BASE_URL
)

from src.embeddings.embedding_service import EmbeddingService
from src.generation.llama_service import LlamaService
from src.retrieval.retriever import Retriever
from src.vector_store.faiss_store import FAISSVectorStore

from src.rag.rag_service import RAGService


# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


st.title("📚 RAG Assistant")

st.write(
    "Ask questions about the uploaded documents."
)


# --------------------------------------------------
# Initialize RAG Service
# --------------------------------------------------

if "rag_service" not in st.session_state:

    # -----------------------------
    # 1. Load embedding model
    # -----------------------------

    embedding_service = EmbeddingService(
        EMBEDDING_MODEL
    )

    # -----------------------------
    # 2. Check FAISS index
    # -----------------------------

    if not FAISS_INDEX_PATH.exists():

        st.error(
            "FAISS index not found. "
            "Please run index_documents.py first."
        )

        st.stop()

    # -----------------------------
    # 3. Load FAISS index
    # -----------------------------

    vector_store = FAISSVectorStore.load(
        FAISS_INDEX_PATH,
        FAISS_METADATA_PATH
    )

    # -----------------------------
    # 4. Create Retriever
    # -----------------------------

    retriever = Retriever(
        embedding_service,
        vector_store
    )

    # -----------------------------
    # 5. Create Llama Service
    # -----------------------------

    llama_service = LlamaService(
        LLAMA_BASE_URL,
        LLAMA_MODEL
    )

    # -----------------------------
    # 6. Create RAG Service
    # -----------------------------

    st.session_state.rag_service = RAGService(
        retriever=retriever,
        llama_service=llama_service,
        max_conversations=4
    )


# --------------------------------------------------
# Get RAG Service
# --------------------------------------------------

rag_service = st.session_state.rag_service


# --------------------------------------------------
# Initialize UI messages
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------------------------
# Clear Conversation
# --------------------------------------------------

if st.button("🗑️ Clear Conversation"):

    rag_service.clear_memory()

    st.session_state.messages = []

    st.rerun()


# --------------------------------------------------
# Display conversation history
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# --------------------------------------------------
# Chat input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about the documents..."
)


# --------------------------------------------------
# Process question
# --------------------------------------------------

if question:

    # -----------------------------
    # Display user message
    # -----------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # -----------------------------
    # Generate answer
    # -----------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                answer, results = (
                    rag_service.ask(question)
                )

            except Exception as e:

                st.error(
                    f"Error while generating answer: {e}"
                )

                st.stop()

        st.markdown(answer)

    # -----------------------------
    # Store assistant message
    # -----------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # -----------------------------
    # Display sources
    # -----------------------------

    if results:

        with st.expander("📚 Sources"):

            for result in results:

                source = result.get(
                    "source",
                    "Unknown"
                )

                chunk_id = result.get(
                    "chunk_id",
                    "N/A"
                )

                score = result.get(
                    "score",
                    0
                )

                st.write(
                    f"- **{source}** "
                    f"(Chunk {chunk_id}, "
                    f"Score: {score:.4f})"
                )