# RAG Assignment

This project implements a Retrieval-Augmented Generation (RAG) system using Python.

## Architecture

PDF Documents
    ↓
Text Extraction
    ↓
Text Cleaning
    ↓
Chunking
    ↓
Text Embeddings
    ↓
FAISS Vector Store
    ↓
Semantic Retrieval
    ↓
Llama LLM
    ↓
Generated Answer

## Technology

- Python
- PDF processing
- Sentence Transformers
- FAISS
- Llama
- RAG
- Streamlit (demonstration UI)

## Project Structure

- `data/pdfs` - Input PDF documents
- `data/processed` - Processed data
- `src/ingestion` - PDF ingestion
- `src/chunking` - Text chunking
- `src/embeddings` - Embedding generation
- `src/vectorstore` - FAISS operations
- `src/retrieval` - Similarity search
- `src/generation` - LLM response generation
- `src/evaluation` - RAG evaluation
- `tests` - Unit tests

## Current Phase

Phase 1 - Project Setup