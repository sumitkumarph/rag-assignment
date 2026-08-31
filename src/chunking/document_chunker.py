from typing import List, Dict

from src.chunking.text_cleaner import clean_text
from src.chunking.text_chunker import create_chunks


def process_document(
    filename: str,
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Dict]:

    cleaned_text = clean_text(text)

    chunks = create_chunks(
        cleaned_text,
        chunk_size,
        chunk_overlap
    )

    documents = []

    for index, chunk in enumerate(chunks):

        documents.append({
            "chunk_id": index,
            "source": filename,
            "text": chunk
        })

    return documents