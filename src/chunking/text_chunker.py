from typing import List, Dict


def create_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[str]:

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk.strip())

        start = end - chunk_overlap

    return chunks