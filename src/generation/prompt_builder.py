from typing import List, Dict


def build_rag_prompt(
    question: str,
    retrieved_chunks: List[Dict]
) -> str:

    context_parts = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_parts.append(
            f"""
Source {index}:
Document: {chunk['source']}
Chunk ID: {chunk['chunk_id']}

{chunk['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the
provided context.

If the answer cannot be found in the context,
say:

"I don't have enough information in the provided documents."

Do not invent or assume information.

Always base your answer on the retrieved context.

Context:
-------------------------
{context}
-------------------------

User Question:
{question}

Answer:
"""

    return prompt