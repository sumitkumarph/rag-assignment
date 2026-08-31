from typing import List, Dict
from src.conversation.memory import ConversationMemory


def build_rag_prompt(
    question: str,
    retrieved_chunks: List[Dict],
    memory: ConversationMemory
) -> str:

    context_parts = []

    conversation_history = (
        memory.get_formatted_history()
    )

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {index}
Document: {chunk['source']}
Chunk ID: {chunk['chunk_id']}

{chunk['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a document-based question answering assistant.

STRICT RULES:

1. Answer the question ONLY using the provided document context.
2. Do NOT use outside knowledge.
3. Do NOT guess, assume, or extrapolate.
4. If the answer is not supported by the context, say:
"I don't have enough information in the provided documents."
5. If the question has multiple parts, answer each part
   that is supported by the context.
6. Keep the answer concise but complete.
7. Every factual statement must be supported by the context.

Previous conversation:
{conversation_history}

Current question:
{question}

Document context:
==================================================
{context}
==================================================

Now provide the answer.
"""

    return prompt