from src.retrieval.retriever import Retriever
from src.generation.prompt_builder import build_rag_prompt
from src.generation.llama_service import LlamaService
from src.conversation.memory import ConversationMemory


class RAGService:

    def __init__(
        self,
        retriever: Retriever,
        llama_service: LlamaService,
        max_conversations: int = 4
    ):
        self.retriever = retriever
        self.llama_service = llama_service

        self.memory = ConversationMemory(
            max_conversations=max_conversations
        )

    def ask(self, query: str):

        # -----------------------------
        # 1. Retrieve relevant chunks
        # -----------------------------

        results = self.retriever.retrieve(
            query=query,
            top_k=5,
            similarity_threshold=0.20
        )

        # -----------------------------
        # 2. No relevant information
        # -----------------------------

        if not results:

            answer = (
                "I don't have enough information "
                "in the provided documents to answer "
                "this question."
            )

            return answer, []

        # -----------------------------
        # 3. Build RAG prompt
        # -----------------------------

        prompt = build_rag_prompt(
            query,
            results,
            self.memory
        )

        # -----------------------------
        # 4. Generate answer
        # -----------------------------

        answer = self.llama_service.generate(
            prompt
        )

        # -----------------------------
        # 5. Save conversation
        # -----------------------------

        self.memory.add(
            query,
            answer
        )

        # -----------------------------
        # 6. Return answer + sources
        # -----------------------------

        return answer, results

    def clear_memory(self):

        self.memory = ConversationMemory(
            max_conversations=4
        )