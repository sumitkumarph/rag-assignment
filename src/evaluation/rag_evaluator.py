from typing import Dict, List, Any

from src.rag.rag_service import RAGService

class RAGEvaluator:
    """
    Adapter between the existing RAGService and RAGAS.

    It executes a question through the existing RAG pipeline
    and converts the result into the format required for
    RAGAS evaluation.
    """

    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service

    def evaluate_question(
        self,
        question: str
    ) -> Dict[str, Any]:
        """
        Ask the existing RAG bot and return the information
        required by the RAGAS evaluation layer.
        """

        # Existing RAG pipeline
        answer, results = self.rag_service.ask(question)

        # Extract text from retrieved FAISS documents
        retrieved_contexts: List[str] = []

        for result in results:
            text = result.get("text")

            if text:
                retrieved_contexts.append(text)

        return {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": retrieved_contexts
        }

    def evaluate_questions(
        self,
        questions: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple questions through the RAG pipeline.
        """

        evaluation_data = []

        for question in questions:

            result = self.evaluate_question(question)

            evaluation_data.append(result)

        return evaluation_data