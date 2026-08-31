from collections import deque


class ConversationMemory:
    """
    Maintains the last N question-answer conversations.
    """

    def __init__(self, max_conversations=4):
        self.max_conversations = max_conversations
        self.history = deque(maxlen=max_conversations)

    def add(self, question, answer):
        self.history.append({
            "question": question,
            "answer": answer
        })

    def get_history(self):
        return list(self.history)

    def get_formatted_history(self):
        if not self.history:
            return "No previous conversation."

        conversation = []

        for item in self.history:
            conversation.append(
                f"User: {item['question']}\n"
                f"Assistant: {item['answer']}"
            )

        return "\n\n".join(conversation)

    def clear(self):
        self.history.clear()