from langchain_openai import ChatOpenAI

from config import (
    LLAMA_BASE_URL,
    LLAMA_MODEL
)


class RagasLLM:

    def __init__(self):

        base_url = LLAMA_BASE_URL.rstrip("/")

        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        print(
            f"RAGAS evaluator endpoint: {base_url}"
        )

        print(
            f"RAGAS evaluator model: {LLAMA_MODEL}"
        )

        self.llm = ChatOpenAI(
            model=LLAMA_MODEL,
            base_url=base_url,
            api_key="not-needed",
            temperature=0,
            max_tokens=256,
            timeout=300,
            max_retries=1
        )

    def get_llm(self):

        return self.llm