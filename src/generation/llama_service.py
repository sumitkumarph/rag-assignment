import requests


class LlamaService:

    def __init__(self, base_url, model_name):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def generate(self, prompt: str) -> str:

        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful RAG assistant. "
                        "Answer only using the provided context."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 512
        }

        print(f"Calling Llama server: {url}")

        response = requests.post(
            url,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        return result["choices"][0]["message"]["content"]