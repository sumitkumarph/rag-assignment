import time

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
                        "You are a document question-answering assistant. "
                        "Answer ONLY using the provided context. "
                        "Do not guess or use outside knowledge. "
                        "If the answer is not in the context, say "
                        "'I don't have enough information in the provided documents.'"
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0,
            "max_tokens": 256
        }

        print(f"Calling Llama server: {url}")

        for attempt in range(1, 4):

            try:

                print(
                    f"Llama attempt {attempt}/3..."
                )

                response = requests.post(
                    url,
                    json=payload,
                    timeout=180
                )

                response.raise_for_status()

                result = response.json()

                answer = (
                    result["choices"][0]
                    ["message"]["content"]
                    .strip()
                )

                if answer:
                    return answer

            except requests.exceptions.Timeout:

                print(
                    f"Llama timeout on attempt {attempt}"
                )

                if attempt < 3:
                    time.sleep(2)

            except requests.exceptions.RequestException as e:

                print(
                    f"Llama request error: {e}"
                )

                if attempt < 3:
                    time.sleep(2)

        raise RuntimeError(
            "Llama server failed after 3 attempts."
        )

        # response = requests.post(
        #     url,
        #     json=payload,
        #     timeout=180
        # )

        # response.raise_for_status()

        # result = response.json()

        # return result["choices"][0]["message"]["content"]