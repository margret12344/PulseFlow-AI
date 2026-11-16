import httpx

from pulseflow.llm.config import LLMSettings, load_llm_config
from pulseflow.llm.exceptions import (
    LLMConnectionError,
    LLMRequestError,
    LLMResponseError,
)

OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaClient:
    """HTTP client for interacting with Ollama chat models."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self.settings = settings or load_llm_config()
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            timeout=self.settings.timeout,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        num_predict: int = 512,
    ) -> str:
        """Generate a response from the configured Ollama model."""

        if not self.settings.model:
            raise LLMRequestError(
                "No LLM model configured in configs/llm.yaml."
            )

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if num_predict <= 0:
            raise ValueError("num_predict must be greater than zero.")

        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        payload = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": num_predict,
            },
        }

        for attempt in range(self.settings.max_retries + 1):
            try:
                response = self.client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()

                try:
                    data = response.json()
                except ValueError as exc:
                    raise LLMResponseError(
                        "Ollama returned invalid JSON."
                    ) from exc

                if not isinstance(data, dict):
                    raise LLMResponseError(
                        "Ollama returned an invalid response structure."
                    )

                message = data.get("message")

                if not isinstance(message, dict):
                    raise LLMResponseError(
                        "Ollama response does not contain a valid message."
                    )

                content = message.get("content")
                done_reason = data.get("done_reason")

                if not isinstance(content, str) or not content.strip():
                    raise LLMResponseError(
                        "Ollama returned an empty or invalid response."
                    )

                if done_reason == "length":
                    raise LLMResponseError(
                        "Response was truncated because it reached "
                        "the token limit."
                    )

                return content.strip()

            except httpx.ConnectError as exc:
                if attempt == self.settings.max_retries:
                    raise LLMConnectionError(
                        "Could not connect to the Ollama server."
                    ) from exc

            except httpx.TimeoutException as exc:
                if attempt == self.settings.max_retries:
                    raise LLMConnectionError(
                        "The Ollama request timed out."
                    ) from exc

            except httpx.HTTPStatusError as exc:
                raise LLMRequestError(
                    f"Ollama returned HTTP {exc.response.status_code}."
                ) from exc

        raise LLMRequestError("Ollama request failed.")

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self.client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()