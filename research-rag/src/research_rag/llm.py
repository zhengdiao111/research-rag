from __future__ import annotations

from collections.abc import Sequence

import httpx

from .config import (
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_THINK,
    LLM_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
)


# =========================================================
# Exceptions
# =========================================================


class OllamaChatError(RuntimeError):
    """
    Raised when local LLM generation fails.
    """


# =========================================================
# Ollama chat client
# =========================================================


class OllamaChatClient:
    """
    Small client for Ollama's /api/chat endpoint.

    Besides returning the generated text, the client stores
    useful metadata about the most recent generation:

        last_done_reason
        last_eval_count

    These allow the RAG layer to detect truncated answers.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = LLM_MODEL,
        timeout_seconds: float = LLM_TIMEOUT_SECONDS,
        temperature: float = LLM_TEMPERATURE,
        max_output_tokens: int = LLM_MAX_OUTPUT_TOKENS,
        think: bool = LLM_THINK,
    ):

        self.base_url = (
            base_url.rstrip("/")
        )

        self.model = model

        self.timeout_seconds = (
            timeout_seconds
        )

        self.temperature = (
            temperature
        )

        self.max_output_tokens = (
            max_output_tokens
        )

        self.think = think


        # -------------------------------------------------
        # Metadata from most recent generation
        # -------------------------------------------------

        self.last_done_reason: str | None = None

        self.last_eval_count: int | None = None


    # =====================================================
    # Chat
    # =====================================================

    def chat(
        self,
        messages: Sequence[dict],
    ) -> str:
        """
        Generate one non-streaming response.

        After completion, inspect:

            self.last_done_reason
            self.last_eval_count
        """

        messages = list(
            messages
        )


        if not messages:

            raise ValueError(
                "messages cannot be empty."
            )


        url = (
            f"{self.base_url}"
            "/api/chat"
        )


        payload = {

            "model":
                self.model,

            "messages":
                messages,

            "stream":
                False,

            "think":
                self.think,

            "options": {

                "temperature":
                    self.temperature,

                "num_predict":
                    self.max_output_tokens,
            },
        }


        # Reset metadata before each call.

        self.last_done_reason = None

        self.last_eval_count = None


        try:

            response = httpx.post(
                url,
                json=payload,
                timeout=self.timeout_seconds,
            )

            response.raise_for_status()


        except httpx.ConnectError as exc:

            raise OllamaChatError(
                "\nCould not connect to Ollama.\n\n"
                f"Expected Ollama at:\n"
                f"{self.base_url}\n\n"
                "Make sure Ollama is running."
            ) from exc


        except httpx.HTTPStatusError as exc:

            raise OllamaChatError(
                "\nOllama chat request failed.\n\n"
                f"Status: "
                f"{exc.response.status_code}\n\n"
                f"Response:\n"
                f"{exc.response.text}"
            ) from exc


        except httpx.HTTPError as exc:

            raise OllamaChatError(
                f"Ollama HTTP error:\n{exc}"
            ) from exc


        # =================================================
        # Parse JSON
        # =================================================

        try:

            data = (
                response.json()
            )

        except ValueError as exc:

            raise OllamaChatError(
                "Ollama returned invalid JSON."
            ) from exc


        # =================================================
        # Store completion metadata
        # =================================================

        done_reason = (
            data.get(
                "done_reason"
            )
        )


        if (
            done_reason is not None
            and not isinstance(
                done_reason,
                str,
            )
        ):

            done_reason = str(
                done_reason
            )


        self.last_done_reason = (
            done_reason
        )


        eval_count = (
            data.get(
                "eval_count"
            )
        )


        if isinstance(
            eval_count,
            int,
        ):

            self.last_eval_count = (
                eval_count
            )

        else:

            self.last_eval_count = (
                None
            )


        # =================================================
        # Extract response
        # =================================================

        message = data.get(
            "message"
        )


        if not isinstance(
            message,
            dict,
        ):

            raise OllamaChatError(
                "Ollama response did not contain "
                "a valid message."
            )


        content = message.get(
            "content"
        )


        if not isinstance(
            content,
            str,
        ):

            raise OllamaChatError(
                "Ollama response did not contain "
                "message.content."
            )


        content = (
            content.strip()
        )


        if not content:

            raise OllamaChatError(
                "Ollama returned an empty answer."
            )


        return content