from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from .config import (
    LLM_CONTEXT_WINDOW,
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

    Supports:

    - normal text generation
    - structured JSON outputs
    - per-request temperature overrides
    - per-request output-token overrides
    - per-request context-window overrides

    Metadata from the most recent generation is retained:

        last_done_reason
        last_prompt_eval_count
        last_eval_count

    Ollama terminology:

        prompt_eval_count
            Number of input / prompt tokens processed.

        eval_count
            Number of output tokens generated.

        done_reason
            Usually "stop" for normal completion or
            "length" if generation reached a limit.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = LLM_MODEL,
        timeout_seconds: float = LLM_TIMEOUT_SECONDS,
        temperature: float = LLM_TEMPERATURE,
        max_output_tokens: int = LLM_MAX_OUTPUT_TOKENS,
        context_window: int = LLM_CONTEXT_WINDOW,
        think: bool = LLM_THINK,
    ):

        self.base_url = (
            base_url.rstrip("/")
        )

        self.model = (
            model
        )

        self.timeout_seconds = (
            timeout_seconds
        )

        self.temperature = (
            temperature
        )

        self.max_output_tokens = (
            max_output_tokens
        )

        self.context_window = (
            context_window
        )

        self.think = (
            think
        )


        # =================================================
        # Validate client configuration
        # =================================================

        if (
            self.max_output_tokens
            <= 0
        ):

            raise ValueError(
                "max_output_tokens must be greater "
                "than zero."
            )


        if (
            self.context_window
            <= 0
        ):

            raise ValueError(
                "context_window must be greater "
                "than zero."
            )


        if (
            self.timeout_seconds
            <= 0
        ):

            raise ValueError(
                "timeout_seconds must be greater "
                "than zero."
            )


        # =================================================
        # Metadata from the most recent generation
        # =================================================

        self.last_done_reason: (
            str | None
        ) = None


        self.last_prompt_eval_count: (
            int | None
        ) = None


        self.last_eval_count: (
            int | None
        ) = None


    # =====================================================
    # Chat
    # =====================================================

    def chat(
        self,
        messages: Sequence[dict],
        response_format: (
            str
            | dict[str, Any]
            | None
        ) = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        context_window: int | None = None,
    ) -> str:
        """
        Generate one non-streaming Ollama response.

        Parameters
        ----------
        messages:
            Ollama chat messages.

        response_format:
            Optional Ollama structured-output format.

            May be:

                "json"

            or a JSON schema dictionary.

        temperature:
            Optional per-call temperature override.

        max_output_tokens:
            Optional per-call Ollama num_predict override.

        context_window:
            Optional per-call Ollama num_ctx override.

        After completion, inspect:

            self.last_done_reason

            self.last_prompt_eval_count

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


        # =================================================
        # Resolve per-request settings
        # =================================================

        effective_temperature = (
            self.temperature

            if temperature is None

            else temperature
        )


        effective_max_tokens = (
            self.max_output_tokens

            if max_output_tokens is None

            else max_output_tokens
        )


        effective_context_window = (
            self.context_window

            if context_window is None

            else context_window
        )


        # =================================================
        # Validate per-call overrides
        # =================================================

        if (
            effective_max_tokens
            <= 0
        ):

            raise ValueError(
                "max_output_tokens must be greater "
                "than zero."
            )


        if (
            effective_context_window
            <= 0
        ):

            raise ValueError(
                "context_window must be greater "
                "than zero."
            )


        # =================================================
        # Build Ollama request
        # =================================================

        payload: dict[
            str,
            Any,
        ] = {

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
                    effective_temperature,

                # Maximum number of tokens Ollama may
                # generate for this response.
                "num_predict":
                    effective_max_tokens,

                # Total model context window available
                # for prompt + generated response.
                "num_ctx":
                    effective_context_window,
            },
        }


        # =================================================
        # Ollama structured outputs
        # =================================================

        if (
            response_format
            is not None
        ):

            payload[
                "format"
            ] = (
                response_format
            )


        # =================================================
        # Reset response metadata
        # =================================================

        self.last_done_reason = (
            None
        )

        self.last_prompt_eval_count = (
            None
        )

        self.last_eval_count = (
            None
        )


        # =================================================
        # Send request
        # =================================================

        try:

            response = (
                httpx.post(
                    url,
                    json=payload,
                    timeout=(
                        self.timeout_seconds
                    ),
                )
            )


            response.raise_for_status()


        except httpx.ConnectError as exc:

            raise OllamaChatError(
                "\nCould not connect to Ollama.\n\n"
                "Expected Ollama at:\n"
                f"{self.base_url}\n\n"
                "Make sure Ollama is running."
            ) from exc


        except httpx.TimeoutException as exc:

            raise OllamaChatError(
                "\nOllama chat request timed out.\n\n"
                f"Timeout: "
                f"{self.timeout_seconds} seconds\n\n"
                f"Model: {self.model}"
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
                "Ollama HTTP error:\n"
                f"{exc}"
            ) from exc


        # =================================================
        # Parse response JSON
        # =================================================

        try:

            data = (
                response.json()
            )


        except ValueError as exc:

            raise OllamaChatError(
                "Ollama returned invalid JSON."
            ) from exc


        if not isinstance(
            data,
            dict,
        ):

            raise OllamaChatError(
                "Ollama returned an unexpected "
                "response structure."
            )


        # =================================================
        # Save completion metadata
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

            done_reason = (
                str(
                    done_reason
                )
            )


        self.last_done_reason = (
            done_reason
        )


        # -------------------------------------------------
        # Prompt / input token count
        # -------------------------------------------------

        prompt_eval_count = (
            data.get(
                "prompt_eval_count"
            )
        )


        if isinstance(
            prompt_eval_count,
            int,
        ):

            self.last_prompt_eval_count = (
                prompt_eval_count
            )


        else:

            self.last_prompt_eval_count = (
                None
            )


        # -------------------------------------------------
        # Generated / output token count
        # -------------------------------------------------

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
        # Extract assistant message
        # =================================================

        message = (
            data.get(
                "message"
            )
        )


        if not isinstance(
            message,
            dict,
        ):

            raise OllamaChatError(
                "Ollama response did not contain "
                "a valid message."
            )


        content = (
            message.get(
                "content"
            )
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


        return (
            content
        )