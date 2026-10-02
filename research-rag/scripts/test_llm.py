from research_rag.config import (
    LLM_MODEL,
    LLM_THINK,
    OLLAMA_BASE_URL,
)

from research_rag.llm import (
    OllamaChatClient,
)


def main():

    print()

    print(
        "=" * 80
    )

    print(
        "Research RAG — LLM test"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"Ollama: {OLLAMA_BASE_URL}"
    )

    print(
        f"Model:  {LLM_MODEL}"
    )

    print(
        f"Think:  {LLM_THINK}"
    )

    print()


    client = (
        OllamaChatClient()
    )


    answer = client.chat(
        [
            {
                "role":
                    "system",

                "content":
                    (
                        "Follow the user's instruction "
                        "exactly and answer concisely."
                    ),
            },

            {
                "role":
                    "user",

                "content":
                    (
                        "Reply with exactly this text:\n"
                        "LLM connection OK"
                    ),
            },
        ]
    )


    print(
        "Response:"
    )

    print()

    print(
        answer
    )

    print()


    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()