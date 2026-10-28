"""
prompt_caching.py -- Demonstrate prompt caching with Claude.

Caches the system prompt and static company context to reduce
repeated processing costs across requests.
"""

import anthropic

client = anthropic.Anthropic()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """
You are a business analyst assistant for Acme Corporation.
Answer questions using only the provided context.
Be specific with numbers.
"""

# Static context shared across queries.
STATIC_CONTEXT = """
Acme Corporation Q3 2025 Strategic Context:
- Product Alpha: Enterprise analytics, $48.2M ARR, 34 FTEs.
- Product Beta: SMB reporting, 4.2% monthly churn.
- Product Gamma: AI assistant, private beta, September 15 launch.
- Q3 target: $18.6M combined revenue.
"""


def query_with_caching(
    question: str,
    retrieved_chunks: list[str],
) -> str:
    """Query Claude with prompt caching enabled."""

    dynamic_context = "\n\n".join(retrieved_chunks)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": f"Company context:\n{STATIC_CONTEXT}",
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    f"Retrieved context:\n{dynamic_context}\n\n"
                    f"Question: {question}"
                ),
            }
        ],
    )

    # Report prompt caching performance.
    usage = response.usage

    print(f"Input tokens: {usage.input_tokens}")
    print(
        "Cache read tokens: "
        f"{getattr(usage, 'cache_read_input_tokens', 0)}"
    )
    print(
        "Cache write tokens: "
        f"{getattr(usage, 'cache_creation_input_tokens', 0)}"
    )

    return response.content[0].text
