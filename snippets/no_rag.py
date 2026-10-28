"""
no_rag.py -- Ask Claude questions about Acme Corp WITHOUT providing
any documents. This demonstrates the hallucination problem.

Usage: python no_rag.py

32M01: RAG Foundations -- Why LLMs Need Your Documents
The RAG Book

Expected: Claude will either refuse to answer or hallucinate plausible
but incorrect numbers.
"""

import anthropic

from dotenv import load_dotenv
load_dotenv()

# Initialize the Anthropic client using the ANTHROPIC_API_KEY environment variable.
client = anthropic.Anthropic()

# Questions about the Acme Corp memo that Claude has never seen.
questions = [
    "What was Product Alpha's Q2 revenue?",
    "What is the monthly churn rate for Product Beta?",
    "How many design partners are testing Product Gamma?",
    "What is the Q3 revenue target for Acme Corporation?",
    "What risks did Acme identify regarding Dataview Inc.?",
]

print("=" * 70)
print("EXPERIMENT: Claude WITHOUT documents (no RAG)")
print("=" * 70)

for i, question in enumerate(questions, start=1):
    print(f"\n--- Question {i} ---")
    print(f"Q: {question}")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=(
            "You are a helpful assistant. Answer the question based on "
            "what you know. If you are not sure, give your best estimate."
        ),
        messages=[
            {"role": "user", "content": question}
        ],
    )

    answer = response.content[0].text
    print(f"A: {answer}")

    # Track token usage for cost estimation.
    usage = response.usage
    print(
        f"\n[Tokens -- input: {usage.input_tokens}, "
        f"output: {usage.output_tokens}]"
    )

print("\n" + "=" * 70)
print("OBSERVATION: Notice how Claude either refuses to answer or")
print("generates plausible-sounding but WRONG numbers. These are")
print("hallucinations -- the model has never seen the Acme memo.")
print("=" * 70)
