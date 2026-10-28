"""
full_pipeline.py -- Complete RAG pipeline in one file.

Stages:
1. Ingestion: Load the Acme Corp memo.
2. Chunking: Split the document into sections and chunks.
3. Embedding: Convert chunks into vectors using OpenAI.
4. Retrieval: Find the top-k chunks using cosine similarity.
5. Generation: Answer questions using Claude.

Usage:
    python full_pipeline.py

Requires:
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
"""

import time

import numpy as np
from openai import OpenAI
from anthropic import Anthropic

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "claude-sonnet-4-6"

TOP_K = 3
MAX_CHUNK_SIZE = 500
CHUNK_OVERLAP_WORDS = 50


# ============================================================
# STAGE 1: INGESTION
# ============================================================

def ingest_document(text: str, metadata: dict | None = None) -> dict:
    """Convert raw text into a document record."""

    print("[Stage 1: Ingestion]")

    text = text.strip()

    document = {
        "text": text,
        "metadata": metadata or {},
        "char_count": len(text),
        "ingested_at": time.time(),
    }

    print(f"  Ingested document: {document['char_count']} characters")

    return document


# ============================================================
# STAGE 2: CHUNKING
# ============================================================

def chunk_by_section(
    text: str,
    max_chunk_size: int = MAX_CHUNK_SIZE,
) -> list[dict]:
    """Split text into section-based, overlapping chunks."""

    print("[Stage 2: Chunking]")

    sections = []
    current_section = {"title": "Header", "lines": []}

    for line in text.splitlines():
        stripped = line.strip()

        # Detect uppercase section headings.
        is_header = (
            stripped
            and stripped == stripped.upper()
            and len(stripped) > 5
            and not any(char.isdigit() for char in stripped[:3])
        )

        if is_header:
            if current_section["lines"]:
                sections.append(current_section)

            current_section = {"title": stripped, "lines": []}
        else:
            current_section["lines"].append(line)

    if current_section["lines"]:
        sections.append(current_section)

    chunks = []
    step = max_chunk_size - CHUNK_OVERLAP_WORDS

    if step <= 0:
        raise ValueError(
            "max_chunk_size must exceed CHUNK_OVERLAP_WORDS."
        )

    for section in sections:
        section_text = "\n".join(section["lines"]).strip()

        if not section_text:
            continue

        words = section_text.split()

        # Split large sections into overlapping chunks.
        for start in range(0, len(words), step):
            chunk_text = " ".join(
                words[start : start + max_chunk_size]
            )

            chunks.append({
                "text": chunk_text,
                "section": section["title"],
                "index": len(chunks),
            })

            if start + max_chunk_size >= len(words):
                break

    print(f"  Created {len(chunks)} chunks from {len(sections)} sections")

    for chunk in chunks:
        word_count = len(chunk["text"].split())
        print(
            f"  Chunk {chunk['index']}: "
            f"[{chunk['section']}] {word_count} words"
        )

    return chunks


# ============================================================
# STAGE 3: EMBEDDING AND STORAGE
# ============================================================

class VectorStore:
    """In-memory vector store using NumPy and OpenAI embeddings."""

    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.openai_client = OpenAI()
        self.embedding_model = embedding_model
        self.vectors = []
        self.chunks = []
        self.embed_time = 0.0

    def add_chunks(self, chunks: list[dict]) -> None:
        """Generate embeddings and store them with their chunks."""

        print("[Stage 3: Embedding and Storage]")

        if not chunks:
            raise ValueError("No chunks available to embed.")

        texts = [chunk["text"] for chunk in chunks]

        start_time = time.perf_counter()

        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )

        self.embed_time = time.perf_counter() - start_time

        for chunk, embedding_data in zip(
            chunks, response.data, strict=True
        ):
            self.vectors.append(
                np.array(embedding_data.embedding, dtype=np.float32)
            )
            self.chunks.append(chunk)

        print(f"  Embedded {len(chunks)} chunks")
        print(f"  Model: {self.embedding_model}")
        print(f"  Dimensions: {len(self.vectors[0])}")
        print(f"  Total tokens: {response.usage.total_tokens}")
        print(f"  Embedding time: {self.embed_time:.2f}s")

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Retrieve the top-k chunks using cosine similarity."""

        if not self.vectors:
            raise ValueError("Vector store is empty. Add chunks first.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=[query],
        )

        query_vector = np.array(
            response.data[0].embedding,
            dtype=np.float32,
        )

        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            raise ValueError("Query embedding has zero magnitude.")

        # Calculate cosine similarity between the query and each chunk.
        scores = []

        for index, document_vector in enumerate(self.vectors):
            document_norm = np.linalg.norm(document_vector)

            if document_norm == 0:
                similarity = 0.0
            else:
                similarity = np.dot(
                    query_vector, document_vector
                ) / (query_norm * document_norm)

            scores.append((index, float(similarity)))

        # Rank chunks by similarity, highest first.
        scores.sort(key=lambda item: item[1], reverse=True)

        results = []

        for index, score in scores[:top_k]:
            results.append({
                **self.chunks[index],
                "similarity_score": score,
            })

        return results


# ============================================================
# STAGE 5: GENERATION
# ============================================================

def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    model: str = GENERATION_MODEL,
) -> dict:
    """Generate an answer using Claude and retrieved context."""

    print("[Stage 5: Generation]")

    if not retrieved_chunks:
        raise ValueError("No relevant chunks were retrieved.")

    client = Anthropic()

    # Build context from retrieved chunks.
    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {index}: {chunk['section']}] "
            f"(similarity: {chunk['similarity_score']:.3f})\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """
You answer questions about internal company documents.

Follow these rules:
1. Use only the provided context.
2. If the context lacks the answer, say so explicitly.
3. Cite the relevant source, such as [Source 1].
4. Use exact numbers and facts from the context.
5. Be concise and do not invent unsupported information.
"""

    user_message = f"""
RETRIEVED CONTEXT:
{context}

QUESTION:
{question}

Answer using only the provided context and cite your sources.
"""

    start_time = time.perf_counter()

    response = client.messages.create(
        model=model,
        max_tokens=500,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ],
    )

    latency_ms = round(
        (time.perf_counter() - start_time) * 1000
    )

    answer = response.content[0].text
    usage = response.usage

    # Illustrative cost estimate; verify current model pricing.
    cost = (
        usage.input_tokens * 3 / 1_000_000
        + usage.output_tokens * 15 / 1_000_000
    )

    print(f"  Model: {model}")
    print(f"  Input tokens: {usage.input_tokens}")
    print(f"  Output tokens: {usage.output_tokens}")
    print(f"  Estimated cost: ${cost:.6f}")
    print(f"  Latency: {latency_ms}ms")

    return {
        "answer": answer,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": cost,
        "latency_ms": latency_ms,
    }


# ============================================================
# ACME CORPORATION MEMO
# ============================================================

ACME_MEMO = """
ACME CORPORATION — Q3 STRATEGIC PLANNING MEMO
Prepared by: Sarah Chen, VP of Strategy
Date: July 15, 2025

EXECUTIVE SUMMARY
Acme Corporation enters Q3 2025 with strong momentum from Product Alpha's
enterprise launch but faces increased competition in the mid-market segment.
This memo outlines strategic priorities, resource allocation, and risk
mitigation plans for the quarter.

PRODUCT PORTFOLIO UPDATE

Product Alpha (Enterprise Analytics Platform)
Q2 revenue: $12.4M (18% above forecast). Alpha secured 23 new enterprise
contracts, including Meridian Healthcare ($1.2M ACV) and Titan Manufacturing
($890K ACV). Current ARR run rate: $48.2M. The v3.2 release added real-time
dashboard streaming. Engineering headcount: 34 FTEs.

Key risk: Dataview Inc. launched a competing product at a 40% lower price.
Three pipeline deals worth $2.1M combined are under competitive review.
Dataview is winning on price but losing on reliability and integration depth.

Product Beta (SMB Reporting Tool)
Q2 revenue: $3.8M (6% below forecast). Monthly churn increased to 4.2%,
up from 2.8% in Q1. Poor onboarding is the primary cause: 62% of churned
customers never completed initial setup. Customer satisfaction dropped
from 4.1 to 3.6.

The Q3 "Beta Quick Start" program will provide dedicated onboarding
specialists. Target: reduce monthly churn to 2.5% by the end of Q3.
Budget: $1.8M for onboarding improvements and $600K for UX redesign.

Product Gamma (AI Data Assistant)
Status: Private beta with 12 design partners. Launch planned for
September 15, 2025. Nine of 12 partners rated the natural language query
feature as "transformative."

Pricing under review: $45/user/month for teams and $120/user/month for
enterprise. Four well-funded startups compete in this space. Acme's
advantage is deep integration with Alpha's data layer.

Estimated TAM: $2.8B by 2027. Go-to-market investment: $3.2M in Q3,
including product marketing ($1.4M), developer relations ($800K),
and launch events ($1M).

FINANCIAL OUTLOOK
Q3 combined revenue target: $18.6M. Operating margin target: 12%.
Headcount plan: 14 new hires: engineering (8), sales (4), and customer
success (2). Annualized compensation budget increase: $2.4M.

RISK REGISTER
1. Dataview pricing pressure on Alpha
   (High impact, Medium likelihood).
2. Beta churn if onboarding improvements underperform
   (Medium impact, Medium likelihood).
3. Gamma launch delay due to API stability issues
   (High impact, Low likelihood).
4. Macroeconomic slowdown affecting enterprise sales cycles
   (High impact, Medium likelihood).
5. Three senior Alpha engineers have received external offers
   (High impact, Medium likelihood).
"""


# ============================================================
# MAIN: RUN THE COMPLETE PIPELINE
# ============================================================

QUESTIONS = [
    "What was Product Alpha's Q2 revenue and how does it compare to forecast?",
    "What is causing Product Beta's churn problem?",
    "When is Product Gamma launching and what is the go-to-market budget?",
    "What are the highest-impact risks facing Acme this quarter?",
    "How many new hires are planned and in which departments?",
]


def main() -> None:
    print("=" * 70)
    print("COMPLETE RAG PIPELINE")
    print("=" * 70)

    # Stage 1: Ingestion.
    document = ingest_document(
        ACME_MEMO,
        metadata={"source": "Q3 Strategy Memo"},
    )

    # Stage 2: Chunking.
    chunks = chunk_by_section(document["text"])

    # Stage 3: Embedding and storage.
    store = VectorStore()
    store.add_chunks(chunks)

    # Stages 4 and 5: Retrieval and generation.
    print("\n" + "=" * 70)
    print("ANSWERING QUESTIONS")
    print("=" * 70)

    total_cost = 0.0
    total_latency = 0

    for index, question in enumerate(QUESTIONS, start=1):
        print("\n" + "=" * 70)
        print(f"Question {index}: {question}")
        print("=" * 70)

        # Stage 4: Retrieval.
        print("[Stage 4: Retrieval]")

        start_time = time.perf_counter()
        results = store.search(question, top_k=TOP_K)

        retrieval_ms = (
            time.perf_counter() - start_time
        ) * 1000

        print(
            f"  Retrieved {len(results)} chunks "
            f"in {retrieval_ms:.0f}ms"
        )

        for result in results:
            preview = result["text"][:80].replace("\n", " ")

            print(
                f"  [{result['section']}] "
                f"score={result['similarity_score']:.3f}: "
                f"{preview}..."
            )

        # Stage 5: Generation.
        result = generate_answer(question, results)

        print(f"\nANSWER: {result['answer']}")

        total_cost += result["cost_usd"]
        total_latency += result["latency_ms"]

    # Pipeline summary.
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print(f"  Questions answered: {len(QUESTIONS)}")
    print(f"  Total generation cost: ${total_cost:.6f}")
    print(
        f"  Average generation latency: "
        f"{total_latency / len(QUESTIONS):.0f}ms"
    )
    print(f"  One-time embedding time: {store.embed_time:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
