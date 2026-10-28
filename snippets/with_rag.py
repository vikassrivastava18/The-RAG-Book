"""
with_rag.py -- Ask Claude questions using the Acme Corp memo as context.
This demonstrates the Retrieval-Augmented Generation (RAG) approach.

Usage: python with_rag.py

Expected: Claude answers using information from the provided memo.
"""

import anthropic

from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic()

# Acme Corp Q3 Strategic Planning Memo.
# In a real RAG system, this context would be retrieved from a vector database.
ACME_MEMO = """
ACME CORPORATION — Q3 STRATEGIC PLANNING MEMO
Prepared by: Sarah Chen, VP of Strategy
Date: July 15, 2025

EXECUTIVE SUMMARY
Acme Corporation enters Q3 2025 with strong momentum from Product Alpha's
enterprise launch but faces increased competition in the mid-market segment.

PRODUCT PORTFOLIO UPDATE

Product Alpha (Enterprise Analytics Platform)
- Q2 revenue: $12.4M, 18% above forecast.
- Secured 23 new enterprise contracts.
- Current ARR run rate: $48.2M.
- Key risk: Dataview Inc. launched a competing product at a 40% lower price.
- Three pipeline deals worth $2.1M combined are under competitive review.
- Dataview is winning on price but losing on reliability and integration depth.

Product Beta (SMB Reporting Tool)
- Q2 revenue: $3.8M, 6% below forecast.
- Monthly churn increased to 4.2%, up from 2.8% in Q1.
- 62% of churned customers never completed initial setup.
- Customer satisfaction dropped from 4.1 to 3.6.
- Q3 target: Reduce monthly churn to 2.5% by the end of Q3.
- Budget: $1.8M for onboarding improvements and $600K for UX redesign.

Product Gamma (AI Data Assistant)
- Private beta with 12 design partners.
- Planned launch date: September 15, 2025.
- 9 of 12 partners rated the natural language query feature as transformative.
- Estimated total addressable market: $2.8B by 2027.
- Planned Q3 investment: $3.2M.

FINANCIAL OUTLOOK
- Q3 combined revenue target: $18.6M.
- Operating margin target: 12%.
- Planned new hires: 14.
- Annualized compensation budget increase: $2.4M.

RISK REGISTER
1. Dataview pricing pressure on Alpha (High impact, Medium likelihood).
2. Beta churn if onboarding improvements underperform
   (Medium impact, Medium likelihood).
3. Gamma launch delays due to API stability issues
   (High impact, Low likelihood).
4. Macroeconomic slowdown affecting enterprise sales cycles
   (High impact, Medium likelihood).
5. Three senior Alpha engineers have received external offers
   (High impact, Medium likelihood).
"""

SYSTEM_PROMPT = """
You are a helpful assistant answering questions about internal company documents.

Follow these rules:
1. Use only the provided context.
2. If the context lacks the answer, say:
   "The provided documents do not contain this information."
3. Cite the relevant section of the document.
4. Use exact numbers and facts from the context.
5. Do not invent or infer unsupported information.
"""

# Same questions as no_rag.py for comparison.
questions = [
    "What is the total Q2 revenue across all products?",
    # "What was Product Alpha's Q2 revenue?",
    # "What is the monthly churn rate for Product Beta?",
    # "How many design partners are testing Product Gamma?",
    # "What is the Q3 revenue target for Acme Corporation?",
    # "What risks did Acme identify regarding Dataview Inc.?",
]

print("=" * 70)
print("EXPERIMENT: Claude WITH documents (RAG approach)")
print("=" * 70)

for i, question in enumerate(questions, start=1):
    print(f"\n--- Question {i} ---")
    print(f"Q: {question}")

    # Combine the retrieved context with the question.
    user_message = f"""
CONTEXT:
{ACME_MEMO}

QUESTION:
{question}

Answer using only the provided context and cite the relevant section.
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
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
print("OBSERVATION: Answers are grounded in the provided document.")
print("Relevant facts and numbers can be cited directly.")
print("RAG helps reduce hallucinations by providing relevant context.")
print("=" * 70)
