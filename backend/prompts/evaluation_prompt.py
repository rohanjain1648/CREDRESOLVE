"""
Part 3: Evaluation Prompt — hallucination check, compliance, accuracy
"""

HALLUCINATION_CHECK_PROMPT = """
आपको नीचे दिए गए agent response की जांच करनी है।

## Agent का उत्तर
{agent_response}

## उपलब्ध स्रोत
{retrieved_sources}

## ग्राहक डेटा
{customer_data}

निम्नलिखित की जांच करें:
1. क्या agent ने कोई ऐसी जानकारी दी जो स्रोतों में नहीं है? (hallucination)
2. क्या agent ने RBI दिशानिर्देशों का उल्लंघन किया? (compliance)
3. क्या राशि और तारीखें सही हैं? (accuracy)
4. क्या टोन विनम्र और पेशेवर है? (tone)

JSON में उत्तर दें:
{{
  "hallucination_detected": true/false,
  "hallucinated_claims": ["<list of claims>"],
  "compliance_violation": true/false,
  "violation_details": "<null or description>",
  "accuracy_score": <0.0-1.0>,
  "tone_score": <0.0-1.0>,
  "overall_pass": true/false,
  "corrected_response": "<null or corrected version>"
}}
"""

RETRIEVAL_QUALITY_PROMPT = """
Evaluate whether the retrieved documents are relevant to the query.

Query: {query}
Retrieved Documents:
{documents}

Score each document:
- relevance: 0.0-1.0
- coverage: does it answer the query?

Return JSON:
{{
  "average_relevance": <0.0-1.0>,
  "coverage": true/false,
  "missing_information": "<what is missing>",
  "recommended_requery": "<null or better query string>"
}}
"""


def get_hallucination_check_prompt(
    agent_response: str,
    retrieved_sources: list,
    customer_data: dict,
) -> str:
    sources_text = "\n".join(f"- {s}" for s in retrieved_sources)
    return HALLUCINATION_CHECK_PROMPT.format(
        agent_response=agent_response,
        retrieved_sources=sources_text or "कोई स्रोत नहीं",
        customer_data=str(customer_data),
    )


def get_retrieval_quality_prompt(query: str, documents: list) -> str:
    docs_text = "\n\n".join(
        f"[{i+1}] {d}" for i, d in enumerate(documents)
    )
    return RETRIEVAL_QUALITY_PROMPT.format(query=query, documents=docs_text)
