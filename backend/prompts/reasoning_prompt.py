"""
Part 3: Reasoning Prompt — structured chain-of-thought for decision making
"""

INTENT_CLASSIFICATION_PROMPT = """
नीचे दिए गए उधारकर्ता के संदेश को पढ़कर उनका इरादा पहचानें।

उधारकर्ता का संदेश: "{user_message}"
ग्राहक संदर्भ: {context_summary}

निम्नलिखित में से एक intent चुनें:
- delay: ग्राहक भुगतान में देरी चाहते हैं
- refusal: ग्राहक भुगतान से मना कर रहे हैं
- dispute: ग्राहक राशि पर विवाद कर रहे हैं
- settlement: ग्राहक एकमुश्त समझौता चाहते हैं
- angry: ग्राहक क्रोधित या परेशान हैं
- cooperative: ग्राहक सहयोग के लिए तैयार हैं
- unknown: स्पष्ट नहीं

JSON में उत्तर दें:
{{
  "intent": "<intent>",
  "confidence": <0.0-1.0>,
  "reasoning": "<हिंदी में संक्षिप्त कारण>",
  "suggested_response_strategy": "<next step>"
}}
"""

NEGOTIATION_REASONING_PROMPT = """
आप एक debt collection officer हैं। निम्नलिखित जानकारी के आधार पर सर्वोत्तम प्रस्ताव तैयार करें।

## ग्राहक की स्थिति
- Intent: {intent}
- बकाया राशि: ₹{outstanding_amount}
- देरी: {days_past_due} दिन
- EMI क्षमता (अनुमानित): ₹{estimated_capacity}

## उपलब्ध नीति विकल्प (RAG से)
{policy_options}

## पिछले वार्ता चरण: {negotiation_round}

## आपका कार्य
चरणबद्ध सोचें:
1. ग्राहक की वास्तविक समस्या क्या है?
2. कौन सा विकल्प उनके लिए व्यावहारिक है?
3. इस विकल्प से CredResolve को क्या लाभ होगा?
4. प्रस्ताव हिंदी में कैसे प्रस्तुत करें?

JSON में उत्तर दें:
{{
  "recommended_option": "settlement|restructuring|ptp|escalate",
  "proposed_amount": <number or null>,
  "proposed_date": "<YYYY-MM-DD or null>",
  "hindi_pitch": "<ग्राहक को दिया जाने वाला हिंदी संदेश>",
  "escalate_reason": "<null or escalation reason>"
}}
"""

FOLLOW_UP_QUESTIONS = {
    "delay": [
        "आप किस तारीख तक भुगतान कर सकते हैं?",
        "क्या आंशिक भुगतान अभी संभव है?",
    ],
    "refusal": [
        "क्या आप मुझे बता सकते हैं कि भुगतान में क्या कठिनाई है?",
        "क्या EMI राशि कम करना आपके लिए मददगार होगा?",
    ],
    "dispute": [
        "आप किस राशि से असहमत हैं?",
        "क्या आपके पास भुगतान का कोई प्रमाण है?",
    ],
    "settlement": [
        "आप एकमुश्त कितनी राशि दे सकते हैं?",
        "क्या आप अगले 7 दिनों में भुगतान कर सकते हैं?",
    ],
}


def get_intent_prompt(user_message: str, context_summary: str = "") -> str:
    return INTENT_CLASSIFICATION_PROMPT.format(
        user_message=user_message,
        context_summary=context_summary or "पहली बातचीत",
    )


def get_negotiation_prompt(
    intent: str,
    outstanding_amount: float,
    days_past_due: int,
    policy_options: str,
    negotiation_round: int = 1,
    estimated_capacity: float = 0,
) -> str:
    return NEGOTIATION_REASONING_PROMPT.format(
        intent=intent,
        outstanding_amount=outstanding_amount,
        days_past_due=days_past_due,
        policy_options=policy_options,
        negotiation_round=negotiation_round,
        estimated_capacity=estimated_capacity,
    )
