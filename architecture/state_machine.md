# Part 1: State Machine Design — CredResolve DCO Agent

## State Diagram

```mermaid
stateDiagram-v2
    [*] --> GREETING

    GREETING --> AUTHENTICATION

    AUTHENTICATION --> CONTEXT_GATHERING : authenticated = true
    AUTHENTICATION --> ESCALATION : auth failed (3 attempts)

    CONTEXT_GATHERING --> DIAGNOSIS

    DIAGNOSIS --> TOOL_EXECUTION : delay / settlement / dispute / cooperative
    DIAGNOSIS --> KNOWLEDGE_RETRIEVAL : refusal / unknown
    DIAGNOSIS --> ESCALATION : angry / emotional

    TOOL_EXECUTION --> KNOWLEDGE_RETRIEVAL

    KNOWLEDGE_RETRIEVAL --> NEGOTIATION

    NEGOTIATION --> RESOLUTION : agreement reached
    NEGOTIATION --> ESCALATION : rounds ≥ 3 or high risk

    RESOLUTION --> FOLLOW_UP
    ESCALATION --> FOLLOW_UP

    FOLLOW_UP --> [*]

    state TOOL_EXECUTION {
        [*] --> CRM_Fetch
        CRM_Fetch --> Payment_Calculator
        Payment_Calculator --> Settlement_Calculator
        Settlement_Calculator --> [*]
    }

    state NEGOTIATION {
        [*] --> Round_1
        Round_1 --> Round_2 : no agreement
        Round_2 --> Round_3 : no agreement
        Round_3 --> [*]
    }
```

## Routing Logic Diagram

```mermaid
flowchart TD
    START([Session Start]) --> G[GREETING\nArya introduces herself in Hindi]
    G --> A[AUTHENTICATION\nVerify Loan ID via CRM]

    A -->|authenticated = true| CG[CONTEXT_GATHERING\nLoad CRM data + SQLite memory]
    A -->|auth failed| ESC

    CG --> D[DIAGNOSIS\nGemini classifies borrower intent as JSON]

    D -->|intent = angry| ESC[ESCALATION\nCreate ticket · Schedule callback\nDo NOT negotiate]
    D -->|intent = delay / settlement\ncooperative / dispute| TE[TOOL_EXECUTION\nCRM · Payment Calculator\nSettlement Offer · Verify Payment]
    D -->|intent = refusal / unknown| KR

    TE --> KR[KNOWLEDGE_RETRIEVAL\nChromaDB RAG query\nTop-3 policies by cosine similarity]

    KR --> N[NEGOTIATION\nPresent options in Hindi\nTrack rounds max 3]

    N -->|customer agrees| R[RESOLUTION\nlog_ptp · send_sms\nUpdate CRM]
    N -->|rounds ≥ 3 or escalation flag| ESC

    R --> FU[FOLLOW_UP\nLog to SQLite · Update CRM\nClose session]
    ESC --> FU

    FU --> END([Session End])

    style ESC fill:#8B0000,color:#fff
    style R fill:#006400,color:#fff
    style N fill:#00008B,color:#fff
    style D fill:#4B0082,color:#fff
```

## State Reference Table

| # | State | Entry Condition | Key Actions | Tools Called | Exit Condition | Failure Path |
|---|-------|----------------|-------------|-------------|----------------|-------------|
| 1 | **GREETING** | New session started | Introduce Arya in Hindi | — | Greeting delivered | → END |
| 2 | **AUTHENTICATION** | Greeting done | Request Loan ID; verify via CRM | `fetch_customer_data` | CRM record found | → ESCALATION after 3 failures |
| 3 | **CONTEXT_GATHERING** | `authenticated=True` | Load CRM profile; fetch SQLite memory summary | `fetch_customer_data` | Customer data loaded | Proceed with minimal context |
| 4 | **DIAGNOSIS** | Context loaded; customer spoke | LLM classifies intent via structured JSON prompt | — | Intent + confidence returned | → NEGOTIATION with `unknown` |
| 5 | **TOOL_EXECUTION** | Intent ∈ {delay, settlement, dispute, cooperative} | Call payment/CRM/verification tools based on intent | `calculate_outstanding` · `calculate_settlement_offer` · `verify_payment` | Tool results returned | → ESCALATION |
| 6 | **KNOWLEDGE_RETRIEVAL** | Tools done (or direct from DIAGNOSIS) | RAG query to ChromaDB; intent-aware query string | ChromaDB | ≥1 doc with relevance > 0.3 | Proceed with empty policies |
| 7 | **NEGOTIATION** | Policies retrieved | Present Hindi options; track round count (max 3) | — | Agreement OR round limit | → ESCALATION |
| 8 | **ESCALATION** | intent=angry OR auth fail OR DPD>90 + refusal | Create ticket; schedule callback; notify supervisor | `create_ticket` · `send_sms` | Ticket created | → FOLLOW_UP |
| 9 | **RESOLUTION** | Customer agrees to PTP or settlement | Log PTP in CRM; send SMS confirmation | `log_ptp` · `send_sms` | PTP/settlement logged | → ESCALATION |
| 10 | **FOLLOW_UP** | Resolution or escalation complete | Write to SQLite; update CRM notes; close session | `update_customer_notes` | Session ends | → END |

## Transition Table

| From | Condition | To |
|------|-----------|-----|
| START | Session initiated | GREETING |
| GREETING | Always | AUTHENTICATION |
| AUTHENTICATION | `authenticated=True` | CONTEXT_GATHERING |
| AUTHENTICATION | `authenticated=False` | ESCALATION |
| CONTEXT_GATHERING | Always | DIAGNOSIS |
| DIAGNOSIS | `intent=angry` | ESCALATION |
| DIAGNOSIS | `intent ∈ {delay, settlement, dispute, cooperative}` | TOOL_EXECUTION |
| DIAGNOSIS | `intent ∈ {refusal, unknown}` | KNOWLEDGE_RETRIEVAL |
| TOOL_EXECUTION | Always | KNOWLEDGE_RETRIEVAL |
| KNOWLEDGE_RETRIEVAL | Always | NEGOTIATION |
| NEGOTIATION | Agreement reached | RESOLUTION |
| NEGOTIATION | `escalation_required=True` OR rounds ≥ 3 | ESCALATION |
| RESOLUTION | Always | FOLLOW_UP |
| ESCALATION | Always | FOLLOW_UP |
| FOLLOW_UP | Always | END |
