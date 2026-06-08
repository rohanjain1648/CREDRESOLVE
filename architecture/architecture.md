# System Architecture — CredResolve DCO Agent

## Full System Architecture

```mermaid
flowchart TD
    subgraph INPUT["🎤 Input Layer"]
        V["Voice\n(Hindi WAV/MP3)"]
        T["Text Chat\n(REST API)"]
        WS["WebSocket\n(Streaming)"]
    end

    subgraph VOICE_IN["Sarvam AI"]
        STT["Saarika v2\nSpeech → Text"]
    end

    subgraph CORE["⚙️ LangGraph Processing Core"]
        FSM["10-State FSM\n(LangGraph)"]
        LLM["Gemini 2.0 Flash\n(Intent · Negotiation · Evaluation)"]
        RAG["ChromaDB RAG\n(Policies + FAQs)"]
        MEM["SQLite Memory\n(User Profiles + History)"]
        TOOLS["LangChain Tools\n(CRM · Payment · Ticketing · Comms)"]
    end

    subgraph OUTPUT["📤 Output Layer"]
        TTS["Sarvam Bulbul v1\nText → Hindi Speech"]
        TEXT_OUT["Text Response"]
        CRM_OUT["CRM Update\n(PTP · Notes)"]
        TICKET_OUT["Escalation Ticket"]
        SMS_OUT["SMS / WhatsApp"]
    end

    subgraph OBS["📊 Observability"]
        PROM["Prometheus\n/metrics"]
        GRAF["Grafana\nDashboard"]
    end

    V --> STT --> FSM
    T --> FSM
    WS --> FSM

    FSM <--> LLM
    FSM <--> RAG
    FSM <--> MEM
    FSM <--> TOOLS

    FSM --> TTS
    FSM --> TEXT_OUT
    TOOLS --> CRM_OUT
    TOOLS --> TICKET_OUT
    TOOLS --> SMS_OUT

    FSM --> PROM --> GRAF

    style CORE fill:#1a1a2e,color:#ffffff,stroke:#4a90d9
    style INPUT fill:#0d3b66,color:#ffffff,stroke:#4a90d9
    style OUTPUT fill:#0d4f2f,color:#ffffff,stroke:#4a90d9
    style OBS fill:#3d1a47,color:#ffffff,stroke:#9b59b6
    style VOICE_IN fill:#4a2000,color:#ffffff,stroke:#e67e22
```

## Single-Call Data Flow

```mermaid
sequenceDiagram
    actor C as 👤 Customer (Hindi)
    participant API as FastAPI
    participant G as LangGraph FSM
    participant LLM as Gemini 2.0 Flash
    participant TOOLS as LangChain Tools
    participant RAG as ChromaDB
    participant MEM as SQLite Memory
    participant PROM as Prometheus

    C->>API: Hindi voice / text message
    API->>G: invoke(state, thread_id=session_id)

    G->>MEM: load user profile + prior history
    MEM-->>G: "1 prior PTP broken — cautious tone"

    G->>LLM: DIAGNOSIS — classify intent (JSON prompt)
    LLM-->>G: {"intent": "delay", "confidence": 0.92}

    G->>TOOLS: calculate_outstanding(LOAN001, 45 DPD, ₹87500)
    TOOLS-->>G: {total: ₹89850, interest: ₹1940, penalty: ₹1750}

    G->>RAG: query("PTP extension policy 45 DPD")
    RAG-->>G: [POL-003: max 30 days, 25% min partial...]

    G->>LLM: NEGOTIATION — generate Hindi response
    LLM-->>G: "Rajesh ji, 20 tarikh tak PTP kar sakti hoon..."

    G->>LLM: EVALUATION — hallucination + compliance check
    LLM-->>G: {overall_pass: true, compliance_violation: false}

    G->>TOOLS: log_ptp(LOAN001, 2024-06-20, ₹87500)
    TOOLS-->>G: {ptp_id: "PTP20240606001"}

    G->>TOOLS: send_sms(+91-9876543210, "PTP confirmed...")
    G->>MEM: log_interaction(intent=delay, outcome=ptp_logged)

    G-->>API: ConversationState + response text
    API->>PROM: record(latency, tokens, outcome=ptp_logged)
    API-->>C: Hindi text + optional Sarvam TTS audio
```

## RAG Pipeline

```mermaid
flowchart LR
    subgraph DOCS["📄 Source Documents"]
        P["policies.json\n8 policy chunks\n(RBI · Settlement · PTP\nEscalation · Hardship · Dispute)"]
        F["faqs.json\n8 Hindi FAQ pairs\n(EMI · CIBIL · Settlement\nPayment · Penalty · Moratorium)"]
    end

    subgraph EMBED["🔢 Embedding"]
        E["paraphrase-multilingual\nMiniLM-L12-v2\n384-dim vectors\nSupports Hindi natively"]
    end

    subgraph STORE["🗄️ Vector Store"]
        DB[("ChromaDB\nPersistent on disk\n./chroma_db/\ncosine similarity")]
    end

    subgraph QUERY["🔍 Retrieval at Runtime"]
        I["Borrower Intent\ne.g. settlement"]
        Q["Intent-Aware Query Builder\nsettlement → 'waiver discount\none-time payment policy'"]
        R["Top-3 Results\nrelevance threshold > 0.3\nwith source IDs cited"]
    end

    subgraph INJECT["💉 Context Injection"]
        CP["Context Prompt\n[POL-002] 5% waiver for 31-90 DPD...\n[POL-003] PTP max 30 days..."]
        LLM["Gemini 2.0 Flash\nGrounded Response"]
    end

    P --> E
    F --> E
    E --> DB

    I --> Q --> DB
    DB --> R --> CP --> LLM
```

## Memory Architecture

```mermaid
flowchart TD
    REQ["Incoming Customer Message"] --> NODE["LangGraph Node Execution"]

    subgraph L1["Layer 1 — In-Context  (ephemeral, current session only)"]
        MSG["messages: Annotated[List[dict], operator.add]\nAll turns appended, never replaced\nLives in ConversationState dict"]
    end

    subgraph L2["Layer 2 — LangGraph Checkpoint  (session-scoped, survives restarts)"]
        CP["SqliteSaver → langgraph_checkpoints.db\nFull ConversationState saved after every node\nRestored from thread_id on next HTTP request"]
    end

    subgraph L3["Layer 3 — User Profile  (permanent, cross-session)"]
        UP["user_profiles table\npreferred_language · risk_score\nsuccessful_ptps · broken_ptps\nhardship_declared"]
        IH["interaction_history table\nintent · outcome · ptp_date\nptp_amount · escalated\nsatisfaction_score"]
    end

    NODE --> MSG
    NODE --> CP
    NODE --> UP
    NODE --> IH

    MSG -- "same session\nnext turn" --> NODE
    CP -- "next HTTP request\nsame session_id" --> NODE
    UP -- "next session\nyears later" --> PERS["Personalised\nAgent Behaviour"]
    IH -- "next session\nyears later" --> PERS

    style L1 fill:#1a3a5c,color:#fff,stroke:#4a90d9
    style L2 fill:#1a4a2e,color:#fff,stroke:#27ae60
    style L3 fill:#4a1a00,color:#fff,stroke:#e67e22
```

## Tool Calling Flow

```mermaid
flowchart TD
    DX["DIAGNOSIS Node\nGemini classifies intent"] --> RT{Route by Intent}

    RT -->|"delay / cooperative"| OC["calculate_outstanding\nPrincipal + 18% p.a. interest\n+ 2% penalty after 30 DPD"]
    RT -->|"settlement"| SO["calculate_settlement_offer\n5% waiver 31-90 DPD\n10% waiver 91-180 DPD\n20% waiver 181+ DPD"]
    RT -->|"dispute"| VP["verify_payment\nLook up UTR/UPI\nin payment ledger"]
    RT -->|"all intents"| CD["fetch_customer_data\nCRM — name · outstanding\nDPD · payment history · risk"]

    subgraph TOOL1["🏦 CRM Tools"]
        CD
        LG["log_ptp\nPTP ID · date · amount"]
        UN["update_customer_notes\nTimestamped CRM note"]
    end

    subgraph TOOL2["💰 Payment Tools"]
        OC
        SO
        VP
        PH["get_payment_history\nFull transaction ledger"]
    end

    subgraph TOOL3["🎫 Ticketing Tools"]
        CT["create_ticket\ncategory · priority\nassigned_to: senior_dco"]
        UT["update_ticket\nstatus · comment"]
    end

    subgraph TOOL4["📱 Communication Tools"]
        SS["send_sms\nDLT-registered Hindi template"]
        SE["send_email\nConfirmation email"]
        SW["send_whatsapp\nWhatsApp Business API"]
    end

    OC --> NEG["NEGOTIATION\nPresent options in Hindi"]
    SO --> NEG
    VP --> NEG
    CD --> NEG

    NEG -->|"agreement"| LG --> SS
    NEG -->|"escalate"| CT
    CT --> ESC["ESCALATION\nSupervisor notified"]
    SS --> FU["FOLLOW_UP"]
    ESC --> FU
    FU --> UN
```

## Technology Stack

| Layer | Technology | Version | Role |
|-------|-----------|---------|------|
| Agent Framework | LangGraph | ≥0.1.0 | 10-state FSM + SQLite checkpointing |
| Orchestration | LangFlow | ≥1.0.0 | Visual workflow builder + JSON export |
| LLM | Gemini 2.5 Flash | gemini-2.5-flash | Reasoning + Hindi generation + thinking (free tier, GA June 2026) |
| LLM (lite) | Gemini 2.5 Flash-Lite | gemini-2.5-flash-lite | Low-latency fast responses, minimal cost |
| Vector Store | ChromaDB | ≥0.5.0 | Persistent cosine similarity search |
| Embeddings | MiniLM-L12-v2 | sentence-transformers | Multilingual Hindi-capable embeddings |
| Voice STT | Sarvam Saarika v2 | API | Hindi speech-to-text |
| Voice TTS | Sarvam Bulbul v1 | API | Hindi text-to-speech |
| Voice TTS (alt) | ElevenLabs multilingual v2 | API | Higher expressiveness fallback |
| Backend | FastAPI | ≥0.111.0 | REST + WebSocket + async |
| Memory (user) | SQLite + SQLAlchemy | built-in | Cross-session user profiles |
| Memory (graph) | LangGraph SqliteSaver | built-in | Per-session state checkpointing |
| Monitoring | Prometheus + Grafana | latest | 15 metrics + dashboards |
| Deployment | Docker Compose | latest | API + Prometheus + Grafana |
