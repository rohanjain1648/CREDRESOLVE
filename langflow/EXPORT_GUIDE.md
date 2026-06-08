# LangFlow Export Guide — CredResolve DCO Agent

## What's in this folder

| File | What it is |
|------|-----------|
| [workflow.json](workflow.json) | LangFlow-compatible JSON — import directly into LangFlow UI |
| [workflow.yaml](workflow.yaml) | YAML equivalent — human-readable, use as documentation / alternative deliverable |
| [EXPORT_GUIDE.md](EXPORT_GUIDE.md) | This file — how to import, run, and export |

---

## Step 1 — Install LangFlow

```bash
pip install langflow
```

Or with uv (faster):

```bash
pip install uv
uv pip install langflow
```

Minimum Python version: **3.10+**

---

## Step 2 — Start LangFlow server

```bash
langflow run
```

LangFlow opens at `http://127.0.0.1:7860` in your browser.

---

## Step 3 — Import `workflow.json`

1. Click **"My Flows"** in the top-left sidebar.
2. Click **"Import"** (upload icon, top-right of the flows grid).
3. Select `langflow/workflow.json` from this project.
4. The flow appears as **"CredResolve DCO Agent"**.

> **Tip**: If import fails due to a version mismatch, open `workflow.json`, find any `"type"` field that doesn't match your LangFlow version's component registry, and rename it using the LangFlow component search panel.

---

## Step 4 — Configure API keys in the canvas

After importing, click each node that has a key icon and fill in:

| Node | Field | Where to get the key |
|------|-------|---------------------|
| Gemini 2.0 Flash | `google_api_key` | https://aistudio.google.com/app/apikey (free) |
| Sarvam STT | `api_key` | https://dashboard.sarvam.ai |
| Sarvam TTS | `api_key` | Same Sarvam dashboard |

---

## Step 5 — Export from LangFlow UI

### Export as JSON (official LangFlow export)

1. Open the flow in the canvas editor.
2. Click the **download icon** (⬇) in the top-right toolbar — it looks like a cloud with a down-arrow.
3. LangFlow saves `CredResolve DCO Agent.json` to your Downloads folder.
4. This is the **official, importable LangFlow JSON** — it contains your API key configuration, exact component versions, and any changes you made in the UI.

### Export as API-ready JSON via LangFlow REST API

With the server running, call:

```bash
curl http://127.0.0.1:7860/api/v1/flows/ \
  -H "Content-Type: application/json" | python -m json.tool > langflow_api_export.json
```

This returns all flows. To get a specific flow by ID:

```bash
# 1. List flows to get the UUID
curl http://127.0.0.1:7860/api/v1/flows/

# 2. Export by UUID
curl http://127.0.0.1:7860/api/v1/flows/<FLOW_UUID> > workflow_official.json
```

### Export as YAML (manual)

LangFlow's native format is JSON. The `workflow.yaml` in this folder is a manually maintained equivalent — keep it in sync when you update the JSON.

---

## Step 6 — Run the flow via LangFlow UI

1. Open the flow, click **"Playground"** (speech bubble icon, top-right).
2. In the chat window, type: `नमस्ते, मेरा loan ID LOAN001 है`
3. The agent greets, authenticates, and begins the DCO conversation.

---

## Step 7 — Run via LangFlow Python SDK

```python
from langflow.load import run_flow_from_json

result = run_flow_from_json(
    flow="langflow/workflow.json",
    input_value="नमस्ते, मेरा loan LOAN001 है",
    session_id="session-001",
    fallback_to_env_vars=True,   # reads GOOGLE_API_KEY etc. from environment
)

print(result[0].outputs[0].results["message"].text)
```

---

## Step 8 — Run via LangFlow REST API

Once the server is running, call the deployed endpoint:

```bash
curl -X POST http://127.0.0.1:7860/api/v1/run/<FLOW_ID> \
  -H "Content-Type: application/json" \
  -d '{
    "input_value": "मेरा loan ID LOAN001 है",
    "output_type": "chat",
    "input_type": "chat",
    "session_id": "session-001"
  }'
```

---

## Node map — 3 layers

```
INPUT LAYER                 PROCESSING LAYER                   OUTPUT LAYER
─────────────────────────   ────────────────────────────────   ──────────────────────────
ChatInput-text-001          SQLiteMemory-user-001              ChatOutput-text-001
(REST text / UI chat)       (cross-session SQLite profile)     (REST response / WebSocket)

CustomComponent-            ConversationBufferWindow           CustomComponent-
  sarvam-stt-001            Memory-001                           sarvam-tts-001
(Saarika v2 STT)            (in-session 10-turn window)        (Bulbul v1 TTS → WAV)

CustomComponent-            HuggingFaceEmbeddings-001          CustomComponent-
  api-webhook-001           (MiniLM 384-dim Hindi vectors)       crm-write-001
(IVR / WhatsApp API)                                           (log_ptp + update_notes)
                            Chroma-retriever-001
                            (ChromaDB cosine sim, k=3)         CustomComponent-
                                                                 ticket-write-001
                            PromptTemplate-system-001          (create_ticket → senior_dco)
                            (Hindi sys + CRM + RAG + history)
                                                               CustomComponent-
                            ChatGoogleGenerativeAI-001           sms-write-001
                            (gemini-2.5-flash, temp=0.3)       (DLT Hindi SMS / WhatsApp)

                            Tool-crm-fetch-001                 CustomComponent-
                            Tool-payment-001                     prometheus-001
                            Tool-ticketing-001                 (15 Prometheus metrics)

                            CustomComponent-
                              langgraph-agent-001
                            (10-state FSM, SqliteSaver)

                            CustomComponent-eval-001
                            (hallucination + RBI check)
```

---

## CRM & Ticket operations — complete audit

All writes happen inside `backend/agent/nodes.py`. No external CRM API required for the demo — mock data is in `backend/tools/crm_tool.py` (`MOCK_CRM_DB`).

| Operation | Type | Node | File:Line | What's written |
|-----------|------|------|-----------|---------------|
| `fetch_customer_data(loan_id)` | CRM READ | AUTHENTICATION | [nodes.py ~128](../backend/agent/nodes.py) | Reads name, outstanding, DPD, risk |
| `upsert_user_profile(...)` | SQLite WRITE | CONTEXT_GATHERING | [nodes.py ~148](../backend/agent/nodes.py) | Writes/updates user_profiles table |
| `verify_payment(loan_id, txn_id)` | Payment READ | TOOL_EXECUTION | [nodes.py ~228](../backend/agent/nodes.py) | Checks UTR/UPI in mock ledger |
| `calculate_outstanding(...)` | Payment READ | TOOL_EXECUTION | [nodes.py ~234](../backend/agent/nodes.py) | Returns interest + penalty total |
| `calculate_settlement_offer(...)` | Payment READ | TOOL_EXECUTION | [nodes.py ~244](../backend/agent/nodes.py) | Returns tiered discount amount |
| `create_ticket(...)` | Ticket WRITE | ESCALATION | [nodes.py ~355](../backend/agent/nodes.py) | Creates ticket → returns ticket_id |
| `log_ptp(loan_id, date, amount)` | CRM WRITE | RESOLUTION | [nodes.py ~391](../backend/agent/nodes.py) | Writes PTP → `PTP20240606001` |
| `send_sms(phone, message)` | SMS SEND | RESOLUTION | [nodes.py ~411](../backend/agent/nodes.py) | Sends Hindi PTP confirmation |
| `log_interaction(...)` | SQLite WRITE | FOLLOW_UP | [nodes.py ~438](../backend/agent/nodes.py) | Writes to interaction_history table |
| `update_customer_notes(loan_id, note)` | CRM WRITE | FOLLOW_UP | [nodes.py ~457](../backend/agent/nodes.py) | Appends call note to CRM record |

**Yes — CRM and ticket updates are fully implemented.**
