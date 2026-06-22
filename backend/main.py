"""
CredResolve DCO Agent — FastAPI Application Entry Point
Exposes REST + WebSocket + Voice endpoints.
"""
import uuid
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.agent.graph import get_graph
from backend.agent.state_machine import ConversationState
from backend.agent.multi_agent.supervisor import run_multi_agent
from backend.rag.knowledge_base import build_knowledge_base
from backend.memory.user_memory import init_db, get_interaction_history, build_memory_summary
from backend.memory.conversation_memory import init_session_db, create_session, close_session
from backend.monitoring.metrics import (
    get_metrics_output, CONTENT_TYPE_LATEST, ACTIVE_SESSIONS, CONVERSATION_DURATION
)
from backend.voice.sarvam_voice import get_sarvam_client
from backend.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[Startup] Initializing databases...")
    init_db()
    init_session_db()
    print("[Startup] Building knowledge base...")
    build_knowledge_base()
    print("[Startup] CredResolve DCO Agent ready.")
    yield
    # Shutdown
    print("[Shutdown] Closing.")


app = FastAPI(
    title="CredResolve DCO Agent",
    description="AI-powered Hindi Debt Collection Officer with LangGraph + RAG + Voice",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
_FRONTEND = Path(__file__).parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


# ── Request / Response Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    loan_id: Optional[str] = None
    language: str = "hi"
    voice_enabled: bool = False


class ChatResponse(BaseModel):
    session_id: str
    message: str
    current_state: str
    intent: Optional[str] = None
    resolution_outcome: Optional[str] = None
    audio_url: Optional[str] = None
    sources: list[str] = []


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main text chat endpoint.
    Maintains conversation state via LangGraph + SQLite checkpointing.
    """
    session_id = req.session_id or str(uuid.uuid4())
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    ACTIVE_SESSIONS.inc()
    start = time.time()

    # The graph pauses (interrupt_after negotiation) awaiting the borrower's
    # reply. If this thread is mid-conversation, resume it with the new message
    # instead of restarting the whole greeting→auth→… pipeline.
    snapshot = graph.get_state(config)
    is_resuming = bool(getattr(snapshot, "next", None))

    if is_resuming:
        graph.update_state(
            config,
            {"messages": [{"role": "user", "content": req.message}]},
        )
        result = graph.invoke(None, config=config)
    else:
        # Build initial state for new sessions
        initial_state: ConversationState = {
            "messages": [{"role": "user", "content": req.message}],
            "session_id": session_id,
            "language": req.language,
            "voice_enabled": req.voice_enabled,
            "authenticated": False,
            "ptp_logged": False,
            "escalation_required": False,
            "negotiation_rounds": 0,
            "token_usage": 0,
            "tool_call_count": 0,
            "hallucination_flags": 0,
        }
        # If loan_id provided, pre-populate
        if req.loan_id:
            initial_state["customer_data"] = {"loan_id": req.loan_id}

        result = graph.invoke(initial_state, config=config)

    duration = time.time() - start
    CONVERSATION_DURATION.observe(duration)
    ACTIVE_SESSIONS.dec()

    # Extract last assistant message
    messages = result.get("messages", [])
    last_assistant = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "assistant"),
        "नमस्ते!"
    )

    # Optional TTS
    audio_url = None
    if req.voice_enabled and settings.sarvam_api_key:
        try:
            client = get_sarvam_client()
            audio_bytes = await client.text_to_speech(last_assistant, language="hi-IN")
            # In production: upload to S3 / GCS, return URL
            audio_url = f"/audio/{session_id}"
        except Exception:
            pass

    create_session(session_id, result.get("customer_id", ""), result.get("customer_data", {}).get("loan_id", ""))

    return ChatResponse(
        session_id=session_id,
        message=last_assistant,
        current_state=result.get("current_state", "unknown"),
        intent=result.get("intent"),
        resolution_outcome=result.get("resolution_outcome"),
        audio_url=audio_url,
        sources=result.get("retrieved_sources", []),
    )


@app.post("/voice/transcribe")
async def transcribe_voice(audio: UploadFile = File(...), language: str = "hi-IN"):
    """
    Upload voice audio, transcribe to text via Sarvam AI.
    Returns transcript for use in /chat endpoint.
    """
    audio_bytes = await audio.read()
    client = get_sarvam_client()
    try:
        transcript = await client.speech_to_text(
            audio_bytes,
            language=language,
            filename=audio.filename or "audio.webm",
            content_type=audio.content_type or "audio/webm",
        )
    except Exception as exc:
        # Always return JSON so the frontend can parse the error message.
        return JSONResponse(
            status_code=502,
            content={"transcript": "", "error": str(exc)},
        )
    return {"transcript": transcript, "language": language}


@app.get("/customer/{customer_id}/history")
async def get_customer_history(customer_id: str):
    """Part 7: Fetch interaction history for memory demo."""
    history = get_interaction_history(customer_id, limit=10)
    summary = build_memory_summary(customer_id)
    return {"customer_id": customer_id, "history": history, "summary": summary}


@app.get("/metrics")
async def metrics():
    """Part 8: Prometheus metrics endpoint."""
    return Response(content=get_metrics_output(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "CredResolve DCO", "version": "1.0.0"}


@app.get("/", response_class=FileResponse)
async def root():
    """Serve the frontend UI."""
    index = _FRONTEND / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return Response(content="Frontend not found. Visit /docs", media_type="text/plain")


class TTSRequest(BaseModel):
    text: str
    language: str = "hi-IN"


@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """Convert text to Hindi speech via Sarvam AI Bulbul v2 (WAV bytes).

    Sarvam's bulbul model is purpose-built for Indian languages, so Hindi
    sounds far more natural than English premade voices. Falls back to
    ElevenLabs if Sarvam fails.
    """
    try:
        client = get_sarvam_client()
        audio_bytes = await client.text_to_speech(req.text, language=req.language)
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as sarvam_exc:
        # Fallback to ElevenLabs so Play still works if Sarvam is unavailable.
        try:
            from backend.voice.elevenlabs_voice import get_elevenlabs_client
            audio_bytes = await get_elevenlabs_client().text_to_speech(req.text)
            return Response(content=audio_bytes, media_type="audio/mpeg")
        except Exception as eleven_exc:
            return JSONResponse(
                status_code=502,
                content={"error": f"Sarvam: {sarvam_exc} | ElevenLabs: {eleven_exc}"},
            )


# ── Part 10: Multi-Agent endpoint ─────────────────────────────────────────────

class MultiAgentChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    loan_id: Optional[str] = None
    language: str = "hi"


class MultiAgentChatResponse(BaseModel):
    session_id: str
    final_response: str
    intent: str
    evaluation_passed: bool
    hallucination_detected: bool
    compliance_violation: bool
    accuracy_score: float
    tone_score: float
    ptp_id: str
    ticket_id: str
    agent_trace: list  # full agent-to-agent communication log


@app.post("/multi-agent/chat", response_model=MultiAgentChatResponse)
async def multi_agent_chat(req: MultiAgentChatRequest):
    """
    Part 10 — Multi-Agent endpoint.
    Routes through: ContextAgent → RetrievalAgent → DecisionAgent
                    → ExecutionAgent → QAAgent
    Returns the full agent-to-agent communication trace alongside
    the validated Hindi response.
    """
    session_id = req.session_id or str(uuid.uuid4())
    ACTIVE_SESSIONS.inc()
    start = time.time()

    result = run_multi_agent(
        customer_message=req.message,
        session_id=session_id,
        loan_id=req.loan_id or "",
        language=req.language,
    )

    CONVERSATION_DURATION.observe(time.time() - start)
    ACTIVE_SESSIONS.dec()

    return MultiAgentChatResponse(
        session_id=session_id,
        final_response=result["final_response"],
        intent=result["intent"],
        evaluation_passed=result["evaluation_passed"],
        hallucination_detected=result["hallucination_detected"],
        compliance_violation=result["compliance_violation"],
        accuracy_score=result["accuracy_score"],
        tone_score=result["tone_score"],
        ptp_id=result["ptp_id"],
        ticket_id=result["ticket_id"],
        agent_trace=result["agent_trace"],
    )


# ── WebSocket — Real-time streaming conversation ──────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket for real-time streaming conversations.
    Supports both text messages and voice audio (base64).
    """
    await websocket.accept()
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}
    ACTIVE_SESSIONS.inc()

    state: ConversationState = {
        "messages": [],
        "session_id": session_id,
        "language": "hi",
        "voice_enabled": True,
        "authenticated": False,
        "ptp_logged": False,
        "escalation_required": False,
        "negotiation_rounds": 0,
        "token_usage": 0,
        "tool_call_count": 0,
        "hallucination_flags": 0,
    }

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "text")
            content = data.get("content", "")

            if msg_type == "voice_audio":
                # Transcribe audio
                import base64
                audio_bytes = base64.b64decode(content)
                client = get_sarvam_client()
                content = await client.speech_to_text(audio_bytes, language="hi-IN")
                await websocket.send_json({"type": "transcript", "content": content})

            state["messages"] = [{"role": "user", "content": content}]
            result = graph.invoke(state, config=config)

            messages = result.get("messages", [])
            reply = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "assistant"),
                "नमस्ते!"
            )

            await websocket.send_json({
                "type": "message",
                "content": reply,
                "state": result.get("current_state", ""),
                "intent": result.get("intent", ""),
                "sources": result.get("retrieved_sources", []),
            })

            state = result

    except WebSocketDisconnect:
        ACTIVE_SESSIONS.dec()
        close_session(session_id, state.get("current_state", "disconnected"))
