from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from agent.graph import graph
from agent.state import AgentState

router = APIRouter()
log = logging.getLogger(__name__)

# Per-call mutex prevents concurrent graph invocations for the same call.
_call_locks: dict[str, asyncio.Lock] = {}
# Dedup: (transcript → reply) for the most recent turn of each call.
_last_turn: dict[str, tuple[str, str]] = {}


# ── helpers ────────────────────────────────────────────────────────────────────

def _extract_reply(result: dict[str, Any]) -> str:
    """Pull the text content out of the last message in the graph result."""
    last_msg = result["messages"][-1]
    content = getattr(last_msg, "content", last_msg)
    if isinstance(content, list):
        return " ".join(
            part["text"] if isinstance(part, dict) else str(part)
            for part in content
            if not isinstance(part, dict) or part.get("type") == "text"
        )
    return str(content)


def _openai_completion(text: str) -> dict:
    """
    Return a VAPI-compatible OpenAI chat-completion response (non-streaming).
    VAPI reads the text from choices[0].delta.content or choices[0].message.content.
    """
    return {
        "id": f"chatcmpl-{uuid4().hex[:24]}",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "delta": {"content": text},
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


# ── main endpoint ──────────────────────────────────────────────────────────────

@router.post("/vapi/webhook", tags=["Vapi"])
async def vapi_webhook(request: Request) -> JSONResponse:
    t_start = time.perf_counter()
    payload = await request.json()

    # Always log the full payload — critical for debugging
    log.info("VAPI raw payload:\n%s", json.dumps(payload, indent=2))
    log.info("VAPI top-level keys: %s", list(payload.keys()))

    # ── Route by payload shape ─────────────────────────────────────────────────
    #
    # Custom LLM request  →  {"messages": [...], "call": {...}, "stream": bool}
    #   VAPI sends this for every conversation turn when the assistant model
    #   is configured as "Custom LLM".  Transcript is in messages[-1].content.
    #
    # Lifecycle event     →  {"message": {"type": "...", "call": {...}, ...}}
    #   VAPI sends these for call lifecycle (status-update, end-of-call-report,
    #   assistant-request, etc.).  No agent processing needed.

    if isinstance(payload.get("messages"), list):
        return await _handle_custom_llm(payload, t_start)

    if isinstance(payload.get("message"), dict):
        return _handle_lifecycle(payload["message"])

    log.warning("VAPI unrecognised payload structure — keys=%s", list(payload.keys()))
    return JSONResponse(content={})


# ── lifecycle events (informational, no agent involvement) ─────────────────────

def _handle_lifecycle(message: dict) -> JSONResponse:
    """
    Handle VAPI server-URL lifecycle events.
    Most are informational — we just acknowledge them.
    """
    msg_type = message.get("type", "")
    call_id = message.get("call", {}).get("id", "") or message.get("callId", "")
    log.info("VAPI lifecycle | type=%r  call=%s", msg_type, call_id or "(none)")

    # Clean up in-memory state when a call ends
    if msg_type in ("end-of-call-report", "status-update"):
        status = message.get("status", "") or message.get("endedReason", "")
        log.info("VAPI lifecycle | status=%r", status)
        if call_id and status == "ended":
            _call_locks.pop(call_id, None)
            _last_turn.pop(call_id, None)
            log.info(
                "VAPI state cleanup | call=%s  active_locks=%d  cached_turns=%d",
                call_id, len(_call_locks), len(_last_turn),
            )

    # assistant-request: VAPI asks which assistant to use for a new call.
    # Return {} — VAPI will fall back to the assistant pre-configured on the
    # phone number / dashboard.  If you need to supply a dynamic assistant,
    # return {"assistantId": "..."} here.
    return JSONResponse(content={})


# ── Custom LLM per-turn handler ────────────────────────────────────────────────

async def _handle_custom_llm(payload: dict, t_start: float) -> JSONResponse:
    """
    Handle VAPI Custom LLM requests (OpenAI-compatible format).

    VAPI sends the full conversation history on every turn.  We extract only
    the latest user message and feed it to the LangGraph agent, which manages
    its own history via MemorySaver.
    """
    messages: list[dict] = payload["messages"]
    call: dict = payload.get("call", {})
    session_id: str = call.get("id") or str(uuid4())
    caller_phone: str = (
        call.get("customer", {}).get("number", "")
        or payload.get("customer", {}).get("number", "")
    )

    log.info(
        "VAPI custom LLM | session=%s  phone=%s  messages=%d",
        session_id, caller_phone or "(none)", len(messages),
    )

    # Log every message for full visibility
    for i, m in enumerate(messages):
        content = m.get("content", "")
        if isinstance(content, str):
            preview = content[:150]
        else:
            preview = " | ".join(
                p.get("text", "")[:60] for p in content if isinstance(p, dict)
            )
        log.info("  messages[%d] role=%-12s  %r", i, m.get("role", "?"), preview)

    # Extract the most recent user message as the current transcript
    transcript = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # OpenAI content-array format (e.g. multimodal)
                transcript = " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ).strip()
            else:
                transcript = str(content).strip()
            if transcript:
                break

    if not transcript:
        log.info("VAPI: no user message in payload | session=%s", session_id)
        return JSONResponse(
            content=_openai_completion("I didn't catch that, could you please repeat?")
        )

    log.info("VAPI user turn | session=%s  transcript=%r", session_id, transcript)

    # ── dedup + per-call serialisation lock ────────────────────────────────────
    if session_id not in _call_locks:
        _call_locks[session_id] = asyncio.Lock()

    async with _call_locks[session_id]:
        cached = _last_turn.get(session_id)
        if cached and cached[0] == transcript:
            log.info("VAPI dedup hit | session=%s", session_id)
            return JSONResponse(content=_openai_completion(cached[1]))

        # ── build LangGraph input ──────────────────────────────────────────────
        config: dict[str, Any] = {"configurable": {"thread_id": session_id}}
        existing = graph.get_state(config)
        is_first_turn = not existing.values

        if is_first_turn:
            log.info(
                "VAPI first turn | session=%s  phone=%s", session_id, caller_phone
            )
            input_state: AgentState = {
                "messages": [{"role": "user", "content": transcript}],
                "caller_phone": caller_phone,
                "session_id": session_id,
                "caller_name": "",
                "handoff": False,
                "call_summary": "",
            }
        else:
            current = existing.values
            log.info(
                "VAPI continuing | session=%s  turn=%d  caller_name=%r  handoff=%s",
                session_id,
                len(current.get("messages", [])),
                current.get("caller_name") or "",
                current.get("handoff", False),
            )
            input_state = {
                "messages": [{"role": "user", "content": transcript}],
                "caller_phone": caller_phone,
                "session_id": session_id,
            }

        # ── invoke the LangGraph agent ─────────────────────────────────────────
        t_graph = time.perf_counter()
        result = await graph.ainvoke(input_state, config=config)
        graph_ms = (time.perf_counter() - t_graph) * 1000

        log.info(
            "VAPI graph done | session=%s  messages=%d  caller_name=%r  handoff=%s  %.1fms",
            session_id,
            len(result.get("messages", [])),
            result.get("caller_name") or "",
            result.get("handoff", False),
            graph_ms,
        )

        reply = _extract_reply(result)
        _last_turn[session_id] = (transcript, reply)

    total_ms = (time.perf_counter() - t_start) * 1000
    log.info(
        "VAPI response | session=%s  reply=%r  total=%.1fms",
        session_id, reply[:200], total_ms,
    )
    return JSONResponse(content=_openai_completion(reply))
