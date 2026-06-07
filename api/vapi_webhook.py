from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agent.graph import graph
from agent.state import AgentState
from config.settings import get_settings

router = APIRouter()
log = logging.getLogger(__name__)

# Per-call mutex prevents concurrent graph invocations for the same call.
_call_locks: dict[str, asyncio.Lock] = {}
# Dedup: (transcript → reply) for the most recent turn of each call.
_last_turn: dict[str, tuple[str, str]] = {}


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
    VAPI reads the text from choices[0].message.content.
    """
    return {
        "id": f"chatcmpl-{uuid4().hex[:24]}",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


def _openai_sse_stream(text: str):
    """
    Yield SSE chunks for a VAPI Custom LLM streaming response.
    VAPI always requests stream=true; we send the full text in one chunk
    then a stop chunk, followed by [DONE].
    """
    cid = f"chatcmpl-{uuid4().hex[:24]}"
    # Single content chunk
    chunk = json.dumps(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }
            ],
        }
    )
    yield f"data: {chunk}\n\n"
    # Stop chunk
    stop = json.dumps(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield f"data: {stop}\n\n"
    yield "data: [DONE]\n\n"


def _vapi_transfer_call(handoff_text: str, destination: str) -> dict:
    """
    Return a VAPI-compatible response that triggers a real phone transfer.

    VAPI intercepts the 'transferCall' tool call in the Custom LLM response
    and initiates the actual PSTN/SIP transfer to `destination`.
    The `content` field is spoken to the caller just before the transfer.
    """
    call_id = f"call_{uuid4().hex[:24]}"
    return {
        "id": f"chatcmpl-{uuid4().hex[:24]}",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": handoff_text,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "transferCall",
                                "arguments": json.dumps({"destination": destination}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


@router.post("/vapi/webhook/chat/completions", tags=["Vapi"])
async def vapi_custom_llm(request: Request) -> JSONResponse:
    t_start = time.perf_counter()
    payload = await request.json()
    return await _handle_custom_llm(payload, t_start)


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
    caller_phone: str = call.get("customer", {}).get("number", "") or payload.get(
        "customer", {}
    ).get("number", "")

    log.info(
        "VAPI custom LLM | session=%s  phone=%s  turn=%d",
        session_id,
        caller_phone or "(none)",
        len([m for m in messages if m.get("role") == "user"]),
    )

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

    log.info("USER  | session=%s  %r", session_id, transcript)

    # ── dedup + per-call serialisation lock ────────────────────────────────────
    if session_id not in _call_locks:
        _call_locks[session_id] = asyncio.Lock()

    async with _call_locks[session_id]:
        cached = _last_turn.get(session_id)
        if cached and cached[0] == transcript:
            log.info("DEDUP | session=%s  returning cached reply", session_id)
            return JSONResponse(content=_openai_completion(cached[1]))

        # ── build LangGraph input ──────────────────────────────────────────────
        config: dict[str, Any] = {"configurable": {"thread_id": session_id}}
        existing = graph.get_state(config)
        is_first_turn = not existing.values

        if is_first_turn:
            log.info("CALL START | session=%s  phone=%s", session_id, caller_phone)
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

        reply = _extract_reply(result)
        _last_turn[session_id] = (transcript, reply)

        log.info(
            "AGENT | session=%s  caller_name=%r  handoff=%s  %.0fms",
            session_id,
            result.get("caller_name") or "(none)",
            result.get("handoff", False),
            graph_ms,
        )

    total_ms = (time.perf_counter() - t_start) * 1000

    # ── Transfer the call if the agent escalated ───────────────────────────────
    if result.get("handoff"):
        transfer_number = get_settings().studio_transfer_number
        if transfer_number:
            log.info("TRANSFER | session=%s  → %s", session_id, transfer_number)
            return JSONResponse(content=_vapi_transfer_call(reply, transfer_number))
        else:
            log.warning(
                "TRANSFER requested but STUDIO_TRANSFER_NUMBER not set | session=%s",
                session_id,
            )

    log.info(
        "REPLY → VAPI | session=%s  stream=%s  content=%r",
        session_id,
        payload.get("stream"),
        reply[:400],
    )
    if payload.get("stream"):
        return StreamingResponse(
            _openai_sse_stream(reply), media_type="text/event-stream"
        )
    return JSONResponse(content=_openai_completion(reply))
