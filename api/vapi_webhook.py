from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from agent.graph import graph
from agent.state import AgentState

router = APIRouter()
log = logging.getLogger(__name__)


_call_locks: dict[str, asyncio.Lock] = {}
_last_turn: dict[str, tuple[str, str]] = {}


def _extract_reply(result: dict[str, Any]) -> str:
    last_msg = result["messages"][-1]
    content = getattr(last_msg, "content", last_msg)
    if isinstance(content, list):
        return " ".join(
            part["text"] if isinstance(part, dict) else str(part)
            for part in content
            if not isinstance(part, dict) or part.get("type") == "text"
        )
    return str(content)


class VapiResponse(BaseModel):
    response: str = Field(..., description="Sol's reply — Vapi speaks this text.")


@router.post("/vapi/webhook", response_model=VapiResponse, tags=["Vapi"])
async def vapi_webhook(request: Request) -> VapiResponse:
    payload = await request.json()
    message = payload.get("message", payload)
    msg_type = message.get("type", "")

    if msg_type in ("end-of-call-report", "status-update"):
        call_id = message.get("call", {}).get("id") or message.get("callId", "")
        status = message.get("status", "") or message.get("endedReason", "")
        if call_id and status in ("ended", "idle"):
            _call_locks.pop(call_id, None)
            _last_turn.pop(call_id, None)
            log.info("Cleaned up state for ended call=%s", call_id)
        return VapiResponse(response="")

    if msg_type and msg_type != "assistant-request":
        log.debug("Ignoring VAPI message type=%s", msg_type)
        return VapiResponse(response="")

    call = message.get("call", {})
    session_id = call.get("id") or "unknown"
    caller_phone = call.get("customer", {}).get("number", "")

    transcript = ""
    for art_msg in reversed(message.get("artifact", {}).get("messages", [])):
        if art_msg.get("role") == "user":
            transcript = art_msg.get("message", "").strip()
            break
    if not transcript:
        transcript = message.get("transcript", "").strip()

    if not transcript:
        log.info(
            "Empty transcript — skipping | call=%s payload=%s", session_id, payload
        )
        return VapiResponse(response="")

    log.info(
        "vapi turn | call=%s phone=%s transcript=%r",
        session_id,
        caller_phone,
        transcript,
    )

    if session_id not in _call_locks:
        _call_locks[session_id] = asyncio.Lock()

    async with _call_locks[session_id]:
        cached = _last_turn.get(session_id)
        if cached and cached[0] == transcript:
            log.info("Dedup hit — returning cached reply for call=%s", session_id)
            return VapiResponse(response=cached[1])

        config: dict[str, Any] = {"configurable": {"thread_id": session_id}}
        existing = graph.get_state(config)
        is_first_turn = not existing.values

        if is_first_turn:
            input_state: AgentState = {
                "messages": [{"role": "user", "content": transcript}],
                "caller_phone": caller_phone,
                "session_id": session_id,
                "caller_name": "",
                "handoff": False,
                "call_summary": "",
            }
        else:
            input_state = {
                "messages": [{"role": "user", "content": transcript}],
                "caller_phone": caller_phone,
                "session_id": session_id,
            }

        result = await graph.ainvoke(input_state, config=config)

    reply = _extract_reply(result)

    _last_turn[session_id] = (transcript, reply)

    log.info("vapi reply | call=%s reply=%r", session_id, reply[:120])
    return VapiResponse(response=reply)
