from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from agent.graph import graph
from agent.state import AgentState
from config.settings import get_settings

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
    t_start = time.perf_counter()

    # Validate VAPI webhook secret.
    settings = get_settings()
    if settings.vapi_webhook_secret:
        incoming = request.headers.get("x-vapi-secret", "")
        if incoming != settings.vapi_webhook_secret:
            log.warning("VAPI webhook: rejected request — invalid x-vapi-secret header")
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

    payload = await request.json()
    log.debug("VAPI raw payload: %s", json.dumps(payload, indent=2))

    message = payload.get("message", payload)
    msg_type = message.get("type", "")

    log.info("── VAPI WEBHOOK ── type=%r", msg_type)

    # ── lifecycle events ──────────────────────────────────────────────────────
    if msg_type in ("end-of-call-report", "status-update"):
        call_id = message.get("call", {}).get("id") or message.get("callId", "")
        status = message.get("status", "") or message.get("endedReason", "")
        log.info(
            "lifecycle event | type=%s call=%s status=%s", msg_type, call_id, status
        )
        if call_id and status in ("ended", "idle"):
            _call_locks.pop(call_id, None)
            _last_turn.pop(call_id, None)
            log.info(
                "state cleanup | call=%s  active_locks=%d  cached_turns=%d",
                call_id,
                len(_call_locks),
                len(_last_turn),
            )
        return VapiResponse(response="")

    if msg_type and msg_type != "assistant-request":
        log.info("ignoring unsupported message type=%r", msg_type)
        return VapiResponse(response="")

    # ── parse call metadata ───────────────────────────────────────────────────
    call = message.get("call", {})
    session_id = call.get("id") or "unknown"
    caller_phone = call.get("customer", {}).get("number", "")
    assistant_id = call.get("assistantId") or call.get("assistant", {}).get("id", "")

    log.info(
        "call metadata | session=%s  phone=%s  assistant=%s",
        session_id,
        caller_phone or "(none)",
        assistant_id or "(none)",
    )

    # ── extract transcript ────────────────────────────────────────────────────
    artifact = message.get("artifact", {})
    artifact_msgs = artifact.get("messages", [])
    log.debug("artifact message count: %d", len(artifact_msgs))

    transcript = ""
    for art_msg in reversed(artifact_msgs):
        if art_msg.get("role") == "user":
            transcript = art_msg.get("message", "").strip()
            log.debug("transcript extracted from artifact[role=user]: %r", transcript)
            break
    if not transcript:
        transcript = message.get("transcript", "").strip()
        if transcript:
            log.debug("transcript extracted from message.transcript: %r", transcript)

    if not transcript:
        log.info(
            "empty transcript — no user speech detected, skipping | call=%s", session_id
        )
        return VapiResponse(response="")

    log.info("user transcript | call=%s  %r", session_id, transcript)

    # ── dedup + lock ──────────────────────────────────────────────────────────
    if session_id not in _call_locks:
        _call_locks[session_id] = asyncio.Lock()
        log.debug("created lock for new session=%s", session_id)

    async with _call_locks[session_id]:
        cached = _last_turn.get(session_id)
        if cached and cached[0] == transcript:
            log.info("dedup hit — returning cached reply | call=%s", session_id)
            return VapiResponse(response=cached[1])

        # ── session state ─────────────────────────────────────────────────────
        config: dict[str, Any] = {"configurable": {"thread_id": session_id}}
        existing = graph.get_state(config)
        is_first_turn = not existing.values

        if is_first_turn:
            log.info(
                "first turn — initialising session state | call=%s  phone=%s",
                session_id,
                caller_phone,
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
                "continuing session | call=%s  turn=%d  caller_name=%r  handoff=%s",
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

        log.debug(
            "invoking graph | session=%s  state_keys=%s",
            session_id,
            list(input_state.keys()),
        )

        # ── graph invocation ──────────────────────────────────────────────────
        t_graph = time.perf_counter()
        result = await graph.ainvoke(input_state, config=config)
        graph_ms = (time.perf_counter() - t_graph) * 1000

        final_state = result
        log.info(
            "graph done | call=%s  messages=%d  caller_name=%r  handoff=%s  %.1fms",
            session_id,
            len(final_state.get("messages", [])),
            final_state.get("caller_name") or "",
            final_state.get("handoff", False),
            graph_ms,
        )

    # ── build reply ───────────────────────────────────────────────────────────
    reply = _extract_reply(result)
    _last_turn[session_id] = (transcript, reply)

    total_ms = (time.perf_counter() - t_start) * 1000
    log.info(
        "VAPI reply | call=%s  reply=%r  total=%.1fms",
        session_id,
        reply[:200],
        total_ms,
    )
    return VapiResponse(response=reply)
