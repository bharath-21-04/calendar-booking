from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

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


def _openai_completion(text: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid4().hex[:24]}",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }


def _openai_sse_stream(text: str):
    cid = f"chatcmpl-{uuid4().hex[:24]}"
    chunk = json.dumps({
        "id": cid, "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
    })
    yield f"data: {chunk}\n\n"
    stop = json.dumps({
        "id": cid, "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })
    yield f"data: {stop}\n\n"
    yield "data: [DONE]\n\n"


def _vapi_transfer_call(handoff_text: str, destination: str) -> dict:
    call_id = f"call_{uuid4().hex[:24]}"
    return {
        "id": f"chatcmpl-{uuid4().hex[:24]}",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": handoff_text,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "transferCall", "arguments": json.dumps({"destination": destination})},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }


@router.post("/vapi/webhook/chat/completions", tags=["Vapi"])
async def vapi_custom_llm(request: Request) -> JSONResponse:
    payload = await request.json()
    return await _handle_custom_llm(payload)


async def _handle_custom_llm(payload: dict) -> JSONResponse:
    messages: list[dict] = payload["messages"]
    call: dict = payload.get("call", {})
    session_id: str = call.get("id") or str(uuid4())
    caller_phone: str = (
        call.get("customer", {}).get("number", "")
        or payload.get("customer", {}).get("number", "")
    )

    transcript = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                transcript = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ).strip()
            else:
                transcript = str(content).strip()
            if transcript:
                break

    if not transcript:
        return JSONResponse(content=_openai_completion("I didn't catch that, could you please repeat?"))

    log.info("USER | session=%s  %r", session_id, transcript)

    if session_id not in _call_locks:
        _call_locks[session_id] = asyncio.Lock()

    async with _call_locks[session_id]:
        cached = _last_turn.get(session_id)
        if cached and cached[0] == transcript:
            log.info("DEDUP | session=%s", session_id)
            return JSONResponse(content=_openai_completion(cached[1]))

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

        log.info("AGENT | session=%s  handoff=%s  reply=%r", session_id, result.get("handoff", False), reply[:200])

    if result.get("handoff"):
        transfer_number = get_settings().studio_transfer_number
        if transfer_number:
            return JSONResponse(content=_vapi_transfer_call(reply, transfer_number))
        log.warning("TRANSFER requested but STUDIO_TRANSFER_NUMBER not set | session=%s", session_id)

    if payload.get("stream"):
        return StreamingResponse(_openai_sse_stream(reply), media_type="text/event-stream")
    return JSONResponse(content=_openai_completion(reply))
