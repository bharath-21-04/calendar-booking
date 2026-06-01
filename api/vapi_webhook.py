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
from agent.tools import (
    list_upcoming_classes,
    check_class_availability,
    book_class,
    reschedule_booking,
    cancel_booking,
    get_studio_info,
    log_caller_note,
    escalate_to_human,
)
from config.settings import get_settings

# Registry of tools callable by VAPI's native LLM via server-URL tool calls.
_TOOL_REGISTRY: dict[str, Any] = {
    "list_upcoming_classes": list_upcoming_classes,
    "check_class_availability": check_class_availability,
    "book_class": book_class,
    "reschedule_booking": reschedule_booking,
    "cancel_booking": cancel_booking,
    "get_studio_info": get_studio_info,
    "log_caller_note": log_caller_note,
    "escalate_to_human": escalate_to_human,
}

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


# ── main endpoint ──────────────────────────────────────────────────────────────


@router.post("/vapi/webhook/chat/completions", tags=["Vapi"])
@router.post("/vapi/webhook", tags=["Vapi"])
async def vapi_webhook(request: Request) -> JSONResponse:
    t_start = time.perf_counter()
    payload = await request.json()

    # ── diagnose payload shape ─────────────────────────────────────────────────
    has_messages_list = isinstance(payload.get("messages"), list)
    has_message_dict = isinstance(payload.get("message"), dict)
    msg_type = (
        (payload.get("message") or {}).get("type", "") if has_message_dict else ""
    )
    log.info(
        "WEBHOOK | keys=%s  has_messages=%s  has_message=%s  msg_type=%r",
        list(payload.keys()),
        has_messages_list,
        has_message_dict,
        msg_type,
    )

    # ── Route by payload shape ─────────────────────────────────────────────────
    #
    # Custom LLM request  →  {"messages": [...], "call": {...}, "stream": bool}
    #   VAPI sends this for every conversation turn when the assistant model
    #   is configured as "Custom LLM".  Transcript is in messages[-1].content.
    #
    # Lifecycle event     →  {"message": {"type": "...", "call": {...}, ...}}
    #   VAPI sends these for call lifecycle (status-update, end-of-call-report,
    #   assistant-request, etc.).  No agent processing needed.

    if has_messages_list:
        log.info("WEBHOOK → Custom LLM handler  messages=%d", len(payload["messages"]))
        return await _handle_custom_llm(payload, t_start)

    if has_message_dict:
        msg = payload["message"]
        if msg.get("type") == "tool-calls":
            log.info("WEBHOOK → tool-calls handler")
            return await _handle_tool_calls(msg)
        log.info("WEBHOOK → lifecycle handler  type=%r", msg_type)
        return _handle_lifecycle(msg)

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
                call_id,
                len(_call_locks),
                len(_last_turn),
            )

    # assistant-request: VAPI asks which assistant to use for a new call.
    # Return {} — VAPI will fall back to the assistant pre-configured on the
    # phone number / dashboard.  If you need to supply a dynamic assistant,
    # return {"assistantId": "..."} here.
    return JSONResponse(content={})


# ── VAPI native-LLM tool-call handler ─────────────────────────────────────────


async def _handle_tool_calls(message: dict) -> JSONResponse:
    """
    Handle tool-calls sent by VAPI's native LLM (GPT-4.1 etc.) via server URL.

    VAPI sends:
      {"type": "tool-calls", "toolCallList": [{"id": "...", "function": {"name": ..., "arguments": "..."}}], "call": {...}}

    We execute each tool using the same implementations as the LangGraph agent
    and return:
      {"results": [{"toolCallId": "...", "result": "..."}]}
    """
    tool_call_list: list[dict] = message.get("toolCallList", [])
    call: dict = message.get("call", {})
    caller_phone: str = (
        call.get("customer", {}).get("number", "")
        or call.get("phoneNumber", {}).get("number", "")
        or ""
    )

    log.info(
        "VAPI tool-calls | count=%d  caller_phone=%s  tools=%s",
        len(tool_call_list),
        caller_phone or "(none)",
        [tc.get("function", {}).get("name") for tc in tool_call_list],
    )

    results: list[dict] = []
    for tc in tool_call_list:
        tool_id: str = tc.get("id", "")
        fn: dict = tc.get("function", {})
        tool_name: str = fn.get("name", "")

        try:
            args: dict = json.loads(fn.get("arguments", "{}"))
        except (json.JSONDecodeError, ValueError):
            args = {}

        log.info("  executing | name=%s  args=%s", tool_name, args)

        # Inject caller_phone from call metadata when the LLM omitted it.
        # These tools all accept caller_phone; inject it from VAPI metadata.
        _PHONE_TOOLS = {
            "book_class",
            "reschedule_booking",
            "cancel_booking",
            "log_caller_note",
            "escalate_to_human",
        }
        if tool_name in _PHONE_TOOLS and not args.get("caller_phone") and caller_phone:
            args["caller_phone"] = caller_phone
            log.info("  injected caller_phone=%s into %s", caller_phone, tool_name)

        tool_fn = _TOOL_REGISTRY.get(tool_name)
        if tool_fn is None:
            result = f"ERROR: unknown tool '{tool_name}'"
            log.warning("  unknown tool: %s", tool_name)
        else:
            try:
                result = tool_fn.invoke(args)
                log.info("  result | %s → %r", tool_name, str(result)[:200])
            except Exception as exc:
                result = f"ERROR: {tool_name} failed — {exc}"
                log.error("  tool error | %s: %s", tool_name, exc, exc_info=True)

        results.append({"toolCallId": tool_id, "result": str(result)})

    return JSONResponse(content={"results": results})


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
