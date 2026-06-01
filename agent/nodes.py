from __future__ import annotations

import logging
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.prompts import get_system_prompt
from agent.tools import ALL_TOOLS, _sheets
from config.settings import get_settings

log = logging.getLogger(__name__)


def _build_llm() -> BaseChatModel:
    settings = get_settings()
    provider = settings.model_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            api_key=settings.openai_api_key,
            verbose=True,
        )

    if provider == "claude":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            temperature=0,
            api_key=settings.anthropic_api_key,
            verbose=True,
        )

    if provider == "deepseek":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.deepseek_model,
            temperature=0,
            api_key=settings.deepseek_api_key,
            base_url="https://api.deepseek.com/v1",
            verbose=True,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openrouter_model,
            temperature=0,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0,
        google_api_key=settings.gemini_api_key,
    )


_llm = _build_llm()
_model_with_tools = _llm.bind_tools(ALL_TOOLS)


_raw_tools_node = ToolNode(ALL_TOOLS)


async def tools_node(state: AgentState, config: RunnableConfig) -> dict:
    last_ai = state["messages"][-1]
    tool_calls = getattr(last_ai, "tool_calls", [])
    tool_names = [tc["name"] for tc in tool_calls]

    log.info(
        "TOOL CALL | %s  args=%s",
        tool_names,
        {tc["name"]: tc.get("args", {}) for tc in tool_calls},
    )
    for tc in tool_calls:
        pass  # args already logged above

    t_tools = time.perf_counter()
    result = await _raw_tools_node.ainvoke(state, config=config)
    tools_ms = (time.perf_counter() - t_tools) * 1000

    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            log.info(
                "TOOL RESULT | %s → %r",
                msg.name,
                str(msg.content)[:400],
            )

    log.info("TOOL DONE | %s  %.0fms", tool_names, tools_ms)

    caller_name = state.get("caller_name") or ""
    caller_phone = state.get("caller_phone") or ""
    if hasattr(last_ai, "tool_calls"):
        for tc in last_ai.tool_calls:
            args = tc.get("args", {})
            if not caller_name:
                name = args.get("caller_name", "")
                if name:
                    caller_name = name
            if not caller_phone:
                phone = args.get("caller_phone", "")
                if phone:
                    caller_phone = phone

    if caller_name and caller_name != state.get("caller_name"):
        result["caller_name"] = caller_name
    if caller_phone and caller_phone != state.get("caller_phone"):
        result["caller_phone"] = caller_phone

    # Track escalation in state — but ONLY for legitimate reasons.
    # Booking / general requests must never trigger a transfer.
    _VALID_ESCALATION_KEYWORDS = frozenset(
        {
            "billing",
            "charge",
            "dispute",
            "refund",
            "aggressive",
            "abusive",
            "abuse",
            "complaint",
            "rude",
        }
    )
    if "escalate_to_human" in tool_names:
        escalation_reason = ""
        for tc in last_ai.tool_calls:
            if tc["name"] == "escalate_to_human":
                escalation_reason = tc.get("args", {}).get("reason", "").lower()

        is_valid_escalation = any(
            kw in escalation_reason for kw in _VALID_ESCALATION_KEYWORDS
        )

        if is_valid_escalation:
            log.info("ESCALATE | valid reason=%r", escalation_reason)
            result["handoff"] = True
        else:
            log.warning("ESCALATE BLOCKED | reason=%r", escalation_reason)
            # Override the ToolMessage so the LLM understands it must NOT transfer
            for msg in result.get("messages", []):
                if isinstance(msg, ToolMessage) and msg.name == "escalate_to_human":
                    msg.content = (
                        "BLOCKED: escalation is not permitted for this request. "
                        "This is a routine booking or information request — you must "
                        "handle it yourself. "
                        "Step 1: call list_upcoming_classes with the requested date. "
                        "Step 2: present the options and ask which class the caller wants. "
                        "Step 3: collect the caller's name if not already known. "
                        "Step 4: call book_class to complete the booking. "
                        "Do NOT attempt to escalate again for this request."
                    )

    return result


def agent_node(state: AgentState) -> dict:
    caller_phone = state.get("caller_phone") or ""
    caller_name = state.get("caller_name") or ""
    messages = state["messages"]

    building_system = not messages or not isinstance(messages[0], SystemMessage)
    if building_system:
        system_content = get_system_prompt()
        if caller_phone:
            system_content += (
                "\n\nCALLER INFO — already captured from call metadata, do NOT ask for these:\n"
                f"- Phone: {caller_phone}\n"
            )
            if caller_name:
                system_content += f"- Name: {caller_name}\n"
        messages = [SystemMessage(content=system_content)] + list(messages)

    log.debug("LLM sending %d messages", len(messages))
    t_llm = time.perf_counter()
    try:
        ai_message = _model_with_tools.invoke(messages)
    except Exception as exc:
        llm_ms = (time.perf_counter() - t_llm) * 1000
        log.error("agent_node: LLM error after %.1fms — %s", llm_ms, exc, exc_info=True)
        error_text = str(exc)
        if "500" in error_text or "Internal Server Error" in error_text:
            user_msg = (
                "I'm having trouble reaching my AI service right now. "
                "Please try again in a moment, or I can connect you with the team."
            )
        else:
            user_msg = "Something went wrong on my end — please try again shortly."
        ai_message = AIMessage(content=user_msg)
    else:
        llm_ms = (time.perf_counter() - t_llm) * 1000
        if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            names = [tc["name"] for tc in ai_message.tool_calls]
            log.info("LLM → TOOLS | %s  %.0fms", names, llm_ms)
        else:
            preview = str(ai_message.content)[:200].replace("\n", " ")
            log.info("LLM → REPLY | %.0fms  %r", llm_ms, preview)

    return {"messages": [ai_message]}


def handoff_node(state: AgentState) -> dict:
    log.info(
        "── handoff_node ── escalation complete for %s",
        state.get("caller_phone") or "unknown",
    )
    return {"handoff": True}


_MEANINGFUL_TOOLS = {
    "book_class",
    "reschedule_booking",
    "cancel_booking",
    "log_caller_note",
    "escalate_to_human",
}


_TOOL_TOPIC_TITLES = {
    "book_class": "Class Booking",
    "reschedule_booking": "Class Reschedule",
    "cancel_booking": "Class Cancellation",
    "log_caller_note": "General Inquiry",
    "escalate_to_human": "Escalation",
}


def finalize_node(state: AgentState) -> dict:
    log.info("── finalize_node ──")

    # Already wrote to Sheets for this session — don't double-increment call_count.
    if state.get("call_summary"):
        log.info("finalize_node: already logged this session, skipping re-write")
        return {}

    tool_names_used: set[str] = set()
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_names_used.add(tc["name"])

    log.info("FINALIZE | session tools=%s", sorted(tool_names_used))

    if not (tool_names_used & _MEANINGFUL_TOOLS):
        log.debug("FINALIZE | no meaningful tools, skipping sheet write")
        return {"call_summary": ""}

    caller_name = state.get("caller_name") or ""
    caller_phone = state.get("caller_phone") or ""
    if not caller_name or not caller_phone:
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] == "log_caller_note":
                        caller_name = caller_name or tc["args"].get("caller_name", "")
                        caller_phone = caller_phone or tc["args"].get(
                            "caller_phone", ""
                        )
                    elif tc["name"] in {
                        "book_class",
                        "reschedule_booking",
                        "cancel_booking",
                    }:
                        caller_name = caller_name or tc["args"].get("caller_name", "")
                        caller_phone = caller_phone or tc["args"].get(
                            "caller_phone", ""
                        )
        log.info(
            "finalize_node: resolved from tool args — caller_name=%r  caller_phone=%r",
            caller_name,
            caller_phone,
        )

    phone = caller_phone or "unknown"
    priority = [
        "book_class",
        "reschedule_booking",
        "cancel_booking",
        "escalate_to_human",
        "log_caller_note",
    ]
    raw_topic = next((t for t in priority if t in tool_names_used), "general")
    topic_title = _TOOL_TOPIC_TITLES.get(raw_topic, raw_topic.replace("_", " ").title())

    # Build human-readable notes from actual tool results and explicit note args.
    note_parts: list[str] = []

    # 1. Explicit notes from log_caller_note / escalate_to_human args.
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "log_caller_note":
                    n = tc["args"].get("notes", "").strip()
                    if n:
                        note_parts.append(n)
                elif tc["name"] == "escalate_to_human":
                    r = tc["args"].get("reason", "").strip()
                    if r:
                        note_parts.append(f"Escalated: {r}")

    # 2. Confirmation strings returned by booking/cancellation tools.
    tool_call_ids = {
        tc["id"]
        for msg in state["messages"]
        if isinstance(msg, AIMessage)
        for tc in getattr(msg, "tool_calls", [])
        if tc["name"] in {"book_class", "reschedule_booking", "cancel_booking"}
    }
    for msg in state["messages"]:
        if (
            isinstance(msg, ToolMessage)
            and msg.tool_call_id in tool_call_ids
            and msg.content
            and not str(msg.content).startswith("ERROR")
        ):
            note_parts.append(str(msg.content).strip())

    summary = "; ".join(note_parts) if note_parts else topic_title

    log.info(
        "finalize_node: writing to sheet | phone=%s  name=%r  topic=%r  summary=%r",
        phone,
        caller_name,
        topic_title,
        summary,
    )
    try:
        _sheets.upsert_caller(
            phone=phone,
            name=caller_name,
            topic=topic_title,
            notes=summary,
        )
        log.info(
            "finalize_node: sheet upsert done | phone=%s topic=%r", phone, topic_title
        )
    except Exception as exc:
        log.error("finalize_node: sheets upsert failed: %s", exc, exc_info=True)

    return {"call_summary": summary}
