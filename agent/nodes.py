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

    log.info("── tools_node ── executing: %s", tool_names)
    for tc in tool_calls:
        log.info(
            "  tool call | name=%s  args=%s",
            tc["name"],
            tc.get("args", {}),
        )

    t_tools = time.perf_counter()
    result = await _raw_tools_node.ainvoke(state, config=config)
    tools_ms = (time.perf_counter() - t_tools) * 1000

    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            log.info(
                "  tool result | name=%s  content=%r",
                msg.name,
                str(msg.content)[:300],
            )

    log.info("── tools_node ── finished: %s  %.1fms", tool_names, tools_ms)

    caller_name = state.get("caller_name") or ""
    caller_phone = state.get("caller_phone") or ""
    if hasattr(last_ai, "tool_calls"):
        for tc in last_ai.tool_calls:
            args = tc.get("args", {})
            if not caller_name:
                name = args.get("caller_name", "")
                if name:
                    caller_name = name
                    log.info(
                        "tools_node: captured caller_name=%r from tool %s",
                        name,
                        tc["name"],
                    )
            if not caller_phone:
                phone = args.get("caller_phone", "")
                if phone:
                    caller_phone = phone
                    log.info(
                        "tools_node: captured caller_phone=%r from tool %s",
                        phone,
                        tc["name"],
                    )

    if caller_name and caller_name != state.get("caller_name"):
        log.info("tools_node: updating state caller_name=%r", caller_name)
        result["caller_name"] = caller_name
    if caller_phone and caller_phone != state.get("caller_phone"):
        log.info("tools_node: updating state caller_phone=%r", caller_phone)
        result["caller_phone"] = caller_phone

    # Track escalation in state so downstream nodes and logs can see it
    if "escalate_to_human" in tool_names:
        log.info("tools_node: escalation detected — marking handoff=True")
        result["handoff"] = True

    return result


def agent_node(state: AgentState) -> dict:
    caller_phone = state.get("caller_phone") or ""
    caller_name = state.get("caller_name") or ""
    log.info(
        "── agent_node ── messages=%d  caller=%s  phone=%s",
        len(state["messages"]),
        caller_name or "(unknown)",
        caller_phone or "(none)",
    )
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
        log.info(
            "agent_node: system prompt built | phone_injected=%s  name_injected=%s  prompt_tail=%r",
            bool(caller_phone),
            bool(caller_name),
            system_content[-200:].replace("\n", " "),
        )

    log.info("agent_node: sending %d messages to LLM", len(messages))
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
            log.info("agent_node: LLM → tools %s  %.1fms", names, llm_ms)
            for tc in ai_message.tool_calls:
                log.info("  planned call | %s  args=%s", tc["name"], tc.get("args", {}))
        else:
            preview = str(ai_message.content)[:120].replace("\n", " ")
            log.info("agent_node: LLM → text reply  %.1fms  %r", llm_ms, preview)

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

    log.info("finalize_node: all tools used this session: %s", sorted(tool_names_used))

    if not (tool_names_used & _MEANINGFUL_TOOLS):
        log.info("finalize_node: no meaningful tools used, skipping sheet write")
        return {"call_summary": ""}

    caller_name = state.get("caller_name") or ""
    caller_phone = state.get("caller_phone") or ""
    log.info(
        "finalize_node: state has caller_name=%r  caller_phone=%r",
        caller_name,
        caller_phone,
    )
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
