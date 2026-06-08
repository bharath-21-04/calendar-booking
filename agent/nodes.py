from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
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
        return ChatOpenAI(model=settings.openai_model, temperature=0, api_key=settings.openai_api_key)

    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=settings.anthropic_model, temperature=0, api_key=settings.anthropic_api_key)

    if provider == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.deepseek_model, temperature=0, api_key=settings.deepseek_api_key, base_url="https://api.deepseek.com/v1")

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.openrouter_model, temperature=0, api_key=settings.openrouter_api_key, base_url="https://openrouter.ai/api/v1")

    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0, google_api_key=settings.gemini_api_key)


_llm = _build_llm()
_model_with_tools = _llm.bind_tools(ALL_TOOLS)
_raw_tools_node = ToolNode(ALL_TOOLS)

_ESCALATION_KEYWORDS = frozenset({
    "billing dispute", "charge dispute", "dispute",
    "aggressive", "abusive", "abuse", "complaint", "rude", "threatening",
})

_MEANINGFUL_TOOLS = {"book_class", "reschedule_booking", "cancel_booking", "log_caller_note", "escalate_to_human"}

_TOOL_TOPIC_TITLES = {
    "book_class": "Class Booking",
    "reschedule_booking": "Class Reschedule",
    "cancel_booking": "Class Cancellation",
    "log_caller_note": "General Inquiry",
    "escalate_to_human": "Escalation",
}


async def tools_node(state: AgentState, config: RunnableConfig) -> dict:
    last_ai = state["messages"][-1]
    tool_calls = getattr(last_ai, "tool_calls", [])
    tool_names = [tc["name"] for tc in tool_calls]

    log.info("TOOL CALL | %s", tool_names)
    result = await _raw_tools_node.ainvoke(state, config=config)

    caller_name = state.get("caller_name") or ""
    caller_phone = state.get("caller_phone") or ""
    for tc in tool_calls:
        args = tc.get("args", {})
        caller_name = caller_name or args.get("caller_name", "")
        caller_phone = caller_phone or args.get("caller_phone", "")

    if caller_name and caller_name != state.get("caller_name"):
        result["caller_name"] = caller_name
    if caller_phone and caller_phone != state.get("caller_phone"):
        result["caller_phone"] = caller_phone

    if "escalate_to_human" in tool_names:
        reason = next(
            (tc.get("args", {}).get("reason", "") for tc in tool_calls if tc["name"] == "escalate_to_human"),
            "",
        ).lower()
        if any(kw in reason for kw in _ESCALATION_KEYWORDS):
            result["handoff"] = True
        else:
            for msg in result.get("messages", []):
                if isinstance(msg, ToolMessage) and msg.name == "escalate_to_human":
                    msg.content = (
                        "BLOCKED: escalation is not permitted for this request. "
                        "Handle it yourself: call list_upcoming_classes, present options, "
                        "collect the caller's name, then call book_class. Do NOT escalate again."
                    )

    return result


def agent_node(state: AgentState) -> dict:
    caller_phone = state.get("caller_phone") or ""
    caller_name = state.get("caller_name") or ""
    messages = state["messages"]

    if not messages or not isinstance(messages[0], SystemMessage):
        system_content = get_system_prompt()
        if caller_phone:
            system_content += f"\n\nCALLER INFO — do NOT ask for these:\n- Phone: {caller_phone}\n"
            if caller_name:
                system_content += f"- Name: {caller_name}\n"
        messages = [SystemMessage(content=system_content)] + list(messages)

    try:
        ai_message = _model_with_tools.invoke(messages)
    except Exception as exc:
        log.error("agent_node: LLM error — %s", exc, exc_info=True)
        err = str(exc)
        msg = (
            "I'm having trouble reaching my AI service right now. Please try again in a moment."
            if "500" in err or "Internal Server Error" in err
            else "Something went wrong on my end — please try again shortly."
        )
        ai_message = AIMessage(content=msg)

    return {"messages": [ai_message]}


def handoff_node(state: AgentState) -> dict:
    return {"handoff": True}


def finalize_node(state: AgentState) -> dict:
    if state.get("call_summary"):
        return {}

    tool_names_used: set[str] = set()
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_names_used.add(tc["name"])

    if not (tool_names_used & _MEANINGFUL_TOOLS):
        return {"call_summary": ""}

    caller_name = state.get("caller_name") or ""
    caller_phone = state.get("caller_phone") or ""
    if not caller_name or not caller_phone:
        booking_tools = {"book_class", "reschedule_booking", "cancel_booking", "log_caller_note"}
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] in booking_tools:
                        caller_name = caller_name or tc["args"].get("caller_name", "")
                        caller_phone = caller_phone or tc["args"].get("caller_phone", "")

    priority = ["book_class", "reschedule_booking", "cancel_booking", "escalate_to_human", "log_caller_note"]
    raw_topic = next((t for t in priority if t in tool_names_used), "general")
    topic_title = _TOOL_TOPIC_TITLES.get(raw_topic, raw_topic.replace("_", " ").title())

    note_parts: list[str] = []
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "log_caller_note":
                    if n := tc["args"].get("notes", "").strip():
                        note_parts.append(n)
                elif tc["name"] == "escalate_to_human":
                    if r := tc["args"].get("reason", "").strip():
                        note_parts.append(f"Escalated: {r}")

    booking_tool_ids = {
        tc["id"]
        for msg in state["messages"]
        if isinstance(msg, AIMessage)
        for tc in getattr(msg, "tool_calls", [])
        if tc["name"] in {"book_class", "reschedule_booking", "cancel_booking"}
    }
    for msg in state["messages"]:
        if (
            isinstance(msg, ToolMessage)
            and msg.tool_call_id in booking_tool_ids
            and msg.content
            and not str(msg.content).startswith("ERROR")
        ):
            note_parts.append(str(msg.content).strip())

    summary = "; ".join(note_parts) if note_parts else topic_title

    try:
        _sheets.upsert_caller(
            phone=caller_phone or "unknown",
            name=caller_name,
            topic=topic_title,
            notes=summary,
        )
    except Exception as exc:
        log.error("finalize_node: sheets upsert failed: %s", exc, exc_info=True)

    return {"call_summary": summary}
