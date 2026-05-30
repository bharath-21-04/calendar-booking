from __future__ import annotations

import logging

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
    tool_names = [tc["name"] for tc in getattr(last_ai, "tool_calls", [])]
    log.info("── tools_node ── executing: %s", tool_names)
    result = await _raw_tools_node.ainvoke(state, config=config)
    log.info("── tools_node ── finished: %s", tool_names)

    caller_name = state.get("caller_name") or ""
    if not caller_name and hasattr(last_ai, "tool_calls"):
        for tc in last_ai.tool_calls:
            name = tc.get("args", {}).get("caller_name", "")
            if name:
                caller_name = name
                log.info(
                    "tools_node: captured caller_name=%r from tool %s", name, tc["name"]
                )
                break

    if caller_name and caller_name != state.get("caller_name"):
        result["caller_name"] = caller_name

    return result


def agent_node(state: AgentState) -> dict:
    log.info("── agent_node ── (messages=%d)", len(state["messages"]))
    messages = state["messages"]

    if not messages or not isinstance(messages[0], SystemMessage):
        system_content = get_system_prompt()
        caller_phone = state.get("caller_phone") or ""
        caller_name = state.get("caller_name") or ""
        if caller_phone:
            system_content += (
                "\n\nCALLER INFO — already captured from call metadata, do NOT ask for these:\n"
                f"- Phone: {caller_phone}\n"
            )
            if caller_name:
                system_content += f"- Name: {caller_name}\n"
        messages = [SystemMessage(content=system_content)] + list(messages)

    try:
        ai_message = _model_with_tools.invoke(messages)
    except Exception as exc:
        error_text = str(exc)
        if "500" in error_text or "Internal Server Error" in error_text:
            user_msg = (
                "I'm having trouble reaching my AI service right now. "
                "Please try again in a moment, or I can connect you with the team."
            )
        else:
            user_msg = "Something went wrong on my end — please try again shortly."
        ai_message = AIMessage(content=user_msg)

    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        names = [tc["name"] for tc in ai_message.tool_calls]
        log.info("agent_node: LLM requesting tools %s", names)
    else:
        log.info("agent_node: LLM returning text reply (no tool calls)")

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
    tool_names_used: set[str] = set()
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_names_used.add(tc["name"])

    if not (tool_names_used & _MEANINGFUL_TOOLS):
        log.info("finalize_node: no meaningful tools used, skipping sheet write")
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
    summary = f"Tools used: {', '.join(sorted(tool_names_used & _MEANINGFUL_TOOLS))}"

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
