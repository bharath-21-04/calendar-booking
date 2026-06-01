from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.state import AgentState
from agent.nodes import agent_node, tools_node, handoff_node, finalize_node


def _route_after_agent(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools_node"
    return "finalize_node"


def _route_after_tools(state: AgentState) -> str:
    if state.get("handoff"):
        return "handoff_node"
    return "agent_node"


def build_graph() -> "CompiledStateGraph":
    builder = StateGraph(AgentState)

    builder.add_node("agent_node", agent_node)
    builder.add_node("tools_node", tools_node)
    builder.add_node("handoff_node", handoff_node)
    builder.add_node("finalize_node", finalize_node)

    builder.add_edge(START, "agent_node")

    builder.add_conditional_edges(
        "agent_node",
        _route_after_agent,
        {"tools_node": "tools_node", "finalize_node": "finalize_node"},
    )

    builder.add_conditional_edges(
        "tools_node",
        _route_after_tools,
        {"handoff_node": "handoff_node", "agent_node": "agent_node"},
    )

    builder.add_edge("handoff_node", END)
    builder.add_edge("finalize_node", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
