from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    caller_phone: str  
    caller_name: str  
    session_id: str  
    handoff: bool
    call_summary: str
