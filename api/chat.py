from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agent.graph import graph
from agent.state import AgentState

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str
    phone: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    handoff: bool = False
    session_id: str


def _extract_content(msg) -> str:
    content = msg.content if hasattr(msg, "content") else msg
    if isinstance(content, list):
        return " ".join(
            part["text"] if isinstance(part, dict) else str(part)
            for part in content
            if not isinstance(part, dict) or part.get("type") == "text"
        )
    return str(content)


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    config = {"configurable": {"thread_id": request.session_id}}
    is_first_turn = not graph.get_state(config).values

    if is_first_turn:
        input_state: AgentState = {
            "messages": [{"role": "user", "content": request.message}],
            "caller_phone": request.phone or "",
            "caller_name": "",
            "session_id": request.session_id,
            "handoff": False,
            "call_summary": "",
        }
    else:
        input_state = {
            "messages": [{"role": "user", "content": request.message}],
            "caller_phone": request.phone or "",
            "session_id": request.session_id,
        }

    result = await graph.ainvoke(input_state, config=config)
    return ChatResponse(
        reply=_extract_content(result["messages"][-1]),
        handoff=result.get("handoff", False),
        session_id=request.session_id,
    )
