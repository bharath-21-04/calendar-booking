from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agent.graph import graph
from agent.state import AgentState

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique session identifier. Generate on client if starting a new session.",
    )
    message: str = Field(..., description="The user's chat message.")
    phone: Optional[str] = Field(
        None,
        description="Caller phone in E.164 format, e.g. +15551234567. Optional for text chat.",
    )


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Sol's response to the user.")
    handoff: bool = Field(
        False, description="True if this session has been escalated to a human."
    )
    session_id: str = Field(..., description="Echo of the session_id from the request.")


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    input_state: AgentState = {
        "messages": [{"role": "user", "content": request.message}],
        "caller_phone": request.phone or "",
        "caller_name": "",
        "session_id": request.session_id,
        "handoff": False,
        "call_summary": "",
    }

    config = {"configurable": {"thread_id": request.session_id}}

    result = await graph.ainvoke(input_state, config=config)

    last_msg = result["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else last_msg
    if isinstance(content, list):
        reply = " ".join(
            part["text"] if isinstance(part, dict) else str(part)
            for part in content
            if not isinstance(part, dict) or part.get("type") == "text"
        )
    else:
        reply = str(content)

    return ChatResponse(
        reply=reply,
        handoff=result.get("handoff", False),
        session_id=request.session_id,
    )
