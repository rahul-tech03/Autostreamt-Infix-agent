from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Central state object passed across LangGraph nodes.
    """

    # Conversation memory (last N messages)
    messages: List[BaseMessage]

    # Detected intent of the latest user message
    intent: Optional[str]

    # Lead qualification fields
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]

    # Tool execution guard
    lead_captured: bool


def get_initial_state() -> AgentState:
    return {
        "messages": [],
        "intent": None,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
    }

