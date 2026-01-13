from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from agent.state import AgentState
from agent.intent_classifier import IntentClassifier
from agent.rag import AutoStreamRAG
from agent.tools import mock_lead_capture


# Initialize components
intent_classifier = IntentClassifier()
rag = AutoStreamRAG()

def is_lead_in_progress(state: AgentState) -> bool:
    return (
        state["intent"] == "HIGH_INTENT"
        and not state["lead_captured"]
    )


#Node: Intent Classification
def classify_intent(state: AgentState) -> AgentState:
    if is_lead_in_progress(state):
        return state

    last_message = state["messages"][-1].content
    intent = intent_classifier.classify(last_message)
    state["intent"] = intent

    return state



#Node: Greeting Handler
def handle_greeting(state: AgentState) -> AgentState:
    response = (
        "Hi! I’m AutoStream’s assistant. "
        "I can help with pricing, features, or getting you started."
    )

    state["messages"].append(AIMessage(content=response))
    return state


#Node: Info Handler (RAG)
def handle_info(state: AgentState) -> AgentState:
    user_question = state["messages"][-1].content

    answer = rag.answer(user_question)

    state["messages"].append(AIMessage(content=answer))
    return state


#Node: High-Intent Lead Handling
def handle_high_intent(state: AgentState) -> AgentState:
    if not state["lead_name"]:
        response = "Great! May I have your name?"
    elif not state["lead_email"]:
        response = "Thanks! Could you share your email address?"
    elif not state["lead_platform"]:
        response = "Which platform do you create content on? (YouTube, Instagram, etc.)"
    else:
        return state

    state["messages"].append(AIMessage(content=response))
    return state


#Node: Tool Execution (Guarded)
def capture_lead(state: AgentState) -> AgentState:
    if not state["lead_captured"]:
        mock_lead_capture(
            state["lead_name"],
            state["lead_email"],
            state["lead_platform"]
        )
        state["lead_captured"] = True

        state["messages"].append(
            AIMessage(content="You’re all set! Your lead has been captured successfully.")
        )

    return state



# Conditional Routing Logic
def route_by_intent(state: AgentState):
    intent = state["intent"]

    if intent == "GREETING":
        return "handle_greeting"
    elif intent == "INFO":
        return "handle_info"
    elif intent == "HIGH_INTENT":
        return "handle_high_intent"
    else:
        return "handle_info"



# Build the Langgraph
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("handle_greeting", handle_greeting)
    graph.add_node("handle_info", handle_info)
    graph.add_node("handle_high_intent", handle_high_intent)
    graph.add_node("capture_lead", capture_lead)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "handle_greeting": "handle_greeting",
            "handle_info": "handle_info",
            "handle_high_intent": "handle_high_intent",
        },
    )

    graph.add_edge("handle_greeting", END)
    graph.add_edge("handle_info", END)

    # High-intent flow
    graph.add_conditional_edges(
        "handle_high_intent",
        lambda state: "capture_lead"
        if state["lead_name"]
        and state["lead_email"]
        and state["lead_platform"]
        else END,
        {
            "capture_lead": "capture_lead",
            END: END,
        },
    )

    graph.add_edge("capture_lead", END)

    return graph.compile()




