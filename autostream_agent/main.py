from langchain_core.messages import HumanMessage
from agent.state import get_initial_state
from agent.graph import build_graph


def assign_lead_field(user_input: str, state: dict):
    """
    Assign user input ONLY to the field currently being requested.
    No guessing, no extraction.
    """

    if state["intent"] != "HIGH_INTENT" or state["lead_captured"]:
        return

    if state["lead_name"] is None:
        state["lead_name"] = user_input.strip()
        return

    if state["lead_email"] is None:
        state["lead_email"] = user_input.strip()
        return

    if state["lead_platform"] is None:
        state["lead_platform"] = user_input.strip()
        return



def main():
    print("AutoStream AI Agent (type 'exit' to quit)\n")

    state = get_initial_state()
    graph = build_graph()

    while True:
        user_input = input("You: ")

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        state["messages"].append(HumanMessage(content=user_input))

        assign_lead_field(user_input, state)

        state = graph.invoke(state)

        last_message = state["messages"][-1]
        print(f"Agent: {last_message.content}\n")


if __name__ == "__main__":
    main()
