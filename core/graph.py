from langgraph.graph import StateGraph, END
from core.state import AgentState
from core.nodes import extract_and_route, handle_refusal, ask_question, retrieve_and_generate, compare_items
from core.routers import intent_router

workflow = StateGraph(AgentState)

# Add graph nodes
workflow.add_node("extract_and_route", extract_and_route)
workflow.add_node("handle_refusal", handle_refusal)
workflow.add_node("ask_question", ask_question)
workflow.add_node("retrieve_and_generate", retrieve_and_generate)
workflow.add_node("compare_items", compare_items)

# Add graph edges
workflow.set_entry_point("extract_and_route")

workflow.add_conditional_edges(
    "extract_and_route",
    intent_router,
    {
        "refuse": "handle_refusal",
        "clarify": "ask_question",
        "recommend": "retrieve_and_generate",
        "compare": "compare_items"
    }
)

# Route to end
workflow.add_edge("handle_refusal", END)
workflow.add_edge("ask_question", END)
workflow.add_edge("retrieve_and_generate", END)
workflow.add_edge("compare_items", END)

# Compile the graph
agent_app = workflow.compile()
