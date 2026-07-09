from langgraph.graph import StateGraph, END
from core.state import AgentState
from core.nodes import (
    StateExtract,
    HandleRefusal,
    RetrieveSpecific,
    HandleCompare,
    InfoSufficiency,
    AskQuestion,
    QueryForm,
    CandidateRetrieval,
    CandidateRanking,
    GroundedGen,
    GroundingVal,
)
from core.routers import intent_router, sufficiency_router, confidence_router

workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("StateExtract", StateExtract)
workflow.add_node("HandleRefusal", HandleRefusal)
workflow.add_node("RetrieveSpecific", RetrieveSpecific)
workflow.add_node("HandleCompare", HandleCompare)
workflow.add_node("InfoSufficiency", InfoSufficiency)
workflow.add_node("AskQuestion", AskQuestion)
workflow.add_node("QueryForm", QueryForm)
workflow.add_node("CandidateRetrieval", CandidateRetrieval)
workflow.add_node("CandidateRanking", CandidateRanking)
workflow.add_node("GroundedGen", GroundedGen)
workflow.add_node("GroundingVal", GroundingVal)

# Set entry point
workflow.set_entry_point("StateExtract")

# Define intent router
workflow.add_conditional_edges(
    "StateExtract",
    intent_router,
    {
        "Refuse (Out of Scope)": "HandleRefusal",
        "Compare": "RetrieveSpecific",
        "Refine": "QueryForm",
        "Clarify / Recommend": "InfoSufficiency",
    },
)

# Define branch edges
workflow.add_edge("HandleRefusal", END)
workflow.add_edge("RetrieveSpecific", "HandleCompare")
workflow.add_edge("HandleCompare", END)

# Define sufficiency router
workflow.add_conditional_edges(
    "InfoSufficiency",
    sufficiency_router,
    {
        "Budget Reached (>= 4 user turns)": "QueryForm",
        "Missing Slots": "AskQuestion",
        "Sufficient Info": "QueryForm",
    },
)

# Define ask question
workflow.add_edge("AskQuestion", END)

# Define retrieval pipeline
workflow.add_edge("QueryForm", "CandidateRetrieval")
workflow.add_edge("CandidateRetrieval", "CandidateRanking")

# Define confidence router
workflow.add_conditional_edges(
    "CandidateRanking",
    confidence_router,
    {
        "Low Confidence AND Under Budget": "AskQuestion",
        "Confidence OK OR Budget Reached": "GroundedGen",
    },
)

# Define generation pipeline
workflow.add_edge("GroundedGen", "GroundingVal")
workflow.add_edge("GroundingVal", END)

# Compile execution graph
agent_app = workflow.compile()
