from core.state import AgentState


def intent_router(state: AgentState) -> str:
    # Get active intent
    intent = state.get("intent", "evaluate")
    # Route refusal intent
    if intent == "refuse":
        return "Refuse (Out of Scope)"
    # Route comparison intent
    if intent == "compare":
        return "Compare"
    # Route refinement intent
    if intent == "refine":
        return "Refine"
    # Route evaluation intent
    return "Clarify / Recommend"


def sufficiency_router(state: AgentState) -> str:
    # Check user-turn budget. Four user turns plus four assistant turns equals the 8-turn cap.
    if state.get("turn_count", 0) >= 4:
        return "Budget Reached (>= 4 user turns)"
    # Check missing slots
    if state.get("missing_slots", False):
        return "Missing Slots"
    # Return sufficient state
    return "Sufficient Info"


def confidence_router(state: AgentState) -> str:
    # Check budget overflow
    budget_reached = state.get("turn_count", 0) >= 4
    # Check search confidence
    high_confidence = state.get("confidence_ok", False)
    # Evaluate confidence gate
    if budget_reached or high_confidence:
        return "Confidence OK OR Budget Reached"
    # Request more info
    return "Low Confidence AND Under Budget"
