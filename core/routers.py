from core.state import AgentState

def intent_router(state: AgentState) -> str:
    intent = state.get("intent", "clarify")
    
    if intent == "refuse":
        return "refuse"
    elif intent == "compare":
        return "compare"
    elif intent == "clarify":
        # Turn limit guardrail
        if state.get("turn_count", 0) >= 6:
            return "recommend"
        return "clarify"
    else: 
        # Handle other intents
        return "recommend"
