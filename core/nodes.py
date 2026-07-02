from core.state import AgentState
from services.llm import llm
from services.prompts import ROUTER_SYSTEM_PROMPT, CLARIFY_PROMPT, REFUSE_PROMPT, GENERATE_RECOMMENDATIONS_PROMPT, COMPARE_PROMPT
from services.retriever import ensemble_retriever
from models.schemas import Recommendation

def get_content(response):
    content = response.content
    if isinstance(content, list):
        return content[0].get("text", "") if content and isinstance(content[0], dict) else str(content[0]) if content else ""
    return str(content)

def extract_and_route(state: AgentState) -> AgentState:
    history = "\n".join([f"{m.role}: {m.content}" for m in state["messages"]])
    prompt = f"{ROUTER_SYSTEM_PROMPT}\n\nHistory:\n{history}"
    response = llm.invoke(prompt)
    intent = get_content(response).strip().lower()
    
    # Extract constraints
    last_msg = state["messages"][-1].content
    constraints = state.get("constraints", "") + " " + last_msg
    
    return {"intent": intent, "constraints": constraints.strip()}

def handle_refusal(state: AgentState) -> AgentState:
    response = llm.invoke(f"{REFUSE_PROMPT}\n\nUser: {state['messages'][-1].content}")
    return {"reply": get_content(response), "recommendations": [], "end_of_conversation": False}

def ask_question(state: AgentState) -> AgentState:
    history = "\n".join([f"{m.role}: {m.content}" for m in state["messages"]])
    response = llm.invoke(f"{CLARIFY_PROMPT}\n\nHistory:\n{history}")
    return {"reply": get_content(response), "recommendations": [], "end_of_conversation": False}

def retrieve_and_generate(state: AgentState) -> AgentState:
    # Retrieve document candidates
    docs = ensemble_retriever.invoke(state["constraints"])
    if not docs:
        return {"reply": "I couldn't find any assessments matching those constraints.", "recommendations": [], "end_of_conversation": False}
        
    # Generate grounded response
    context = "\n".join([f"{d.metadata['name']} - {d.page_content}" for d in docs[:5]])
    prompt = f"{GENERATE_RECOMMENDATIONS_PROMPT}\n\nConstraints: {state['constraints']}\n\nCatalog Items:\n{context}"
    response = llm.invoke(prompt)
    
    # Format the output
    recs = []
    for d in docs[:5]:
        recs.append(Recommendation(
            name=d.metadata["name"],
            url=d.metadata["url"],
            test_type=d.metadata["test_type"]
        ))
        
    return {"reply": get_content(response), "recommendations": recs, "end_of_conversation": True}

def compare_items(state: AgentState) -> AgentState:
    docs = ensemble_retriever.invoke(state["messages"][-1].content)
    context = "\n".join([f"{d.metadata['name']} - {d.page_content}" for d in docs[:3]])
    prompt = f"{COMPARE_PROMPT}\n\nQuery: {state['messages'][-1].content}\n\nCatalog Items:\n{context}"
    response = llm.invoke(prompt)
    
    return {"reply": get_content(response), "recommendations": [], "end_of_conversation": False}
