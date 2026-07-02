from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from models.schemas import ChatRequest, ChatResponse
from core.graph import agent_app

app = FastAPI(title="SHL Conversational Assessment Recommender")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    # Determine the turn count based on the number of messages
    turn_count = len(request.messages)
    
    # Initialize the state for LangGraph
    initial_state = {
        "messages": request.messages,
        "intent": None,
        "constraints": "",
        "recommendations": None,
        "reply": "",
        "end_of_conversation": False,
        "turn_count": turn_count
    }
    
    # Execute the graph
    final_state = agent_app.invoke(initial_state)
    
    # Schema Validator Boundary
    # Ensure we return an empty array if recommendations is None
    recs = final_state.get("recommendations", [])
    if recs is None:
        recs = []
        
    return ChatResponse(
        reply=final_state.get("reply", "I am having trouble processing that right now."),
        recommendations=recs,
        end_of_conversation=final_state.get("end_of_conversation", False)
    )
