from dotenv import load_dotenv

load_dotenv()

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.schemas import ChatRequest, ChatResponse
from core.graph import agent_app

app = FastAPI(title="SHL Conversational Assessment Recommender")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger = logging.getLogger(__name__)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    # Count user turns. The public budget is 8 total turns, i.e. 4 user turns.
    turn_count = sum(1 for m in request.messages if m.role == "user")

    # Initialize Graph state
    initial_state = {
        "messages": request.messages,
        "intent": None,
        "slots": {},
        "query": "",
        "candidates": [],
        "recommendations": None,
        "reply": "",
        "end_of_conversation": False,
        "turn_count": turn_count,
        "confidence_ok": False,
        "missing_slots": True,
    }

    try:
        # Run the graph
        final_state = agent_app.invoke(initial_state)
    except Exception:
        logger.exception("Failed to process chat request")
        return ChatResponse(
            reply="I am having trouble processing that right now. Please try again with the role and skills you want to assess.",
            recommendations=[],
            end_of_conversation=False,
        )

    # Schema validation boundary
    # Default empty recommendations
    recs = final_state.get("recommendations", [])
    if recs is None:
        recs = []

    return ChatResponse(
        reply=final_state.get(
            "reply", "I am having trouble processing that right now."
        ),
        recommendations=recs,
        end_of_conversation=final_state.get("end_of_conversation", False),
    )
