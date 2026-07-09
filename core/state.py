from typing import TypedDict, List, Optional, Any, Annotated
from models.schemas import Message, Recommendation


def merge_slots(existing: dict, new: dict) -> dict:
    if not existing:
        return new
    if not new:
        return existing

    merged = existing.copy()

    for key in ("role", "seniority", "duration", "remote", "adaptive"):
        if new.get(key):
            merged[key] = new[key]

    if new.get("skills"):
        old_skills = existing.get("skills", [])
        merged["skills"] = list(dict.fromkeys(old_skills + new["skills"]))

    if new.get("languages"):
        old_languages = existing.get("languages", [])
        merged["languages"] = list(dict.fromkeys(old_languages + new["languages"]))

    return merged


class SlotState(TypedDict, total=False):
    role: Optional[str]
    seniority: Optional[str]
    skills: List[str]
    languages: List[str]
    duration: Optional[str]
    remote: Optional[str]
    adaptive: Optional[str]


class AgentState(TypedDict):
    # Track conversation history
    messages: List[Message]
    # Track extracted slots
    slots: Annotated[SlotState, merge_slots]
    # Store current intent
    intent: Optional[str]
    # Store search query
    query: str
    # Track retrieved documents
    candidates: List[Any]
    # Track generated reply
    reply: str
    # Store final recommendations
    recommendations: Optional[List[Recommendation]]
    # Flag conversation end
    end_of_conversation: bool
    # Track message turns
    turn_count: int
    # Flag search confidence
    confidence_ok: bool
    # Flag missing slots
    missing_slots: bool
