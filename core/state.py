from typing import TypedDict, List, Optional
from models.schemas import Message, Recommendation
import operator
from typing import Annotated

class AgentState(TypedDict):
    messages: List[Message]
    intent: Optional[str]
    constraints: str
    recommendations: Optional[List[Recommendation]]
    reply: str
    end_of_conversation: bool
    turn_count: int
