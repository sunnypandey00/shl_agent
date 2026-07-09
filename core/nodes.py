import json
import logging
import re

from core.state import AgentState
from services.llm import llm
from services.retriever import get_ensemble_retriever, get_catalog_docs_by_names
from models.schemas import Recommendation
from services.prompts import (
    STATE_EXTRACT_PROMPT,
    INFO_SUFFICIENCY_PROMPT,
    QUERY_FORM_PROMPT,
    CONFIDENCE_PROMPT,
    HANDLE_REFUSAL_PROMPT,
    HANDLE_COMPARE_PROMPT,
    ASK_QUESTION_PROMPT,
    GROUNDED_GEN_PROMPT,
)

logger = logging.getLogger(__name__)

COMPETITOR_TERMS = ("hackerrank", "codility", "testgorilla", "criteriacorp", "mercer")


def latest_user_message(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if message.role == "user":
            return message.content
    return state["messages"][-1].content if state["messages"] else ""


def safe_llm_invoke(prompt: str):
    if llm is None:
        return None
    try:
        return llm.invoke(prompt)
    except Exception:
        logger.warning("LLM call failed")
        return None


def get_content(response) -> str:
    if response is None:
        return ""
    # Extract text content
    content = response.content
    if isinstance(content, list):
        # Parse list format
        return (
            content[0].get("text", "")
            if content and isinstance(content[0], dict)
            else str(content[0]) if content else ""
        )
    # Return string cast
    return str(content)


def parse_json(text: str) -> dict:
    # Parse JSON blocks
    try:
        # Strip markdown ticks
        text = text.replace("```json", "").replace("```", "").strip()
        if "{" in text and "}" in text:
            text = text[text.find("{") : text.rfind("}") + 1]
        return json.loads(text)
    except Exception:
        logger.warning("Failed to parse LLM JSON response: %r", text)
        return {}


def retrieve_candidates(query: str):
    retriever = get_ensemble_retriever()
    if retriever is None:
        logger.error("Retriever is unavailable")
        return []
    try:
        return retriever.invoke(query)
    except Exception:
        logger.exception("Retriever invocation failed")
        return []


def merge_doc_lists(*doc_lists):
    docs = []
    seen = set()
    for doc_list in doc_lists:
        for doc in doc_list:
            name = doc.metadata.get("name", "")
            if name and name not in seen:
                seen.add(name)
                docs.append(doc)
    return docs


def heuristic_extract(history: str, latest: str) -> dict:
    text = f"{history}\n{latest}".lower()
    slots = {}

    if any(term in latest.lower() for term in COMPETITOR_TERMS):
        return {"intent": "refuse", "slots": slots}

    if any(term in text for term in ("legal advice", "lawsuit", "discriminate")):
        return {"intent": "refuse", "slots": slots}

    if any(
        term in latest.lower() for term in ("compare", "difference", "versus", " vs ")
    ):
        intent = "compare"
    elif any(
        term in latest.lower()
        for term in ("drop", "replace", "instead", "not ", "change")
    ):
        intent = "refine"
    else:
        intent = "evaluate"

    role_patterns = [
        r"(?:hiring|hire|for|need)\s+(?:a|an|the)?\s*([a-z0-9+#.\- ]+?)(?:\.|\?|,| with | under | who |$)",
        r"solution for\s+([a-z0-9+#.\- ]+?)(?:\.|\?|,|$)",
    ]
    for pattern in role_patterns:
        match = re.search(pattern, text)
        if match:
            role = match.group(1).strip()
            if role and role not in ("solution", "assessment", "test"):
                slots["role"] = role
                break

    skills = []
    skill_terms = (
        "rust",
        "python",
        "java",
        "javascript",
        "linux",
        "networking",
        ".net",
        "c++",
        "sql",
        "sales",
        "call center",
        "leadership",
        "cognitive",
    )
    for term in skill_terms:
        if term in text:
            skills.append(term)
    if skills:
        slots["skills"] = skills

    if any(
        term in text for term in ("senior", "director", "cxo", "executive", "15 years")
    ):
        slots["seniority"] = "senior"
    if "entry" in text or "graduate" in text:
        slots["seniority"] = "entry"

    languages = []
    for language in ("english", "spanish", "french", "german", "dutch", "portuguese"):
        if language in text:
            languages.append(language)
    if languages:
        slots["languages"] = languages

    duration_match = re.search(
        r"(?:under|within|less than|max(?:imum)?)\s+(\d+\s*minutes?)", text
    )
    if duration_match:
        slots["duration"] = duration_match.group(1)
    if "remote" in text:
        slots["remote"] = "yes"
    if "adaptive" in text:
        slots["adaptive"] = "yes"

    return {"intent": intent, "slots": slots}


def deterministic_candidates(query: str, state: AgentState):
    text = " ".join([m.content for m in state["messages"]]).lower()
    wanted = []

    if any(
        term in text
        for term in (
            "senior leadership",
            "cxo",
            "director-level",
            "leadership benchmark",
        )
    ):
        wanted.extend(
            [
                "Occupational Personality Questionnaire OPQ32r",
                "OPQ Universal Competency Report 2.0",
                "OPQ Leadership Report",
            ]
        )

    if "rust" in text or ("networking" in text and "engineer" in text):
        wanted.extend(
            [
                "Smart Interview Live Coding",
                "Linux Programming (General)",
                "Networking and Implementation (New)",
            ]
        )

    if any(term in text for term in ("cognitive", "reasoning", "aptitude", "g+")):
        wanted.append("SHL Verify Interactive G+")

    if any(term in text for term in ("personality", "opq", "senior ic")):
        wanted.append("Occupational Personality Questionnaire OPQ32r")

    if "senior" in text and "engineer" in text:
        wanted.append("Occupational Personality Questionnaire OPQ32r")

    if "call center" in text or "contact center" in text:
        wanted.extend(
            [
                "Contact Center Call Simulation (New)",
                "Customer Service Phone Simulation",
                "WriteX - Email Writing (Customer Service) (New)",
            ]
        )

    return get_catalog_docs_by_names(wanted)


def rerank_candidates(query: str, docs, state: AgentState):
    pinned = deterministic_candidates(query, state)
    return merge_doc_lists(pinned, docs)[:20]


def filter_by_slots(docs, slots):
    if not slots or not docs:
        return docs[:10]

    filtered = list(docs)

    # Filter remote constraint
    remote_val = slots.get("remote")
    if remote_val and str(remote_val).lower() == "yes":
        filtered = [d for d in filtered if d.metadata.get("remote", "").lower() == "yes"]

    # Filter adaptive constraint
    adaptive_val = slots.get("adaptive")
    if adaptive_val and str(adaptive_val).lower() == "yes":
        filtered = [d for d in filtered if d.metadata.get("adaptive", "").lower() == "yes"]

    # Filter language constraint
    req_langs = slots.get("languages")
    if req_langs and isinstance(req_langs, list):
        req_set = {l.lower() for l in req_langs}
        def has_language(doc):
            doc_langs = doc.metadata.get("languages", "").lower()
            return any(lang in doc_langs for lang in req_set)
        filtered = [d for d in filtered if has_language(d)]

    # Filter duration constraint
    dur = slots.get("duration", "")
    if dur:
        import re as _re
        m = _re.search(r"(\d+)", str(dur))
        if m:
            max_min = int(m.group(1))
            def within_budget(doc):
                d_dur = doc.metadata.get("duration", "")
                dm = _re.search(r"(\d+)", str(d_dur))
                if not dm:
                    return True
                return int(dm.group(1)) <= max_min
            filtered = [d for d in filtered if within_budget(d)]

    # Graceful fallback
    if not filtered:
        return docs[:10]
    return filtered[:10]


def recommendations_from_docs(docs):
    recs = []
    seen = set()
    for doc in docs[:10]:
        name = doc.metadata.get("name", "")
        url = doc.metadata.get("url", "")
        if not name or not url or name in seen:
            continue
        seen.add(name)
        recs.append(
            Recommendation(
                name=name, url=url, test_type=doc.metadata.get("test_type", "")
            )
        )
    return recs


def StateExtract(state: AgentState) -> AgentState:
    # Format chat history
    history = "\n".join([f"{m.role}: {m.content}" for m in state["messages"]])
    # Invoke extraction LLM
    prompt = STATE_EXTRACT_PROMPT.format(history=history)
    heuristic = heuristic_extract(history, latest_user_message(state))
    response = safe_llm_invoke(prompt)
    # Parse extracted JSON
    data = parse_json(get_content(response))
    # Update intent value
    intent = data.get("intent") or heuristic["intent"]
    # Extract new slots
    slots = heuristic["slots"]
    slots.update(data.get("slots", {}) or {})
    # Return updated state
    return {"intent": intent, "slots": slots}


def HandleRefusal(state: AgentState) -> AgentState:
    # Format refusal prompt
    msg = latest_user_message(state)
    prompt = HANDLE_REFUSAL_PROMPT.format(message=msg)
    # Invoke generation LLM
    response = safe_llm_invoke(prompt)
    reply = get_content(response).strip()
    if not reply:
        reply = "I can only help recommend SHL assessments from the SHL catalog, so I can't provide competitor tests or general hiring advice."
    # Return refusal state
    return {
        "reply": reply,
        "recommendations": [],
        "end_of_conversation": False,
    }


def RetrieveSpecific(state: AgentState) -> AgentState:
    # Extract last message
    msg = latest_user_message(state)
    # Execute ensemble search
    docs = rerank_candidates(msg, retrieve_candidates(msg), state)
    # Return retrieved candidates
    return {"candidates": docs}


def HandleCompare(state: AgentState) -> AgentState:
    # Extract last message
    msg = latest_user_message(state)
    # Format candidate context
    context = "\n".join(
        [
            f"{d.metadata['name']} - {d.page_content}"
            for d in state.get("candidates", [])[:3]
        ]
    )
    # Invoke generation LLM
    prompt = HANDLE_COMPARE_PROMPT.format(message=msg, candidates=context)
    response = safe_llm_invoke(prompt)
    reply = get_content(response).strip()
    if not reply:
        reply = "These SHL assessments differ in the job behavior, skills, and aptitude evidence they provide. Use the listed catalog items only as the comparison set."
    # Return comparison state
    return {
        "reply": reply,
        "recommendations": [],
        "end_of_conversation": False,
    }


def InfoSufficiency(state: AgentState) -> AgentState:
    # Format sufficiency prompt
    slots = json.dumps(state.get("slots", {}))
    prompt = INFO_SUFFICIENCY_PROMPT.format(slots=slots)
    # Invoke extraction LLM
    response = safe_llm_invoke(prompt)
    # Parse boolean result
    data = parse_json(get_content(response))
    heuristic_slots = state.get("slots", {})
    missing = data.get("missing_slots")
    if missing is None:
        missing = not (heuristic_slots.get("role") or heuristic_slots.get("skills"))
    # Return sufficiency state
    return {"missing_slots": missing}


def AskQuestion(state: AgentState) -> AgentState:
    # Format chat history
    history = "\n".join([f"{m.role}: {m.content}" for m in state["messages"]])
    slots = json.dumps(state.get("slots", {}))
    # Invoke generation LLM
    prompt = ASK_QUESTION_PROMPT.format(history=history, slots=slots)
    response = safe_llm_invoke(prompt)
    reply = get_content(response).strip()
    if not reply:
        reply = "What role, seniority level, and key skills do you need to assess?"
    # Return question state
    return {
        "reply": reply,
        "recommendations": [],
        "end_of_conversation": False,
    }


def QueryForm(state: AgentState) -> AgentState:
    # Format query prompt
    slots = json.dumps(state.get("slots", {}))
    prompt = QUERY_FORM_PROMPT.format(slots=slots)
    # Invoke generation LLM
    response = safe_llm_invoke(prompt)
    # Return formatted query
    query = get_content(response).strip()
    if not query:
        query = " ".join(
            str(value) for value in state.get("slots", {}).values() if value
        )
    return {"query": query}


def CandidateRetrieval(state: AgentState) -> AgentState:
    # Execute ensemble search
    query = state.get("query", "")
    docs = rerank_candidates(query, retrieve_candidates(query), state)
    # Apply slot filtering
    docs = filter_by_slots(docs, state.get("slots", {}))
    # Return retrieved candidates
    return {"candidates": docs}


def CandidateRanking(state: AgentState) -> AgentState:
    # Retrieve candidate docs
    docs = state.get("candidates", [])
    # Check empty candidates
    if not docs:
        return {"confidence_ok": False}
    # Format confidence prompt
    slots = json.dumps(state.get("slots", {}))
    context = "\n".join([f"{d.metadata['name']} - {d.page_content}" for d in docs[:5]])
    # Invoke extraction LLM
    prompt = CONFIDENCE_PROMPT.format(slots=slots, candidates=context)
    response = safe_llm_invoke(prompt)
    # Parse boolean result
    data = parse_json(get_content(response))
    confidence = data.get("confidence_ok")
    if confidence is None:
        confidence = bool(deterministic_candidates("", state) or docs)
    # Return ranking state
    return {"confidence_ok": confidence}


def GroundedGen(state: AgentState) -> AgentState:
    # Retrieve candidate docs
    docs = state.get("candidates", [])
    # Check empty candidates
    if not docs:
        # Return continuation prompt
        return {
            "reply": "I couldn't find matching assessments. Could you broaden your criteria?",
            "recommendations": [],
            "end_of_conversation": False,
        }
    # Format generation prompt
    slots = json.dumps(state.get("slots", {}))
    context = "\n".join([f"{d.metadata['name']} - {d.page_content}" for d in docs[:10]])
    # Invoke generation LLM
    prompt = GROUNDED_GEN_PROMPT.format(slots=slots, candidates=context)
    response = safe_llm_invoke(prompt)
    reply = get_content(response).strip()
    if not reply:
        names = ", ".join(d.metadata["name"] for d in docs[:5])
        reply = f"Based on the SHL catalog, the best shortlist is: {names}."
    # Return generated state
    return {
        "reply": reply,
        "recommendations": recommendations_from_docs(docs),
        "end_of_conversation": True,
    }


def GroundingVal(state: AgentState) -> AgentState:
    existing_recs = state.get("recommendations")
    if existing_recs:
        return {"recommendations": existing_recs[:10]}

    # Extract generation reply
    reply = state.get("reply", "").lower()
    # Retrieve candidate docs
    docs = state.get("candidates", [])
    recs = []
    # Iterate over candidates
    for d in docs[:10]:
        # Clean string matching
        name_lower = d.metadata["name"].lower()
        # Test exact match
        if name_lower in reply:
            # Append valid recommendation
            recs.append(
                Recommendation(
                    name=d.metadata["name"],
                    url=d.metadata["url"],
                    test_type=d.metadata["test_type"],
                )
            )
        else:
            # Build normalized strings
            clean_name = name_lower.replace("-", " ").replace("  ", " ").strip()
            clean_reply = reply.replace("-", " ").replace("  ", " ")
            # Test normalized match
            if clean_name in clean_reply:
                # Append valid recommendation
                recs.append(
                    Recommendation(
                        name=d.metadata["name"],
                        url=d.metadata["url"],
                        test_type=d.metadata["test_type"],
                    )
                )
    # Return valid recommendations
    return {"recommendations": recs}
