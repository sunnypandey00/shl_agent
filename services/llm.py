import logging
import os

from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)


def get_llm():
    # Configure Gemini LLM
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        logger.warning(
            "Gemini API key is not set; deterministic fallbacks will be used"
        )
        return None

    try:
        return ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite", temperature=0.0, max_retries=3, timeout=15
        )
    except Exception:
        logger.warning(
            "Gemini LLM is unavailable; deterministic fallbacks will be used"
        )
        return None


llm = get_llm()
