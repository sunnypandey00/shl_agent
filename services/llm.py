import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm():
    # Configure Gemini LLM
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.0,
        max_retries=3,
        timeout=25
    )

llm = get_llm()
