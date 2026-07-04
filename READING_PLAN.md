# Codebase Reading Plan

Read the files in this exact order. Each file builds on the knowledge from the previous one.

---

## Step 1 — Data Layer (What are we searching?)

| Order | File | Why Read First |
|---|---|---|
| 1 | `data/shl_catalog.csv` | Raw data. Understand columns: name, url, test_type, description. Everything else exists to search this. |
| 2 | `build_index.py` | Reads CSV, compiles into FAISS binary index. Understand how rows become Documents with page_content and metadata. |

---

## Step 2 — Retrieval Layer (How do we search?)

| Order | File | Why Read Next |
|---|---|---|
| 3 | `services/retriever.py` | Brain of the search engine. Understand CustomEnsembleRetriever, how FAISS (dense) and BM25 (sparse) merge via RRF formula: `1 / (rank + 60)`. |

---

## Step 3 — LLM Layer (Who is thinking?)

| Order | File | Why Read Next |
|---|---|---|
| 4 | `services/llm.py` | Tiny file. Initializes Gemini LLM client. Understand the model name and temperature setting. |

---

## Step 4 — Agent Brain (How does it decide?)

| Order | File | Why Read Next |
|---|---|---|
| 5 | `core/routers.py` | **Most critical file.** Contains the ROUTER_PROMPT that classifies every user message into an intent (clarify, recommend, refuse). Read the prompt word by word. |
| 6 | `core/nodes.py` | Each intent maps to a function here. Read recommend_node carefully — it calls the retriever, builds context from docs, formats Pydantic output. |
| 7 | `core/graph.py` | Wires everything into the LangGraph state machine. Flow: START → extract_and_route → (clarify / recommend / refuse) → END. |

---

## Step 5 — API Layer (How does the outside world talk to it?)

| Order | File | Why Read Next |
|---|---|---|
| 8 | `api/index.py` | FastAPI entry point. Understand Pydantic schemas (ChatRequest, ChatResponse, Recommendation) and how /chat invokes the graph. |

---

## Step 6 — Config & Deployment (How does it ship?)

| Order | File | Why Read Next |
|---|---|---|
| 9 | `requirements.txt` | Know every dependency and why it exists. |
| 10 | `vercel.json` | How Vercel maps FastAPI to a serverless function. |
| 11 | `.gitignore` | What is excluded from the repo and why. |

---

## Interview Cheat Sheet

After reading all 11 files, answer these without hesitation:

1. Walk me through what happens when a user sends a POST to /chat.
2. Why did you use Hybrid Search instead of just FAISS?
3. How does RRF work? What does the constant 60 do?
4. How does your system prevent prompt injection?
5. How did you handle Vercel's cold-start timeout?
