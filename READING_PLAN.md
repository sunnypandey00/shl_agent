# Codebase Reading Plan

Read the files in this order. Each file builds on the previous one.

---

## Step 1 - Data Layer

| Order | File | Why Read First |
|---|---|---|
| 1 | `data/shl_catalog.json` | Raw SHL catalog data: names, URLs, descriptions, categories, job levels, duration, languages, remote, and adaptive fields. |
| 2 | `build_index.py` | Rebuilds the FAISS index from the JSON catalog and shows which fields are embedded. |

---

## Step 2 - Retrieval Layer

| Order | File | Why Read Next |
|---|---|---|
| 3 | `services/retriever.py` | Builds runtime BM25 documents, loads FAISS, and fuses both rankings with reciprocal rank fusion: `1 / (rank + 60)`. |

---

## Step 3 - LLM Layer

| Order | File | Why Read Next |
|---|---|---|
| 4 | `services/llm.py` | Initializes the Gemini chat model used by every graph node. |
| 5 | `services/prompts.py` | Defines the extraction, sufficiency, query, confidence, refusal, comparison, clarification, and grounded generation prompts. |

---

## Step 4 - Agent Flow

| Order | File | Why Read Next |
|---|---|---|
| 6 | `core/state.py` | Defines graph state and how extracted slots merge across turns. |
| 7 | `core/routers.py` | Contains deterministic routing decisions after node outputs are written into state. |
| 8 | `core/nodes.py` | Contains the actual graph actions: extraction, retrieval, ranking, response generation, and recommendation shaping. |
| 9 | `core/graph.py` | Wires the LangGraph nodes and conditional edges together. |

---

## Step 5 - API Layer

| Order | File | Why Read Next |
|---|---|---|
| 10 | `models/schemas.py` | Defines request and response schemas. |
| 11 | `api/index.py` | FastAPI entry point for `/health` and `/chat`; builds initial graph state and returns schema-compliant responses. |

---

## Step 6 - Evaluation and Deployment

| Order | File | Why Read Next |
|---|---|---|
| 12 | `strict_eval_harness.py` | Replays every markdown conversation trace and validates response schema and turn budget. |
| 13 | `run_harness.py` | Convenience harness for manual local conversation replay. |
| 14 | `requirements.txt` | Runtime dependencies for deployment. |
| 15 | `vercel.json` | Vercel serverless routing for the FastAPI app. |
| 16 | `.gitignore` | Files intentionally excluded from version control. |

---

## Interview Cheat Sheet

After reading these files, be ready to answer:

1. What happens when a user sends `POST /chat`?
2. Why does the app combine BM25 and FAISS instead of using only one retriever?
3. How does reciprocal rank fusion work, and why is the constant 60 used?
4. Which constraints are extracted from conversation history and used for retrieval?
5. How does the API stay schema-compliant when a provider call or retrieval step fails?
6. What does the app do to reduce, but not eliminate, prompt-injection risk?
