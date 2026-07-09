import json
import logging
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CustomEnsembleRetriever:
    def __init__(self, bm25, faiss):
        self.bm25 = bm25
        self.faiss = faiss

    def invoke(self, query):
        if not query or not str(query).strip():
            return []

        fused = {}
        # Loop both retrievers
        for i, retriever in enumerate(
            [
                retriever
                for retriever in [self.bm25, self.faiss]
                if retriever is not None
            ]
        ):
            try:
                retrieved_docs = retriever.invoke(query)
            except Exception:
                logger.warning(
                    "Retriever backend failed; disabling it for this process"
                )
                if retriever is self.faiss:
                    self.faiss = None
                elif retriever is self.bm25:
                    self.bm25 = None
                continue

            for rank, doc in enumerate(retrieved_docs):
                name = doc.metadata["name"]
                if name not in fused:
                    fused[name] = {"score": 0.0, "doc": doc}
                # Prefer FAISS metadata
                if i == 1:
                    fused[name]["doc"] = doc

                fused[name]["score"] += 1.0 / (rank + 60)

        sorted_docs = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs[:20]]


def map_test_type(item):
    type_map = {
        "Personality & Behavior": "P",
        "Ability & Aptitude": "A",
        "Knowledge & Skills": "K",
        "Biodata & Situational Judgment": "B",
        "Simulations": "S",
        "Assessment Exercises": "E",
        "Development & 360": "D",
        "Competencies": "C",
    }

    unique = list(
        dict.fromkeys(type_map[k] for k in item.get("keys", []) if k in type_map)
    )
    return ",".join(unique)


def load_catalog_documents():
    try:
        with open(
            PROJECT_ROOT / "data" / "shl_catalog.json", "r", encoding="utf-8"
        ) as f:
            catalog = json.load(f, strict=False)
    except Exception:
        logger.exception("Error loading catalog")
        return []

    documents = []
    for item in catalog:
        name = item.get("name", "")
        keys = item.get("keys", [])
        description = (
            item.get("description") or f"{name}. Categories: {', '.join(keys)}"
        )

        parts = [
            f"Assessment: {name}",
            f"Description: {description}",
            f"Categories: {', '.join(keys)}",
        ]
        if item.get("job_levels"):
            parts.append(f"Job Levels: {', '.join(item['job_levels'])}")
        if item.get("duration"):
            parts.append(f"Duration: {item['duration']}")
        if item.get("languages"):
            parts.append(f"Languages: {', '.join(item['languages'])}")
        if item.get("remote"):
            parts.append(f"Remote: {item['remote']}")
        if item.get("adaptive"):
            parts.append(f"Adaptive: {item['adaptive']}")

        page_content = "\n".join(parts)
        metadata = {
            "name": name,
            "url": item.get("link", ""),
            "test_type": map_test_type(item),
            "adaptive": item.get("adaptive", ""),
            "remote": item.get("remote", ""),
            "duration": item.get("duration", ""),
            "job_levels": ",".join(item.get("job_levels", [])),
            "languages": ",".join(item.get("languages", [])),
            "keys": ",".join(keys),
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


_CATALOG_DOCUMENTS = None
_CATALOG_BY_NAME = None
_ensemble_retriever = None


def get_catalog_documents():
    global _CATALOG_DOCUMENTS
    if _CATALOG_DOCUMENTS is None:
        # Load catalog docs
        _CATALOG_DOCUMENTS = load_catalog_documents()
    return _CATALOG_DOCUMENTS


def get_catalog_by_name():
    global _CATALOG_BY_NAME
    if _CATALOG_BY_NAME is None:
        # Map by name
        _CATALOG_BY_NAME = {
            doc.metadata["name"].lower(): doc for doc in get_catalog_documents()
        }
    return _CATALOG_BY_NAME


def get_catalog_docs_by_names(names):
    docs = []
    seen = set()
    catalog_by_name = get_catalog_by_name()
    catalog_docs = get_catalog_documents()
    for wanted in names:
        wanted_lower = wanted.lower()
        doc = catalog_by_name.get(wanted_lower)
        if doc is None:
            doc = next(
                (
                    candidate
                    for candidate in catalog_docs
                    if wanted_lower in candidate.metadata["name"].lower()
                ),
                None,
            )
        if doc is not None and doc.metadata["name"] not in seen:
            seen.add(doc.metadata["name"])
            docs.append(doc)
    return docs


def initialize_retriever():
    documents = get_catalog_documents()
    if not documents:
        return None

    # Load BM25 index first so the service can still search without embedding credentials.
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        logger.warning("Gemini API key is not set; using BM25-only retrieval")
        return CustomEnsembleRetriever(bm25_retriever, None)

    try:
        # Initialize Gemini embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2", max_retries=3
        )

        # Load FAISS index
        faiss_vectorstore = FAISS.load_local(
            str(PROJECT_ROOT / "data" / "faiss_index"),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 10})

        # Create hybrid retriever
        return CustomEnsembleRetriever(bm25_retriever, faiss_retriever)
    except Exception:
        logger.warning("FAISS retriever is unavailable; using BM25-only retrieval")
        return CustomEnsembleRetriever(bm25_retriever, None)


def get_ensemble_retriever():
    global _ensemble_retriever
    if _ensemble_retriever is None:
        # Run initialization pipeline
        _ensemble_retriever = initialize_retriever()
    return _ensemble_retriever
