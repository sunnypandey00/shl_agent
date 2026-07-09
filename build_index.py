import json
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent

# Type code mapping
TYPE_MAP = {
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Knowledge & Skills": "K",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Assessment Exercises": "E",
    "Development & 360": "D",
    "Competencies": "C",
}


def build_index():
    print("Loading catalog...")
    with open(PROJECT_ROOT / "data" / "shl_catalog.json", "r", encoding="utf-8") as f:
        catalog = json.load(f, strict=False)

    documents = []
    for item in catalog:
        name = item.get("name", "")
        keys = item.get("keys", [])
        desc = item.get("description") or f"{name}. Categories: {', '.join(keys)}"

        # Build rich metadata
        meta = {
            "name": name,
            "test_type": ",".join(TYPE_MAP[k] for k in keys if k in TYPE_MAP),
            "url": item.get("link", ""),
            "adaptive": item.get("adaptive", ""),
            "remote": item.get("remote", ""),
            "duration": item.get("duration", ""),
            "job_levels": ",".join(item.get("job_levels", [])),
            "languages": ",".join(item.get("languages", [])),
            "keys": ",".join(keys),
        }

        parts = [
            f"Assessment: {name}",
            f"Description: {desc}",
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

        doc = Document(page_content="\n".join(parts), metadata=meta)
        documents.append(doc)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", max_retries=5)

    batch_size = 50
    vectorstore = None

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        print(f"Embedding batch {i} to {i+len(batch)} of {len(documents)}...")

        success = False
        for attempt in range(5):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
                success = True
                break
            except Exception as e:
                print(f"Rate limit hit. Retrying in 15 seconds... {e}")
                time.sleep(15)

        if not success:
            print("Failed completely.")
            return

        time.sleep(8)

    vectorstore.save_local(str(PROJECT_ROOT / "data" / "faiss_index"))
    print("Index built successfully.")


if __name__ == "__main__":
    build_index()
