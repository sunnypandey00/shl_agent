import os
import json
import time
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

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
    with open("data/shl_catalog.json", "r", encoding="utf-8") as f:
        catalog = json.load(f, strict=False)
        
    documents = []
    for item in catalog:
        name = item.get("name", "")
        keys = item.get("keys", [])
        description = item.get("description", "")
        job_levels = item.get("job_levels", [])
        languages = item.get("languages", [])
        duration = item.get("duration", "")
        remote = item.get("remote", "")
        adaptive = item.get("adaptive", "")
        
        # Collect all types
        test_types = [TYPE_MAP[k] for k in keys if k in TYPE_MAP]
        test_type_str = ",".join(test_types)
        
        # Build rich metadata
        meta = {
            "name": name,
            "test_type": test_type_str,
            "url": item.get("link", ""),
            "adaptive": adaptive,
            "remote": remote,
            "duration": duration,
            "job_levels": ",".join(job_levels),
            "languages": ",".join(languages),
            "keys": ",".join(keys),
        }
        
        # Build rich embedding
        if not description:
            description = f"{name}. Categories: {', '.join(keys)}"
            
        page_content = f"Assessment: {name}\nDescription: {description}\nCategories: {', '.join(keys)}"
        
        if job_levels:
            page_content += f"\nJob Levels: {', '.join(job_levels)}"
        if duration:
            page_content += f"\nDuration: {duration}"
        if languages:
            page_content += f"\nLanguages: {', '.join(languages)}"
        if remote:
            page_content += f"\nRemote: {remote}"
        if adaptive:
            page_content += f"\nAdaptive: {adaptive}"
        
        doc = Document(page_content=page_content, metadata=meta)
        documents.append(doc)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", max_retries=5)
    
    batch_size = 50
    vectorstore = None
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
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
        
    vectorstore.save_local("data/faiss_index")
    print("Index built successfully.")

if __name__ == "__main__":
    build_index()
