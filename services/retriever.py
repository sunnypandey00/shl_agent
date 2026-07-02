import json
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class CustomEnsembleRetriever:
    def __init__(self, bm25, faiss):
        self.bm25 = bm25
        self.faiss = faiss
        
    def invoke(self, query):
        bm25_docs = self.bm25.invoke(query)
        faiss_docs = self.faiss.invoke(query)
        
        fused_scores = {}
        for rank, doc in enumerate(bm25_docs):
            fused_scores[doc.page_content] = {"score": fused_scores.get(doc.page_content, {"score": 0.0})["score"] + 1.0 / (rank + 60), "doc": doc}
            
        for rank, doc in enumerate(faiss_docs):
            fused_scores[doc.page_content] = {"score": fused_scores.get(doc.page_content, {"score": 0.0})["score"] + 1.0 / (rank + 60), "doc": doc}
            
        sorted_docs = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs[:10]]

def map_test_type(item):
    name = item.get("name", "")
    keys = item.get("keys", [])
    
    # Handle language overrides
    if "svar" in name.lower() or "spoken" in name.lower():
        return "K"
        
    # Handle development overrides
    if "Development & 360" in keys:
        return "D"
        
    # Standard key mapping
    letters = []
    for k in keys:
        if k == "Ability & Aptitude":
            letters.append("A")
        elif k == "Biodata & Situational Judgment":
            letters.append("B")
        elif k == "Competencies":
            letters.append("C")
        elif k == "Knowledge & Skills":
            letters.append("K")
        elif k == "Personality & Behavior":
            letters.append("P")
        elif k in ["Simulations", "Assessment Exercises"]:
            letters.append("S")
            
    # Remove duplicate keys
    unique_letters = []
    for l in letters:
        if l not in unique_letters:
            unique_letters.append(l)
            
    if not unique_letters:
        return "K" # Default fallback key
        
    return ",".join(unique_letters)

def initialize_retriever():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "shl_catalog.json")
    
    try:
        # Handle unescaped json
        with open("data/shl_catalog.json", "r", encoding="utf-8") as f:
            catalog = json.load(f, strict=False)
    except Exception as e:
        print(f"Error loading catalog: {e}")
        catalog = []

    documents = []
    for item in catalog:
        # Create text representation
        page_content = f"Name: {item.get('name', '')}\n"
        page_content += f"Description: {item.get('description', '')}\n"
        page_content += f"Keys: {', '.join(item.get('keys', []))}\n"
        
        # Extract document metadata
        metadata = {
            "name": item.get("name", ""),
            "url": item.get("link", ""),
            "test_type": map_test_type(item)
        }
        
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    if not documents:
        return None

    # Initialize Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", max_retries=3)
    
    # Load FAISS index
    faiss_vectorstore = FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # Load BM25 index
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10
    
    # Create hybrid retriever
    ensemble_retriever = CustomEnsembleRetriever(bm25_retriever, faiss_retriever)
    
    return ensemble_retriever

# Global retriever instance
ensemble_retriever = initialize_retriever()
