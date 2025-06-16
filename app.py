import faiss
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Medical PDF Semantic Search API")
model = SentenceTransformer('all-roberta-large-v1')

# Load FAISS index and chunks at startup
faiss_index = None
chunk_store = None

@app.on_event("startup")
def load_index():
    global faiss_index, chunk_store
    faiss_index = faiss.read_index("faiss.index")
    with open("chunks.pkl", "rb") as f:
        chunk_store = pickle.load(f)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
async def search(request: QueryRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if faiss_index is None or chunk_store is None:
        raise HTTPException(status_code=500, detail="Index not available")
    query_embedding = model.encode([request.query])[0].astype('float32')
    distances, indices = faiss_index.search(np.array([query_embedding]), request.top_k)
    results = []
    for idx in indices[0]:
        if idx < len(chunk_store):
            chunk = chunk_store[idx]
            results.append({
                "filename": chunk["filename"],
                "page": chunk["page"],
                "excerpt": chunk["text"]
            })
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)