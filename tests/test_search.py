import os
import pickle
import faiss
import numpy as np
import warnings
# Ignore TensorFlow INFO and WARNING logs:
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all messages, 1 = INFO, 2 = WARNING, 3 = ERROR
# Suppress the oneDNN custom operations message
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sentence_transformers import SentenceTransformer

# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load index and chunks
faiss_index = faiss.read_index("faiss.index")
with open("chunks.pkl", "rb") as f:
    chunk_store = pickle.load(f)

model = SentenceTransformer('all-roberta-large-v1')

def perform_search(query, top_k=3):
    query_embedding = model.encode([query])[0].astype('float32')
    distances, indices = faiss_index.search(np.array([query_embedding]), top_k)
    results = []
    for idx in indices[0]:
        if idx < len(chunk_store):
            chunk = chunk_store[idx]
            results.append(chunk)
    return results

# Test the semantic search functionality with a sample query
query = "Diabetes services"
top_k = 3
results = perform_search(query, top_k)

# Display search results in a formatted manner
print("Response Text (Excerpts):")
print("-" * 50)
for result in results:
    print(result["text"])
    print("-" * 50)

print("\nDocuments:")
print("-" * 50)
for result in results:
    print(f"Document: {result['filename']}, Page: {result['page']}")
    print("-" * 50)