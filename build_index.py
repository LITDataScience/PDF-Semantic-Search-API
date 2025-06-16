import os
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

def extract_text_from_pdf(pdf_path):
    documents = []
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} does not exist.")
        return documents
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    documents.append({
                        "filename": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "text": text.strip()
                    })
        return documents
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return documents

def split_text_into_chunks(text, max_length=500):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_embeddings(texts, model):
    if not texts:
        print("No texts provided to generate embeddings.")
        return np.array([])
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def build_faiss_index(embeddings):
    if embeddings.size == 0:
        raise ValueError("No embeddings to index. Check text extraction and chunking.")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

if __name__ == "__main__":
    pdf_path = "data/Evidence of Coverage Document assignment.pdf"
    model = SentenceTransformer('all-roberta-large-v1')

    documents = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(documents)} pages from {pdf_path}")

    chunks = []
    for doc in documents:
        text_chunks = split_text_into_chunks(doc["text"])
        for chunk in text_chunks:
            chunks.append({
                "filename": doc["filename"],
                "page": doc["page"],
                "text": chunk
            })
    print(f"Split into {len(chunks)} chunks")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts, model)
    if embeddings.size == 0:
        print("Embeddings are empty. No valid text was processed.")
        exit(1)
    embeddings = np.array(embeddings).astype('float32')
    print(f"Generated embeddings with shape: {embeddings.shape}")

    index = build_faiss_index(embeddings)
    print("FAISS index built successfully.")

    # Save index and chunks
    faiss.write_index(index, "faiss.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("Index and chunks saved to disk.")