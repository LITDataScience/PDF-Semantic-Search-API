import os
import tempfile
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from build_index import extract_text_from_pdf, split_text_into_chunks, generate_embeddings, build_faiss_index

def test_extract_text_from_pdf():
    # Use a small sample PDF for testing
    sample_pdf = "data/Evidence of Coverage Document assignment.pdf"
    docs = extract_text_from_pdf(sample_pdf)
    assert isinstance(docs, list)
    assert all("text" in d for d in docs)
    assert len(docs) > 0

def test_split_text_into_chunks():
    text = "Sentence one. Sentence two. Sentence three."
    chunks = split_text_into_chunks(text, max_length=20)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert len(chunks) >= 2

def test_generate_embeddings():
    model = SentenceTransformer('all-roberta-large-v1')
    texts = ["This is a test.", "Another sentence."]
    embeddings = generate_embeddings(texts, model)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2

def test_build_faiss_index():
    arr = np.random.rand(5, 384).astype('float32')
    index = build_faiss_index(arr)
    assert isinstance(index, faiss.IndexFlatL2)
    D, I = index.search(arr, 2)
    assert D.shape == (5, 2)
    assert I.shape == (5, 2)