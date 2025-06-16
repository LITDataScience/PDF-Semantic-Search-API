# PDF Semantic Search API

This project provides a semantic search API over medical PDF documents using Sentence Transformers and FAISS. It also includes scripts for building the vector index and for direct search testing, as well as a full test suite.

---

## 1. Installation

1. **Clone the repository**
2. **Create and activate a virtual environment (optional but recommended):**
   ```sh
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```
3. **Install requirements:**
   ```sh
   pip install -r requirements.txt
   ```

---

## 2. Build the FAISS Index

Before running the API or search, you must build the index:

```sh
python build_index.py
```

This will process the PDF(s) in the `data/` folder and create `faiss.index` and `chunks.pkl` files.

---

## 3. Run the API Server

Start the FastAPI server:

```sh
python app.py
```

The API will be available at `http://localhost:8000`.

### Search Endpoint
- **POST** `/search`
- **Body Example:**
  ```json
  {
    "query": "Diabetes services",
    "top_k": 3
  }
  ```
- **Response:**
  ```json
  {
    "results": [
      {"filename": "...", "page": 1, "excerpt": "..."},
      ...
    ]
  }
  ```

You can test this endpoint using [Postman](https://www.postman.com/), [curl](https://curl.se/), or the FastAPI docs at `http://localhost:8000/docs`.

---

## 4. Run Direct Search (Without API)

You can test the semantic search logic directly in the terminal, without using the API, by running:

```sh
python tests/test_search.py
```

This will print the top search results for a sample query.

---

## 5. Run All Tests

To run all unit and integration tests:

```sh
python -m unittest discover tests
```

Or, if you have `pytest` installed:

```sh
pytest tests
```

---

## 6. Suppress TensorFlow Warnings (Optional)

To reduce TensorFlow log output, add these lines at the top of your test scripts:

```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

Or set the environment variables in your shell before running tests:

```sh
$env:TF_CPP_MIN_LOG_LEVEL=3
$env:TF_ENABLE_ONEDNN_OPTS=0
```

---

## 7. Project Structure

```
├── app.py                # FastAPI app
├── build_index.py        # Script to build FAISS index
├── requirements.txt      # Python dependencies
├── data/                 # PDF files
├── faiss.index           # Saved FAISS index (generated)
├── chunks.pkl            # Saved chunk metadata (generated)
└── tests/
    ├── test_build_index.py
    ├── test_app.py
    └── test_search.py
```

---

## 8. Notes
- Make sure to re-run `build_index.py` if you add or change PDF files in `data/`.
- The API and test scripts expect the index and chunk files to exist.
- For best results, use a GPU-enabled environment for embedding generation.

---

## 9. License
MIT License

---

## 10. Run the Project in Kaggle or Jupyter Notebook

You can also run the entire workflow in a notebook environment such as Kaggle or Jupyter using the included notebook:

- **semantic-search-workflow.ipynb**

This notebook demonstrates all steps: installing requirements, extracting text from PDF, chunking, embedding, building the FAISS index, and performing semantic search—all in one place. Simply open the notebook and run the cells sequentially.
