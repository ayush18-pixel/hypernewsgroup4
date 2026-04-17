# HyperNews MVP

This is an AI-powered hyper-personalized news recommendation system. It uses:
- **FastAPI** for the backend API.
- **SentenceTransformers & FAISS** for retrieval-augmented generation (RAG) capabilities.
- **LinUCB** Contextual Bandit for personalized recommendations based on implicit feedback.
- **NetworkX** for an entity/category-based Knowledge Graph.
- **Streamlit** for the interactive frontend demo.

## Setup Instructions

1. **Activate Virtual Environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate Synthetic Data:**
   Run the data generation script to assemble the news dataset, create sentence embeddings, and prepare the parquet files:
   ```bash
   python backend/generate_data.py
   ```

## Running the Application

1. **Start the Backend (FastAPI):**
   ```bash
   uvicorn backend.app:app --host 0.0.0.0 --port 8000
   ```
   *The backend will automatically build the FAISS index and the Knowledge Graph at startup.*

2. **Start the Frontend (Streamlit):**
   Open a new terminal, activate the `venv`, and run:
   ```bash
   streamlit run demo/demo.py
   ```

3. **Navigate:** Open your browser to `http://localhost:8501`. 
