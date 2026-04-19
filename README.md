# HyperNews

HyperNews is a hyper-personalized news recommendation system with:
- `FastAPI` backend APIs
- `SentenceTransformers + FAISS` retrieval for RAG search
- a saved neural contextual bandit for the RL ranking path
- a `NetworkX` knowledge graph built from article/category/entity links
- a `Next.js` frontend for the web app

## Repo Layout

- `backend/`: recommendation APIs, ranking, graph logic, evaluation scripts
- `frontend/`: Next.js web app
- `data/`: processed article parquet, embeddings, FAISS assets, raw MIND data
- `graph/`: cached knowledge graph pickles
- `models/`: saved RL bandit state

## Setup

1. Create and activate a virtual environment.
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

## Data Pipeline

Build the processed dataset, embeddings, and FAISS index:

```bash
python backend/generate_data.py
```

If raw MIND files are present under `data/mind_full/MIND-small` or `data/mind`, they are used automatically. Otherwise the script falls back to a small synthetic dataset.

## Knowledge Graph

The backend can build the graph automatically at startup, but explicit helper scripts are available at the project root.

Build or rebuild the graph cache:

```bash
python build_knowledge_graph.py --force-rebuild
```

Build a smaller graph for local testing:

```bash
python build_knowledge_graph.py --max-articles 1000 --force-rebuild
```

Inspect the cached graph:

```bash
python inspect_knowledge_graph.py --top 10
python inspect_knowledge_graph.py --article-id N23093
```

There is also a quick plotting helper:

```bash
python s.py
```

## Running The App

### Backend

Start the FastAPI backend:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Useful optional environment variables:

- `HYPERNEWS_ENABLE_GRAPH=0` disables graph loading
- `HYPERNEWS_GRAPH_ARTICLE_LIMIT=100` limits graph size
- `HYPERNEWS_MAX_ARTICLES=100` loads a smaller article subset for low-memory testing
- `GROQ_API_KEY=...` enables LLM-generated explanations

Example lightweight startup:

```bash
HYPERNEWS_MAX_ARTICLES=100 HYPERNEWS_ENABLE_GRAPH=0 uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### Frontend

In a second terminal:

```bash
cd frontend
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Then open:

```text
http://localhost:3000
```

## API Checks

Health:

```bash
curl http://localhost:8000/health
```

Recommendations:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"demo1\",\"mood\":\"curious\",\"query\":\"AI\",\"n\":5}"
```

Graph stats:

```bash
curl http://localhost:8000/graph
```

## Offline Evaluation

Run the ranking evaluator on MIND-style metrics:

```bash
python backend/evaluate_mind.py --limit-impressions 100
```

Larger runs:

```bash
python backend/evaluate_mind.py --limit-impressions 500
python backend/evaluate_mind.py --limit-impressions 1000
```

The evaluator reports:
- baseline naive graph
- improved structured graph
- improved structured graph with the neural RL bandit

Metrics:
- `AUC`
- `MRR`
- `nDCG@5`
- `nDCG@10`

## Notes

- The RAG path and the RL ranking path are separate. RAG retrieval is still FAISS-based; the neural bandit only affects RL scoring.
- Graph cache files and saved local model artifacts are intentionally ignored by git.
- For low-memory local testing, prefer `HYPERNEWS_MAX_ARTICLES` instead of trying to load the full dataset.
