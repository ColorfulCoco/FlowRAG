# FlowRAG

**Flow-Logic Aware Retrieval-Augmented Generation for Industrial SOP Documents**

FlowRAG is a hybrid RAG system designed for industrial Standard Operating Procedure (SOP) documents that interleave **flow logic** (causality, steps, decision branches) with **declarative knowledge** (parameters, specifications, definitions). Unlike vanilla vector-based RAG, FlowRAG builds a knowledge graph from procedural text, performs graph-guided multi-hop reasoning, and supplements results with keyword/vector retrieval under a fixed context token budget.

---

## Architecture

```
                     ┌─────────────────────────────────────────┐
                     │            Build Pipeline               │
                     │                                         │
  SOP Documents ───▶ │  LLM ──▶ Mermaid ──▶ Knowledge Graph   │
                     │           │              │              │
                     │           ▼              ▼              │
                     │     Mermaid files   Neo4j + JSON        │
                     │                                         │
  SOP Documents ───▶ │  Chunk ──▶ Embed ──▶ FAISS VectorDB    │
                     │      │                                  │
                     │      ▼                                  │
                     │  Keyword Index + Entity Index            │
                     │      │                                  │
                     │      ▼                                  │
                     │  Knowledge Mounting (chunks → nodes)     │
                     └─────────────────────────────────────────┘

                     ┌─────────────────────────────────────────┐
                     │          Retrieval Pipeline              │
                     │                                         │
  Query ───────────▶ │  Anchor Nodes (embedding similarity)    │
                     │      │                                  │
                     │      ▼                                  │
                     │  Multi-hop Graph Exploration (LLM)      │
                     │      │                                  │
                     │      ▼                                  │
                     │  Keyword + Vector Supplement             │
                     │      │                                  │
                     │      ▼                                  │
                     │  Context Assembly (token budget λ)       │
                     │      │                                  │
                     │      ▼                                  │
                     │  Answer Generation (LLM)                │
                     └─────────────────────────────────────────┘
```

## Key Features

- **Mermaid-mediated KG construction**: LLM generates Mermaid flowcharts as an intermediate representation, then parsed into a knowledge graph — improves structural fidelity over direct entity extraction.
- **Residual structural embeddings**: Node embeddings are enriched with multi-hop neighbor aggregation (residual fusion), capturing topological context.
- **Graph-guided multi-hop reasoning**: Starting from anchor nodes, an LLM agent iteratively explores the graph, collecting knowledge along causal chains.
- **Hybrid retrieval with token budget**: After graph exploration, remaining token budget is filled via keyword + vector retrieval (RRF fusion).
- **Modular architecture**: Clean, extensible pipeline with well-defined stages.

## Quick Start

### 1. Prerequisites

- Python 3.9+
- Neo4j 5.x running locally (or remote)
- An OpenAI-compatible LLM API (e.g., Alibaba DashScope / Qwen)

### 2. Install

```bash
git clone https://github.com/your-org/FlowRAG.git
cd FlowRAG
pip install -e .
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your API key, Neo4j credentials, etc.
```

### 4. Build Knowledge Base

```bash
python examples/quick_start.py --build
```

### 5. Query

```bash
python examples/quick_start.py --query "What is the operating procedure when turbine speed exceeds limit?"
```

### 6. Batch Retrieval + Evaluation

```bash
python examples/quick_start.py --batch --workers 8 --auto-eval
```

## Project Structure

```
FlowRAG/
├── src/                          # Core library
│   ├── config.py                 # Configuration (reads from .env)
│   ├── models/schemas.py         # Data models (GraphNode, TextChunk, etc.)
│   ├── pipeline/
│   │   ├── build_pipeline.py     # Build pipeline (graph + documents + embeddings)
│   │   ├── retrieval_pipeline.py # Retrieval pipeline (graph + vector + generation)
│   │   └── build_cache.py        # Incremental build cache
│   ├── graph/
│   │   ├── mermaid_parser.py     # Mermaid parsing & graph generation
│   │   ├── neo4j_store.py        # Neo4j storage
│   │   └── graph_store.py        # In-memory graph operations
│   ├── embedding/
│   │   ├── embedder.py           # Embedding model wrapper (OpenAI / local)
│   │   └── context_embedder.py   # Residual structural enrichment
│   ├── vector_store/
│   │   └── vector_db.py          # FAISS/Chroma + BM25 hybrid search
│   ├── retriever/
│   │   ├── graph_retriever.py    # Graph retrieval with causal chain expansion
│   │   ├── vector_retriever.py   # Vector similarity retrieval
│   │   └── hybrid_retriever.py   # RRF / weighted / cascade fusion
│   ├── knowledge/
│   │   ├── knowledge_mounter.py  # Mount declarative chunks onto graph nodes
│   │   ├── keyword_index.py      # Keyword-chunk index (exact + semantic)
│   │   ├── entity_index.py       # Entity-chunk/node index
│   │   └── ...
│   └── utils/
│       ├── openai_client.py      # OpenAI client factory & rate limiter
│       ├── text_splitter.py      # Paragraph / sentence / semantic splitting
│       └── token_counter.py      # Token counting & cost tracking
├── examples/
│   └── quick_start.py            # CLI entry point (build / query / batch)
├── evaluate/                     # Evaluation framework
├── .env.example                  # Environment variable template
├── pyproject.toml                # Package definition
└── requirements.txt              # Dependencies
```

## Citation

If you use FlowRAG in your research, please cite:

```bibtex
@article{flowrag2025,
  title={FlowRAG: Flow-Logic Aware Retrieval-Augmented Generation},
  author={...},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
