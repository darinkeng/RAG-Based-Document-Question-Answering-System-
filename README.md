# RAG-Based arXiv Question Answering System 

A portfolio project that builds the retrieval layer of a Retrieval-Augmented Generation (RAG) system over 100,000 arXiv paper abstracts. The project turns raw research metadata into clean retrieval-ready data, generates text chunks, creates Sentence-BERT embeddings, builds a FAISS vector index, and evaluates retrieval quality across different chunking strategies.

## Project Overview

This project focuses on the data engineering and machine learning workflow behind that retrieval step: cleaning semi-structured arXiv records, preparing document chunks, embedding text, indexing vectors, and testing whether the retriever returns relevant papers for scientific queries.

The main goal of this project is to answer:

> How can we build a scalable retrieval pipeline that returns relevant arXiv abstracts for domain-specific research questions?

## Key Features

* Built an end-to-end RAG retrieval pipeline for 100k arXiv abstracts using Python, Pandas, Sentence-BERT, and FAISS.
* Cleaned raw JSONL metadata into a retrieval-ready schema containing paper IDs, titles, abstracts, categories, update dates, and publication years.
* Compared fixed-size overlapping chunks with semantic chunking based on sentence similarity.
* Generated dense vector embeddings using `all-MiniLM-L6-v2` and normalized them for cosine-similarity search.
* Built a FAISS `IndexFlatIP` vector index for top-k nearest-neighbor retrieval.
* Evaluated retrieval behavior on both in-domain scientific questions and out-of-domain general questions.
* Identified score-threshold abstention as a practical improvement for reducing irrelevant retrieved context.

## Repository Structure

```text
.
├── assets/
│   └── rag_pipeline.png
├── data/
│   ├── README.md
│   └── sample/
│       └── arxiv_sample_100.jsonl
├── docs/
│   ├── assignment_prompt.md
│   └── methodology_and_results.md
├── notebooks/
│   └── DATA342_HW4_Answer.ipynb
├── results/
│   └── .gitkeep
├── src/
│   ├── embedding_utils.py
│   ├── rag_generate.py
│   └── rag_pipeline.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Dataset

I used `arxiv_clean_100k.json`, a JSONL dataset containing 100,000 arXiv records. To keep this GitHub repository lightweight, the full dataset is not committed. 

Generated outputs, such as embeddings, Parquet files, FAISS indexes, and evaluation CSVs, are excluded from Git because they can be large and reproduced from the notebook or source code.

## Methodology

### 1. Data Ingestion and Cleaning

The pipeline reads the raw JSONL file line by line, skips malformed records, standardizes text fields, removes missing abstracts, and drops duplicate paper IDs.

The cleaned table keeps the fields most useful for retrieval:

| Field         | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `id`          | Unique arXiv paper identifier                                |
| `title`       | Paper title shown in retrieval results                       |
| `abstract`    | Main text used for chunking and embedding                    |
| `categories`  | arXiv subject categories                                     |
| `update_date` | Last update date for the paper record                        |
| `year`        | Parsed publication or update year for filtering and analysis |

### 2. Chunking

The project compares two chunking approaches:

| Strategy             | Description                                                     | Benefit                                | Limitation                                  |
| -------------------- | --------------------------------------------------------------- | -------------------------------------- | ------------------------------------------- |
| Overlapping chunking | Splits abstracts into fixed-size character windows with overlap | Simple and fast                        | May split sentences or concepts unnaturally |
| Semantic chunking    | Splits text based on adjacent sentence similarity               | Produces more coherent retrieval units | More computationally expensive              |

### 3. Embedding Generation

Text chunks are converted into dense vector embeddings using `all-MiniLM-L6-v2` from Sentence Transformers. Embeddings are normalized so that inner product search in FAISS behaves like cosine similarity.

### 4. Vector Indexing and Retrieval

The vector index is built with FAISS:

```python
index = faiss.IndexFlatIP(embedding_dim)
index.add(normalized_embeddings)
```

For each query, the retriever embeds the query text, searches the FAISS index, and returns the top-k matching chunks with scores, paper titles, and source metadata.

### 5. Evaluation

Retrieval quality was tested using both in-domain and out-of-domain queries:

| Query Type                  | Example                                               |
| --------------------------- | ----------------------------------------------------- |
| In-domain scientific query  | Entanglement entropy and anti-de Sitter black holes   |
| In-domain scientific query  | Quasinormal frequencies and Schwarzschild black holes |
| Out-of-domain general query | Capital of France                                     |
| Out-of-domain general query | Green tea benefits                                    |

The in-domain physics queries produced much higher top-ranked similarity scores than the off-topic queries. This suggests that a production RAG system should include a minimum similarity threshold so the retriever can abstain when no relevant context is found.


## Example Usage

```python
from src.rag_pipeline import (
    ingest_arxiv_jsonl,
    clean_schema,
    build_chunk_table,
    embed_chunks,
    build_faiss_index,
    retrieve,
)

raw = ingest_arxiv_jsonl("data/sample/arxiv_sample_100.jsonl")
clean = clean_schema(raw)
chunks = build_chunk_table(clean, method="overlap")
embeddings = embed_chunks(chunks)
index = build_faiss_index(embeddings)

results = retrieve(
    "How is entanglement entropy computed for anti-de Sitter black holes?",
    index,
    chunks,
    k=5,
)

print(results[["rank", "score", "title"]])
```

## Results and Insights

* In-domain scientific queries returned high-scoring arXiv chunks that were semantically related to the query topic.
* Out-of-domain general questions returned lower similarity scores, showing that raw nearest-neighbor retrieval will still return something even when the corpus does not contain a good answer.
* Semantic chunking improved context coherence, while overlapping chunks were easier and faster to generate.
* A score threshold would make the system safer by allowing it to return “no relevant context found” instead of passing weak context into a language model.

## Skills Demonstrated

* Data ingestion from semi-structured JSONL files
* Data cleaning and schema design with Pandas
* Text preprocessing and chunking for RAG systems
* Sentence-BERT embedding generation
* Vector search with FAISS
* Retrieval evaluation using top-k search results
* Practical ML system design for reducing irrelevant context and hallucination risk

## Future Improvements

* Add a command-line interface for ingestion, indexing, and querying.
* Add a reranking step after FAISS retrieval.
* Add a score threshold for out-of-domain query abstention.
* Add unit tests for schema cleaning, chunking, embedding shape checks, and retrieval output format.
