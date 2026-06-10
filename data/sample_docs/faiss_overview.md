# FAISS Overview

FAISS (Facebook AI Similarity Search) is a library for efficient similarity
search and clustering of dense vectors. It was developed by Meta AI Research.

FAISS supports exact search with a flat index (IndexFlatL2) as well as
approximate methods such as IVF (inverted file) and HNSW graphs, which trade a
small amount of recall for large speedups on big corpora.

In this project, FAISS stores normalised sentence embeddings and serves
top-k nearest-neighbour queries in milliseconds, which keeps end-to-end
response latency under two seconds even with a local LLM.
