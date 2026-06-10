# Sentence Embeddings

Sentence embeddings map variable-length text to fixed-size dense vectors such
that semantically similar texts are close in vector space. This project uses
the sentence-transformers model all-MiniLM-L6-v2, which produces 384-dimensional
embeddings and offers an excellent speed/quality trade-off for retrieval.

Embeddings are computed locally via Hugging Face sentence-transformers, so no
text ever leaves the machine — a hard requirement for private corpora.
