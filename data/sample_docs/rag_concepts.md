# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines an information retrieval step
with a generative language model. Instead of relying solely on the model's
parametric knowledge, relevant passages are retrieved from a document corpus
and injected into the prompt as grounding context.

The main benefits of RAG are reduced hallucination, the ability to answer
questions about private or recent documents, and source attribution: every
answer can cite the passages it was derived from.

Key tuning parameters include chunk size, chunk overlap, the number of
retrieved passages (top-k), and the similarity score threshold below which
passages are discarded.
