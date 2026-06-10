from rag_qa.embeddings.fake import DeterministicFakeEmbeddings


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_deterministic() -> None:
    emb = DeterministicFakeEmbeddings()
    assert emb.embed_query("hello world") == emb.embed_query("hello world")


def test_similar_texts_closer_than_dissimilar() -> None:
    emb = DeterministicFakeEmbeddings()
    q = emb.embed_query("faiss vector similarity search")
    near = emb.embed_query("faiss is a library for vector similarity search")
    far = emb.embed_query("chocolate cake baking recipe oven")
    assert _cosine(q, near) > _cosine(q, far)


def test_vectors_are_normalised() -> None:
    emb = DeterministicFakeEmbeddings()
    vec = emb.embed_query("some text")
    assert abs(sum(v * v for v in vec) - 1.0) < 1e-6


def test_embed_documents_batches() -> None:
    emb = DeterministicFakeEmbeddings()
    out = emb.embed_documents(["a", "b", "c"])
    assert len(out) == 3
