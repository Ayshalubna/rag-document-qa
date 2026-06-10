from rag_qa.eval.metrics import faithfulness_proxy, hit_rate, keyword_recall, mrr


def test_hit_rate() -> None:
    retrieved = [["a.md", "b.md"], ["c.md"]]
    assert hit_rate(retrieved, ["a.md", "x.md"]) == 0.5


def test_mrr_first_vs_second_rank() -> None:
    assert mrr([["a.md", "b.md"]], ["a.md"]) == 1.0
    assert mrr([["b.md", "a.md"]], ["a.md"]) == 0.5
    assert mrr([["b.md"]], ["a.md"]) == 0.0


def test_faithfulness_fully_grounded() -> None:
    context = "FAISS is a library for similarity search of dense vectors"
    answer = "FAISS is a library for similarity search"
    assert faithfulness_proxy(answer, context) == 1.0


def test_faithfulness_detects_fabrication() -> None:
    context = "FAISS is a library for similarity search"
    answer = "Napoleon invaded Russia in 1812"
    assert faithfulness_proxy(answer, context) < 0.3


def test_keyword_recall() -> None:
    assert keyword_recall("FAISS does similarity search", ["similarity", "vectors"]) == 0.5
    assert keyword_recall("anything", []) == 1.0
