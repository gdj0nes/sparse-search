from sparse_search.bm25 import BM25
import pytest


def tokenizer(text):
    return text.split(" ")


@pytest.fixture(scope="function")
def corpus():
    corpus = [
        "hello world",
        "foo bar baz bat",
        "foo world"
    ]
    return corpus


@pytest.fixture(scope="function")
def bm25(corpus):
    return BM25(tokenizer=tokenizer).fit(corpus)


def test_fit(corpus):
    model = BM25(tokenizer=tokenizer).fit(corpus)

    assert model
    assert model.ndocs == 3
    assert model.avgdl == 8 / 3


def test_doc_transform(bm25, corpus):
    doc = corpus[0]

    doc = bm25.transform_doc(doc)
    assert doc


def test_query_transform(bm25, corpus):
    query = corpus[1]

    query = bm25.transform_query(query)
    assert query
