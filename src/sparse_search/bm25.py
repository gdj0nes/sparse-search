import numpy as np
from typing import List, Callable, Optional, Dict

from sklearn.feature_extraction.text import HashingVectorizer

SparseDict = Dict[int, float]
Array = List[float]


class BM25:
    """Implementation of OKapi BM25 with HashingVectorizer"""

    def __init__(self, tokenizer: Callable[[str], List[str]], n_features=2 ** 16, b=0.75, k1=1.6):
        """

        Args:
            tokenizer: Converts strs to a list of tokens
            n_features: The number of features
            b:
            k1:
        """
        self.ndocs: int = 0
        self.n_features: int = n_features
        self.doc_freq: Array = []
        self.avgdl: Optional[float] = None
        self._tokenizer: Callable[[str], List[str]] = tokenizer
        self._vectorizer = HashingVectorizer(
            n_features=n_features, token_pattern=None, tokenizer=tokenizer, norm=None, alternate_sign=False
        ).transform
        self.b: float = b
        self.k1: float = k1

    def fit(self, corpus: List[str]) -> "BM25":
        """Fit IDF to documents X"""

        X = self._vectorizer(corpus)
        self.avgdl = X.sum(1).mean()
        self.ndocs = X.shape[0]
        self.doc_freq = (
            HashingVectorizer(n_features=self.n_features, tokenizer=self._tokenizer, norm=None, binary=True)
            .transform(corpus)
            .sum(axis=0)
            .A1
        )
        return self

    def vectorize(self, text) -> SparseDict:
        sparse_array = self._vectorizer(text)
        return {int(k): v for k, v in zip(sparse_array.indices, sparse_array.data)}

    def tokenize(self, text: str) -> List[str]:
        return self._tokenizer(text)

    def transform_doc(self, doc: str) -> SparseDict:
        """Normalize document for BM25 scoring

        Args:
            doc:

        Returns:

        """
        doc_vec = self._vectorizer([doc])
        norm_doc_tf = self._norm_doc_tf(doc_vec)
        return {int(k): v for k, v in zip(doc_vec.indices, norm_doc_tf)}

    def transform_query(self, query: str):
        """Normalize query for BM25 scoring"""
        query = self._vectorizer([query])
        norm_query_tf = self._norm_query_tf(query)
        return {int(k): v for k, v in zip(query.indices, norm_query_tf)}

    def _norm_doc_tf(self, doc_tf) -> np.ndarray:
        """Calculate BM25 normalized document term-frequencies

        Args:
            doc_tf:

        Returns:
            norm_tf: normalized scores

        """
        b, k1, avgdl = self.b, self.k1, self.avgdl
        values = doc_tf.data
        dl = values.sum()
        norm_tf = values / (k1 * (1.0 - b + b * (dl / avgdl)) + values)

        return norm_tf

    def _norm_query_tf(
            self,
            query_tf,
    ):
        """Calculate BM25 normalized query term-frequencies

        Args:
            query_tf: sparse csr matrix of query term frequencies

        Returns:
            norm_query_tf:
        """

        idf = np.log((self.ndocs + 1) / (self.doc_freq[query_tf.indices] + 0.5))
        sum_df = idf.sum()
        norm_query_tf = idf / sum_df

        return norm_query_tf
