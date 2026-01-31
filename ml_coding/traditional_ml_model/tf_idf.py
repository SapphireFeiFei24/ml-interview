"""
TF-IDF Term Frequency - Inverse Document Frequency

Formula
TF-IDF(t, d, D) = TF(t, d) x IDF(t, D)
TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in the document)
IDF(t, D) = log_e(Total number of documents / Number of documents with term t in it)

“A compressed, weighted bag-of-words(no order) that turns text into geometry.”
"""

import numpy as np

def tfidf_vectorizer(
    X: np.ndarray,
    smootf_idf: bool = True,
    norm: str | None = "l2",
    eps: float = 1e-12) -> np.ndarray:
    """
    Compute TF-IDF feature matrix
    :param X: (n_docs, n_terms)
    :param smootf_idf:
    :param norm:
    :param eps: Small constant for numerical stability.
    :return:TF-IDF matrix of shape (n_docs, n_terms).
    """
    X = X.astype(np.float64, copy=False)
    n_docs, n_terms = X.shape

    # ----- TF -----------
    # term frequency normalized by document lenth
    doc_length = X.sum(axis=1, keepdims=True)
    tf = X / (doc_length + eps)

    # ---------- IDF ----------
    # document frequency
    df = np.count_nonzero(X, axis=0)
    if smootf_idf:
        idf = np.log((1 + n_docs) / (1 + df)) + 1
    else:
        idf = np.log(n_docs / (df + eps))

    # ---------- TF-IDF ----------
    tfidf = tf * idf

    # ---------- Normalization ----------
    if norm == "l2":
        row_norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        tfidf = tfidf / (row_norms + eps)
    return tfidf
