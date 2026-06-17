import numpy as np
from scipy import sparse
from scipy.spatial.distance import cosine, correlation
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    if np.all(vector_a == 0) or np.all(vector_b == 0):
        return 0.0
    dot = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0


def pearson_similarity(vector_a: np.ndarray, vector_b: np.ndarray,
                       common_indices: Optional[np.ndarray] = None) -> float:
    if common_indices is not None:
        if len(common_indices) < 2:
            return 0.0
        a = vector_a[common_indices]
        b = vector_b[common_indices]
    else:
        a = vector_a
        b = vector_b

    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0

    mean_a = np.mean(a)
    mean_b = np.mean(b)

    centered_a = a - mean_a
    centered_b = b - mean_b

    numerator = np.sum(centered_a * centered_b)
    denominator = np.sqrt(np.sum(centered_a ** 2)) * np.sqrt(np.sum(centered_b ** 2))

    return numerator / denominator if denominator != 0 else 0.0


def sparse_cosine_similarity(sparse_mat: sparse.csr_matrix, idx_a: int, idx_b: int) -> float:
    row_a = sparse_mat.getrow(idx_a).toarray().flatten()
    row_b = sparse_mat.getrow(idx_b).toarray().flatten()
    return cosine_similarity(row_a, row_b)


def sparse_pearson_similarity(sparse_mat: sparse.csr_matrix, idx_a: int, idx_b: int,
                              min_common: int = 3) -> float:
    row_a = sparse_mat.getrow(idx_a)
    row_b = sparse_mat.getrow(idx_b)

    common_cols = np.intersect1d(row_a.indices, row_b.indices)
    if len(common_cols) < min_common:
        return 0.0

    vals_a = row_a[0, common_cols].toarray().flatten()
    vals_b = row_b[0, common_cols].toarray().flatten()

    return pearson_similarity(vals_a, vals_b)
