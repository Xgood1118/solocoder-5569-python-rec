import numpy as np
from scipy import sparse
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod
from .similarity import cosine_similarity, pearson_similarity, sparse_pearson_similarity


class CollaborativeFiltering(ABC):
    def __init__(self, top_k: int = 50, similarity: str = 'cosine',
                 min_common_items: int = 3):
        self.top_k = top_k
        self.similarity = similarity
        self.min_common_items = min_common_items
        self.user_item_matrix: Optional[sparse.csr_matrix] = None
        self.similarity_matrix: Optional[np.ndarray] = None

    @abstractmethod
    def fit(self, user_item_matrix: sparse.csr_matrix):
        pass

    @abstractmethod
    def recommend(self, user_idx: int, n_items: int = 10,
                  exclude_seen: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def _sim_func(self, idx_a: int, idx_b: int) -> float:
        if self.similarity == 'cosine':
            return self._cosine_sim(idx_a, idx_b)
        elif self.similarity == 'pearson':
            return sparse_pearson_similarity(
                self.user_item_matrix, idx_a, idx_b, self.min_common_items
            )
        else:
            raise ValueError(f"未知的相似度度量: {self.similarity}")

    def _cosine_sim(self, idx_a: int, idx_b: int) -> float:
        row_a = self.user_item_matrix.getrow(idx_a)
        row_b = self.user_item_matrix.getrow(idx_b)
        dot = row_a.dot(row_b.T).toarray()[0, 0]
        norm_a = sparse.linalg.norm(row_a)
        norm_b = sparse.linalg.norm(row_b)
        return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0


class UserCF(CollaborativeFiltering):
    def __init__(self, top_k: int = 50, similarity: str = 'cosine',
                 min_common_items: int = 3):
        super().__init__(top_k, similarity, min_common_items)
        self.user_similarity: Optional[np.ndarray] = None

    def fit(self, user_item_matrix: sparse.csr_matrix):
        self.user_item_matrix = user_item_matrix
        n_users = user_item_matrix.shape[0]
        self.user_means = np.zeros(n_users)

        for u in range(n_users):
            row = user_item_matrix.getrow(u)
            if row.nnz > 0:
                self.user_means[u] = row.data.mean()

        self.user_similarity = self._compute_user_similarity_matrix(user_item_matrix)

    def _compute_user_similarity_matrix(self, user_item_matrix: sparse.csr_matrix) -> np.ndarray:
        n_users = user_item_matrix.shape[0]

        if self.similarity == 'cosine':
            norms = sparse.linalg.norm(user_item_matrix, axis=1)
            norms_matrix = norms[:, np.newaxis] * norms[np.newaxis, :]
            norms_matrix[norms_matrix == 0] = 1.0
            dot_products = user_item_matrix.dot(user_item_matrix.T).toarray()
            sim_matrix = dot_products / norms_matrix
            np.fill_diagonal(sim_matrix, 0.0)
            return sim_matrix
        else:
            sim_matrix = np.zeros((n_users, n_users))
            item_user_matrix = user_item_matrix.T.tocsr()
            for i in range(n_users):
                row_i = user_item_matrix.getrow(i)
                if row_i.nnz == 0:
                    continue
                candidate_users = set()
                for item_idx in row_i.indices:
                    col = item_user_matrix.getrow(item_idx)
                    candidate_users.update(col.indices.tolist())
                candidate_users.discard(i)
                for j in candidate_users:
                    if j > i:
                        sim = sparse_pearson_similarity(
                            user_item_matrix, i, j, self.min_common_items
                        )
                        sim_matrix[i, j] = sim
                        sim_matrix[j, i] = sim
            return sim_matrix

    def _find_neighbors(self, user_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.user_similarity is None:
            n_users = self.user_item_matrix.shape[0]
            user_row = self.user_item_matrix.getrow(user_idx)
            if user_row.nnz == 0:
                return np.array([]), np.array([])
            similarities = np.zeros(n_users)
            for u in range(n_users):
                if u == user_idx:
                    continue
                similarities[u] = self._sim_func(user_idx, u)
        else:
            similarities = self.user_similarity[user_idx].copy()

        neighbor_indices = np.argsort(similarities)[::-1][:self.top_k]
        neighbor_sims = similarities[neighbor_indices]

        valid = neighbor_sims > 0
        return neighbor_indices[valid], neighbor_sims[valid]

    def recommend(self, user_idx: int, n_items: int = 10,
                  exclude_seen: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if self.user_item_matrix is None:
            return np.array([]), np.array([])

        n_items_total = self.user_item_matrix.shape[1]
        neighbor_indices, neighbor_sims = self._find_neighbors(user_idx)

        if len(neighbor_indices) == 0:
            return np.array([]), np.array([])

        user_mean = self.user_means[user_idx]
        neighbor_matrix = self.user_item_matrix[neighbor_indices]

        if self.similarity == 'pearson':
            neighbor_means = self.user_means[neighbor_indices]
            centered = neighbor_matrix.copy()
            for k in range(len(neighbor_indices)):
                if centered[k].nnz > 0:
                    centered.data[centered.indptr[k]:centered.indptr[k+1]] -= neighbor_means[k]
            scores = (neighbor_sims @ centered).toarray().flatten()
            sim_sums = np.abs(neighbor_sims) @ (neighbor_matrix != 0).astype(float).toarray().flatten()
            valid = sim_sums > 0
            scores[valid] = scores[valid] / sim_sums[valid] + user_mean
        else:
            scores = (neighbor_sims @ neighbor_matrix).toarray().flatten()
            sim_sums = neighbor_sims @ (neighbor_matrix != 0).astype(float).toarray().flatten()
            valid = sim_sums > 0
            scores[valid] = scores[valid] / sim_sums[valid]

        if exclude_seen:
            seen_items = self.user_item_matrix.getrow(user_idx).indices
            scores[seen_items] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n_items]
        top_scores = scores[top_indices]

        valid = top_scores > -np.inf
        return top_indices[valid], top_scores[valid]


class ItemCF(CollaborativeFiltering):
    def __init__(self, top_k: int = 50, similarity: str = 'cosine',
                 min_common_items: int = 3):
        super().__init__(top_k, similarity, min_common_items)
        self.item_similarity: Optional[np.ndarray] = None

    def fit(self, user_item_matrix: sparse.csr_matrix):
        self.user_item_matrix = user_item_matrix
        n_items = user_item_matrix.shape[1]
        item_item_matrix = user_item_matrix.T.tocsr()

        if self.similarity == 'cosine':
            self.item_similarity = self._compute_cosine_similarity_matrix(item_item_matrix)
        else:
            self.item_similarity = self._compute_pearson_similarity_matrix(item_item_matrix, n_items)

    def _compute_cosine_similarity_matrix(self, item_item_matrix: sparse.csr_matrix) -> np.ndarray:
        n_items = item_item_matrix.shape[0]
        norms = sparse.linalg.norm(item_item_matrix, axis=1)
        norms_matrix = norms[:, np.newaxis] * norms[np.newaxis, :]
        norms_matrix[norms_matrix == 0] = 1.0
        dot_products = item_item_matrix.dot(item_item_matrix.T).toarray()
        sim_matrix = dot_products / norms_matrix
        np.fill_diagonal(sim_matrix, 0.0)
        return sim_matrix

    def _compute_pearson_similarity_matrix(self, item_item_matrix: sparse.csr_matrix, n_items: int) -> np.ndarray:
        sim_matrix = np.zeros((n_items, n_items))
        user_item_matrix = item_item_matrix.T.tocsr()

        for i in range(n_items):
            col_i = item_item_matrix.getrow(i)
            if col_i.nnz == 0:
                continue
            candidate_items = set()
            for user_idx in col_i.indices:
                row = user_item_matrix.getrow(user_idx)
                candidate_items.update(row.indices.tolist())
            candidate_items.discard(i)
            for j in candidate_items:
                if j > i:
                    sim = self._item_pearson_sim(i, j)
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
        return sim_matrix

    def _item_cosine_sim(self, item_a: int, item_b: int) -> float:
        col_a = self.user_item_matrix.getcol(item_a)
        col_b = self.user_item_matrix.getcol(item_b)
        dot = col_a.T.dot(col_b).toarray()[0, 0]
        norm_a = sparse.linalg.norm(col_a)
        norm_b = sparse.linalg.norm(col_b)
        return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

    def _item_pearson_sim(self, item_a: int, item_b: int) -> float:
        col_a = self.user_item_matrix.getcol(item_a)
        col_b = self.user_item_matrix.getcol(item_b)

        common_rows = np.intersect1d(col_a.indices, col_b.indices)
        if len(common_rows) < self.min_common_items:
            return 0.0

        vals_a = col_a[common_rows].toarray().flatten()
        vals_b = col_b[common_rows].toarray().flatten()

        return pearson_similarity(vals_a, vals_b)

    def recommend(self, user_idx: int, n_items: int = 10,
                  exclude_seen: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if self.user_item_matrix is None or self.item_similarity is None:
            return np.array([]), np.array([])

        n_items_total = self.user_item_matrix.shape[1]
        user_row = self.user_item_matrix.getrow(user_idx)

        if user_row.nnz == 0:
            return np.array([]), np.array([])

        rated_items = user_row.indices
        ratings = user_row.data

        sim_subset = self.item_similarity[rated_items]
        scores = ratings @ sim_subset

        if exclude_seen:
            scores[rated_items] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n_items]
        top_scores = scores[top_indices]

        valid = top_scores > 0
        return top_indices[valid], top_scores[valid]

    def get_similar_items(self, item_idx: int, n: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if self.item_similarity is None:
            return np.array([]), np.array([])

        sims = self.item_similarity[item_idx]
        top_indices = np.argsort(sims)[::-1][:n + 1]
        top_indices = top_indices[top_indices != item_idx][:n]
        return top_indices, sims[top_indices]
