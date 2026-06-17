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

    def fit(self, user_item_matrix: sparse.csr_matrix):
        self.user_item_matrix = user_item_matrix
        n_users = user_item_matrix.shape[0]
        self.user_means = np.zeros(n_users)

        for u in range(n_users):
            row = user_item_matrix.getrow(u)
            if row.nnz > 0:
                self.user_means[u] = row.data.mean()

    def _find_neighbors(self, user_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        n_users = self.user_item_matrix.shape[0]
        user_row = self.user_item_matrix.getrow(user_idx)

        if user_row.nnz == 0:
            return np.array([]), np.array([])

        similarities = np.zeros(n_users)
        for u in range(n_users):
            if u == user_idx:
                continue
            similarities[u] = self._sim_func(user_idx, u)

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

        scores = np.zeros(n_items_total)
        sim_sums = np.zeros(n_items_total)

        user_mean = self.user_means[user_idx]

        for i, neighbor_idx in enumerate(neighbor_indices):
            sim = neighbor_sims[i]
            neighbor_row = self.user_item_matrix.getrow(neighbor_idx)

            if self.similarity == 'pearson':
                neighbor_mean = self.user_means[neighbor_idx]
                for j, item_idx in enumerate(neighbor_row.indices):
                    rating = neighbor_row.data[j]
                    scores[item_idx] += sim * (rating - neighbor_mean)
                    sim_sums[item_idx] += abs(sim)
            else:
                for j, item_idx in enumerate(neighbor_row.indices):
                    rating = neighbor_row.data[j]
                    scores[item_idx] += sim * rating
                    sim_sums[item_idx] += sim

        valid = sim_sums > 0
        scores[valid] = scores[valid] / sim_sums[valid]

        if self.similarity == 'pearson':
            scores[valid] += user_mean

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
        self.item_similarity = np.zeros((n_items, n_items))

        item_item_matrix = user_item_matrix.T.tocsr()

        for i in range(n_items):
            for j in range(i + 1, n_items):
                if self.similarity == 'cosine':
                    sim = self._item_cosine_sim(i, j)
                else:
                    sim = self._item_pearson_sim(i, j)
                self.item_similarity[i, j] = sim
                self.item_similarity[j, i] = sim

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

        vals_a = np.array([self.user_item_matrix[r, item_a] for r in common_rows])
        vals_b = np.array([self.user_item_matrix[r, item_b] for r in common_rows])

        return pearson_similarity(vals_a, vals_b)

    def recommend(self, user_idx: int, n_items: int = 10,
                  exclude_seen: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if self.user_item_matrix is None or self.item_similarity is None:
            return np.array([]), np.array([])

        n_items_total = self.user_item_matrix.shape[1]
        user_row = self.user_item_matrix.getrow(user_idx)

        if user_row.nnz == 0:
            return np.array([]), np.array([])

        scores = np.zeros(n_items_total)

        for i, rated_item in enumerate(user_row.indices):
            rating = user_row.data[i]

            similar_items = np.argsort(self.item_similarity[rated_item])[::-1][:self.top_k + 1]

            for sim_item in similar_items:
                if sim_item == rated_item:
                    continue
                scores[sim_item] += self.item_similarity[rated_item, sim_item] * rating

        if exclude_seen:
            seen_items = user_row.indices
            scores[seen_items] = -np.inf

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
