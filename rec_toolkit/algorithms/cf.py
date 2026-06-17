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

        if self.similarity == 'cosine':
            self.user_sim_matrix = self._compute_cosine_sim_matrix(user_item_matrix)
        else:
            self.user_sim_matrix = self._compute_pearson_sim_matrix(user_item_matrix)

    def _compute_cosine_sim_matrix(self, mat: sparse.csr_matrix) -> np.ndarray:
        norm = sparse.linalg.norm(mat, axis=1)
        norm[norm == 0] = 1.0
        normed = mat.multiply(1.0 / norm[:, np.newaxis])
        sim = normed.dot(normed.T).toarray()
        np.fill_diagonal(sim, 0.0)
        return sim

    def _compute_pearson_sim_matrix(self, mat: sparse.csr_matrix) -> np.ndarray:
        n_users = mat.shape[0]
        sim = np.zeros((n_users, n_users))
        for u in range(n_users):
            for v in range(u + 1, n_users):
                s = sparse_pearson_similarity(mat, u, v, self.min_common_items)
                sim[u, v] = s
                sim[v, u] = s
        return sim

    def recommend(self, user_idx: int, n_items: int = 10,
                  exclude_seen: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if self.user_item_matrix is None:
            return np.array([]), np.array([])

        n_items_total = self.user_item_matrix.shape[1]
        user_row = self.user_item_matrix.getrow(user_idx)
        if user_row.nnz == 0:
            return np.array([]), np.array([])

        sims = self.user_sim_matrix[user_idx].copy()
        top_neighbors = np.argsort(sims)[::-1][:self.top_k]
        neighbor_sims = sims[top_neighbors]
        valid = neighbor_sims > 0
        top_neighbors = top_neighbors[valid]
        neighbor_sims = neighbor_sims[valid]

        if len(top_neighbors) == 0:
            return np.array([]), np.array([])

        scores = np.zeros(n_items_total)
        sim_sums = np.zeros(n_items_total)
        user_mean = self.user_means[user_idx]

        for i, neighbor_idx in enumerate(top_neighbors):
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

        valid_mask = sim_sums > 0
        scores[valid_mask] = scores[valid_mask] / sim_sums[valid_mask]

        if self.similarity == 'pearson':
            scores[valid_mask] += user_mean

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

        if self.similarity == 'cosine':
            self.item_similarity = self._compute_cosine_sim_matrix(user_item_matrix)
        else:
            self.item_similarity = self._compute_pearson_sim_matrix(user_item_matrix)

    def _compute_cosine_sim_matrix(self, mat: sparse.csr_matrix) -> np.ndarray:
        item_mat = mat.T.tocsr()
        norm = sparse.linalg.norm(item_mat, axis=1)
        norm[norm == 0] = 1.0
        normed = item_mat.multiply(1.0 / norm[:, np.newaxis])
        sim = normed.dot(normed.T).toarray()
        np.fill_diagonal(sim, 0.0)
        return sim

    def _compute_pearson_sim_matrix(self, mat: sparse.csr_matrix) -> np.ndarray:
        n_items = mat.shape[1]
        sim = np.zeros((n_items, n_items))
        for i in range(n_items):
            col_i = mat.getcol(i)
            if col_i.nnz < self.min_common_items:
                continue
            for j in range(i + 1, n_items):
                col_j = mat.getcol(j)
                common = np.intersect1d(col_i.indices, col_j.indices)
                if len(common) < self.min_common_items:
                    continue
                vals_i = np.array([mat[r, i] for r in common])
                vals_j = np.array([mat[r, j] for r in common])
                s = pearson_similarity(vals_i, vals_j)
                sim[i, j] = s
                sim[j, i] = s
        return sim

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
            sim_row = self.item_similarity[rated_item]
            scores += sim_row * rating

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
