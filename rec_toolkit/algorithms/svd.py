import numpy as np
from scipy import sparse
from typing import Tuple, Optional


class SVDRecommender:
    def __init__(self, n_factors: int = 100, n_epochs: int = 5,
                 lr_all: float = 0.005, reg_all: float = 0.02,
                 incremental: bool = True, batch_size: int = 256):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.incremental = incremental
        self.batch_size = batch_size

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        self.global_mean: float = 0.0
        self.n_users: int = 0
        self.n_items: int = 0

    def fit(self, user_item_matrix: sparse.csr_matrix):
        self.n_users, self.n_items = user_item_matrix.shape
        self.global_mean = self._compute_global_mean(user_item_matrix)

        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)

        rows, cols, ratings = sparse.find(user_item_matrix)
        n_samples = len(ratings)

        for epoch in range(self.n_epochs):
            perm = np.random.permutation(n_samples)
            rows_shuffled = rows[perm]
            cols_shuffled = cols[perm]
            ratings_shuffled = ratings[perm]

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_u = rows_shuffled[start:end]
                batch_i = cols_shuffled[start:end]
                batch_r = ratings_shuffled[start:end]
                batch_size_actual = end - start

                pred = self._predict_batch(batch_u, batch_i)
                error = batch_r - pred

                np.add.at(self.user_bias, batch_u, self.lr_all * (error - self.reg_all * self.user_bias[batch_u]))
                np.add.at(self.item_bias, batch_i, self.lr_all * (error - self.reg_all * self.item_bias[batch_i]))

                uf_batch = self.user_factors[batch_u].copy()
                self.user_factors[batch_u] += self.lr_all * (
                    error[:, np.newaxis] * self.item_factors[batch_i] - self.reg_all * self.user_factors[batch_u]
                )
                self.item_factors[batch_i] += self.lr_all * (
                    error[:, np.newaxis] * uf_batch - self.reg_all * self.item_factors[batch_i]
                )

    def _compute_global_mean(self, user_item_matrix: sparse.csr_matrix) -> float:
        if user_item_matrix.nnz == 0:
            return 0.0
        return user_item_matrix.data.mean()

    def _predict_single(self, user_idx: int, item_idx: int) -> float:
        pred = self.global_mean + self.user_bias[user_idx] + self.item_bias[item_idx]
        pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return pred

    def _predict_batch(self, user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        pred = self.global_mean + self.user_bias[user_indices] + self.item_bias[item_indices]
        pred += np.sum(self.user_factors[user_indices] * self.item_factors[item_indices], axis=1)
        return pred

    def incremental_update(self, new_interactions: list):
        if not self.incremental or self.user_factors is None:
            return

        for user_idx, item_idx, rating in new_interactions:
            u = int(user_idx)
            i = int(item_idx)

            if u >= self.n_users or i >= self.n_items:
                continue

            pred = self._predict_single(u, i)
            error = rating - pred

            self.user_bias[u] += self.lr_all * (error - self.reg_all * self.user_bias[u])
            self.item_bias[i] += self.lr_all * (error - self.reg_all * self.item_bias[i])

            uf = self.user_factors[u].copy()
            self.user_factors[u] += self.lr_all * (
                error * self.item_factors[i] - self.reg_all * self.user_factors[u]
            )
            self.item_factors[i] += self.lr_all * (
                error * uf - self.reg_all * self.item_factors[i]
            )

    def add_user(self, user_idx: int):
        if self.user_factors is None:
            return

        if user_idx >= self.n_users:
            new_factors = np.random.normal(0, 0.1, (user_idx - self.n_users + 1, self.n_factors))
            new_bias = np.zeros(user_idx - self.n_users + 1)
            self.user_factors = np.vstack([self.user_factors, new_factors])
            self.user_bias = np.concatenate([self.user_bias, new_bias])
            self.n_users = user_idx + 1

    def add_item(self, item_idx: int):
        if self.item_factors is None:
            return

        if item_idx >= self.n_items:
            new_factors = np.random.normal(0, 0.1, (item_idx - self.n_items + 1, self.n_factors))
            new_bias = np.zeros(item_idx - self.n_items + 1)
            self.item_factors = np.vstack([self.item_factors, new_factors])
            self.item_bias = np.concatenate([self.item_bias, new_bias])
            self.n_items = item_idx + 1

    def recommend(self, user_idx: int, n_items: int = 10,
                  exclude_seen: bool = True,
                  seen_items: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.user_factors is None or user_idx >= self.n_users:
            return np.array([]), np.array([])

        predictions = self.global_mean + self.user_bias[user_idx] + self.item_bias
        predictions += np.dot(self.user_factors[user_idx], self.item_factors.T)

        if exclude_seen and seen_items is not None and len(seen_items) > 0:
            predictions[seen_items] = -np.inf

        top_indices = np.argsort(predictions)[::-1][:n_items]
        top_scores = predictions[top_indices]

        valid = top_scores > -np.inf
        return top_indices[valid], top_scores[valid]

    def predict(self, user_idx: int, item_idx: int) -> float:
        if self.user_factors is None:
            return self.global_mean
        if user_idx >= self.n_users or item_idx >= self.n_items:
            return self.global_mean
        return self._predict_single(user_idx, item_idx)
