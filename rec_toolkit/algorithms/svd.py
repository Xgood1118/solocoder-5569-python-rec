import numpy as np
from scipy import sparse
from typing import Tuple, Optional


class SVDRecommender:
    def __init__(self, n_factors: int = 100, n_epochs: int = 20,
                 lr_all: float = 0.005, reg_all: float = 0.02,
                 incremental: bool = True):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.incremental = incremental

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
        n_ratings = len(ratings)
        if n_ratings == 0:
            return

        rows = rows.astype(np.intp)
        cols = cols.astype(np.intp)
        ratings = ratings.astype(np.float64)

        for epoch in range(self.n_epochs):
            perm = np.random.permutation(n_ratings)
            r_rows = rows[perm]
            r_cols = cols[perm]
            r_vals = ratings[perm]

            u_f = self.user_factors[r_rows]
            i_f = self.item_factors[r_cols]
            u_b = self.user_bias[r_rows]
            i_b = self.item_bias[r_cols]

            preds = self.global_mean + u_b + i_b + np.sum(u_f * i_f, axis=1)
            errors = r_vals - preds

            self.user_bias[r_rows] += self.lr_all * (errors - self.reg_all * u_b)
            self.item_bias[r_cols] += self.lr_all * (errors - self.reg_all * i_b)

            u_f_grad = errors[:, np.newaxis] * i_f - self.reg_all * u_f
            i_f_grad = errors[:, np.newaxis] * u_f - self.reg_all * i_f

            np.add.at(self.user_factors, r_rows, self.lr_all * u_f_grad)
            np.add.at(self.item_factors, r_cols, self.lr_all * i_f_grad)

    def _compute_global_mean(self, user_item_matrix: sparse.csr_matrix) -> float:
        if user_item_matrix.nnz == 0:
            return 0.0
        return float(user_item_matrix.data.mean())

    def _predict_single(self, user_idx: int, item_idx: int) -> float:
        pred = self.global_mean + self.user_bias[user_idx] + self.item_bias[item_idx]
        pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return pred

    def incremental_update(self, new_interactions: list):
        if not self.incremental or self.user_factors is None:
            return

        for user_id, item_id, rating in new_interactions:
            if isinstance(user_id, str):
                continue
            if isinstance(item_id, str):
                continue
            u = int(user_id)
            i = int(item_id)

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
