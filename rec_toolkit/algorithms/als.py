import numpy as np
from scipy import sparse
from typing import Tuple, Optional


class ALSRecommender:
    def __init__(self, n_factors: int = 100, n_epochs: int = 5,
                 reg: float = 0.1, alpha: float = 40, implicit: bool = True):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.alpha = alpha
        self.implicit = implicit

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.n_users: int = 0
        self.n_items: int = 0

    def fit(self, user_item_matrix: sparse.csr_matrix):
        self.n_users, self.n_items = user_item_matrix.shape

        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))

        if self.implicit:
            self._fit_implicit(user_item_matrix)
        else:
            self._fit_explicit(user_item_matrix)

    def _fit_explicit(self, user_item_matrix: sparse.csr_matrix):
        item_user_matrix = user_item_matrix.T.tocsr()
        for epoch in range(self.n_epochs):
            self._als_step_users_explicit(user_item_matrix)
            self._als_step_items_explicit(item_user_matrix)

    def _als_step_users_explicit(self, user_item_matrix: sparse.csr_matrix):
        YtY = self.item_factors.T @ self.item_factors
        reg_I = self.reg * np.eye(self.n_factors)

        for u in range(self.n_users):
            row = user_item_matrix.getrow(u)
            if row.nnz == 0:
                continue

            indices = row.indices
            ratings = row.data

            Y_u = self.item_factors[indices]
            self.user_factors[u] = np.linalg.solve(
                YtY + Y_u.T @ Y_u + reg_I,
                Y_u.T @ ratings
            )

    def _als_step_items_explicit(self, item_user_matrix: sparse.csr_matrix):
        XtX = self.user_factors.T @ self.user_factors
        reg_I = self.reg * np.eye(self.n_factors)

        for i in range(self.n_items):
            row = item_user_matrix.getrow(i)
            if row.nnz == 0:
                continue

            indices = row.indices
            ratings = row.data

            X_i = self.user_factors[indices]
            self.item_factors[i] = np.linalg.solve(
                XtX + X_i.T @ X_i + reg_I,
                X_i.T @ ratings
            )

    def _fit_implicit(self, user_item_matrix: sparse.csr_matrix):
        item_user_matrix = user_item_matrix.T.tocsr()
        for epoch in range(self.n_epochs):
            self._als_step_users_implicit(user_item_matrix)
            self._als_step_items_implicit(item_user_matrix)

    def _als_step_users_implicit(self, user_item_matrix: sparse.csr_matrix):
        YtY = self.item_factors.T @ self.item_factors
        reg_I = self.reg * np.eye(self.n_factors)

        for u in range(self.n_users):
            row = user_item_matrix.getrow(u)
            indices = row.indices

            if len(indices) == 0:
                self.user_factors[u] = np.zeros(self.n_factors)
                continue

            cu_values = 1.0 + self.alpha * row.data
            cu_minus_1 = cu_values - 1.0

            Y_indices = self.item_factors[indices]
            Y_Cu_Y = YtY + Y_indices.T @ (cu_minus_1[:, np.newaxis] * Y_indices)
            Y_Cu_pu = Y_indices.T @ cu_values

            self.user_factors[u] = np.linalg.solve(Y_Cu_Y + reg_I, Y_Cu_pu)

    def _als_step_items_implicit(self, item_user_matrix: sparse.csr_matrix):
        XtX = self.user_factors.T @ self.user_factors
        reg_I = self.reg * np.eye(self.n_factors)

        for i in range(self.n_items):
            row = item_user_matrix.getrow(i)
            indices = row.indices

            if len(indices) == 0:
                self.item_factors[i] = np.zeros(self.n_factors)
                continue

            ci_values = 1.0 + self.alpha * row.data
            ci_minus_1 = ci_values - 1.0

            X_indices = self.user_factors[indices]
            X_Ci_X = XtX + X_indices.T @ (ci_minus_1[:, np.newaxis] * X_indices)
            X_Ci_pi = X_indices.T @ ci_values

            self.item_factors[i] = np.linalg.solve(X_Ci_X + reg_I, X_Ci_pi)

    def recommend(self, user_idx: int, n_items: int = 10,
                  exclude_seen: bool = True,
                  seen_items: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.user_factors is None or user_idx >= self.n_users:
            return np.array([]), np.array([])

        scores = np.dot(self.user_factors[user_idx], self.item_factors.T)

        if self.implicit:
            scores = 1.0 / (1.0 + np.exp(-scores))

        if exclude_seen and seen_items is not None and len(seen_items) > 0:
            scores[seen_items] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n_items]
        top_scores = scores[top_indices]

        valid = top_scores > -np.inf
        return top_indices[valid], top_scores[valid]

    def predict(self, user_idx: int, item_idx: int) -> float:
        if self.user_factors is None:
            return 0.0
        if user_idx >= self.n_users or item_idx >= self.n_items:
            return 0.0
        score = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        if self.implicit:
            return 1.0 / (1.0 + np.exp(-score))
        return score
