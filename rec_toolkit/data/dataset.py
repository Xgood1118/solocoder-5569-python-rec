import pandas as pd
import numpy as np
from scipy import sparse
from typing import Dict, List, Optional, Tuple
from .loader import DataLoader


class Dataset:
    def __init__(self, data_dir: str = 'data'):
        self.loader = DataLoader(data_dir)
        self.users: pd.DataFrame = pd.DataFrame()
        self.items: pd.DataFrame = pd.DataFrame()
        self.interactions: pd.DataFrame = pd.DataFrame()
        self.user2idx: Dict[str, int] = {}
        self.item2idx: Dict[str, int] = {}
        self.idx2user: List[str] = []
        self.idx2item: List[str] = []
        self.user_item_matrix: Optional[sparse.csr_matrix] = None

    def load_all(self):
        self.users = self.loader.load_csv('users')
        self.items = self.loader.load_csv('items')
        self.interactions = self.loader.load_csv('interactions')
        self._build_indexes()
        self._build_user_item_matrix()

    def _build_indexes(self):
        if not self.users.empty and 'user_id' in self.users.columns:
            self.idx2user = list(self.users['user_id'].unique())
        else:
            user_ids = self.interactions['user_id'].unique() if not self.interactions.empty else []
            self.idx2user = list(user_ids)
        self.user2idx = {uid: i for i, uid in enumerate(self.idx2user)}

        if not self.items.empty and 'item_id' in self.items.columns:
            self.idx2item = list(self.items['item_id'].unique())
        else:
            item_ids = self.interactions['item_id'].unique() if not self.interactions.empty else []
            self.idx2item = list(item_ids)
        self.item2idx = {iid: i for i, iid in enumerate(self.idx2item)}

    def _build_user_item_matrix(self):
        if self.interactions.empty:
            self.user_item_matrix = sparse.csr_matrix(
                (len(self.idx2user), len(self.idx2item))
            )
            return

        df = self.interactions.copy()
        df = df[df['user_id'].isin(self.user2idx) & df['item_id'].isin(self.item2idx)]

        if df.empty:
            self.user_item_matrix = sparse.csr_matrix(
                (len(self.idx2user), len(self.idx2item))
            )
            return

        user_indices = df['user_id'].map(self.user2idx).values
        item_indices = df['item_id'].map(self.item2idx).values

        if 'rating' in df.columns:
            ratings = df['rating'].astype(float).values
        else:
            ratings = np.ones(len(df))

        self.user_item_matrix = sparse.csr_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(len(self.idx2user), len(self.idx2item))
        )

    def add_interactions(self, df: pd.DataFrame, incremental: bool = True):
        if incremental and not self.interactions.empty:
            self.interactions = pd.concat([self.interactions, df], ignore_index=True)
        else:
            self.interactions = df

        self._build_indexes()
        self._build_user_item_matrix()

    def add_users(self, df: pd.DataFrame, incremental: bool = True):
        if incremental and not self.users.empty:
            self.users = pd.concat([self.users, df], ignore_index=True)
            self.users = self.users.drop_duplicates(subset=['user_id'], keep='last')
        else:
            self.users = df

        self._build_indexes()
        self._build_user_item_matrix()

    def add_items(self, df: pd.DataFrame, incremental: bool = True):
        if incremental and not self.items.empty:
            self.items = pd.concat([self.items, df], ignore_index=True)
            self.items = self.items.drop_duplicates(subset=['item_id'], keep='last')
        else:
            self.items = df

        self._build_indexes()
        self._build_user_item_matrix()

    def get_user_items(self, user_idx: int) -> np.ndarray:
        if self.user_item_matrix is None:
            return np.array([])
        row = self.user_item_matrix.getrow(user_idx)
        return row.indices

    def get_item_users(self, item_idx: int) -> np.ndarray:
        if self.user_item_matrix is None:
            return np.array([])
        col = self.user_item_matrix.getcol(item_idx)
        return col.indices

    def get_user_rating(self, user_idx: int, item_idx: int) -> float:
        if self.user_item_matrix is None:
            return 0.0
        return self.user_item_matrix[user_idx, item_idx]

    def is_new_user(self, user_id: str) -> bool:
        return user_id not in self.user2idx

    def is_new_item(self, item_id: str) -> bool:
        return item_id not in self.item2idx

    @property
    def n_users(self) -> int:
        return len(self.idx2user)

    @property
    def n_items(self) -> int:
        return len(self.idx2item)
