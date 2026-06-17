import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta


class PopularRecommender:
    def __init__(self, half_life_days: int = 30, time_granularity: str = 'day',
                 top_k: int = 100):
        self.half_life_days = half_life_days
        self.time_granularity = time_granularity
        self.top_k = top_k
        self.item_scores: Dict[str, float] = {}
        self.item_popularity: List[Tuple[str, float]] = []

    def _get_decay_factor(self, days: float) -> float:
        if self.time_granularity == 'day':
            half_life = self.half_life_days
        elif self.time_granularity == 'week':
            half_life = self.half_life_days * 7
        elif self.time_granularity == 'month':
            half_life = self.half_life_days * 30
        else:
            half_life = self.half_life_days

        return 0.5 ** (days / half_life) if half_life > 0 else 1.0

    def fit(self, interactions_df: pd.DataFrame, items_df: Optional[pd.DataFrame] = None,
            reference_time: Optional[datetime] = None):
        if reference_time is None:
            if 'timestamp' in interactions_df.columns and interactions_df['timestamp'].notna().any():
                reference_time = pd.to_datetime(interactions_df['timestamp']).max()
            else:
                reference_time = datetime.now()

        self.item_scores = {}

        for _, row in interactions_df.iterrows():
            item_id = str(row['item_id'])
            rating = float(row.get('rating', 1.0))

            if 'timestamp' in row and pd.notna(row['timestamp']):
                try:
                    ts = pd.to_datetime(row['timestamp'])
                    days = (reference_time - ts).total_seconds() / 86400.0
                    decay = self._get_decay_factor(days)
                except (ValueError, TypeError):
                    decay = 1.0
            else:
                decay = 1.0

            self.item_scores[item_id] = self.item_scores.get(item_id, 0) + rating * decay

        if items_df is not None and 'popularity' in items_df.columns:
            for _, row in items_df.iterrows():
                item_id = str(row['item_id'])
                if item_id not in self.item_scores:
                    self.item_scores[item_id] = float(row.get('popularity', 0))

        self.item_popularity = sorted(
            self.item_scores.items(), key=lambda x: x[1], reverse=True
        )

    def recommend(self, n_items: int = 10,
                  exclude_items: Optional[List[str]] = None) -> Tuple[List[str], List[float]]:
        results = []
        scores = []

        exclude_set = set(exclude_items) if exclude_items else set()

        for item_id, score in self.item_popularity:
            if item_id in exclude_set:
                continue
            results.append(item_id)
            scores.append(score)
            if len(results) >= n_items:
                break

        return results, scores

    def get_popular_items(self, n: int = 100) -> List[str]:
        items, _ = self.recommend(n)
        return items

    def get_item_score(self, item_id: str) -> float:
        return self.item_scores.get(item_id, 0.0)
