import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ItemProfile:
    item_id: str
    title: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None
    popularity: float = 0.0
    create_time: Optional[str] = None
    feature_vector: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    is_new: bool = True
    interaction_count: int = 0
    avg_rating: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'item_id': self.item_id,
            'title': self.title,
            'category': self.category,
            'tags': self.tags,
            'popularity': self.popularity,
            'interaction_count': self.interaction_count,
            'avg_rating': self.avg_rating,
            'is_new': self.is_new,
            'extra': self.extra,
        }


class ItemProfileBuilder:
    def __init__(self, dataset):
        self.dataset = dataset
        self.profiles: Dict[str, ItemProfile] = {}

    def build_all(self):
        self.profiles = {}
        for _, row in self.dataset.items.iterrows():
            profile = self._build_from_row(row)
            self.profiles[profile.item_id] = profile

        self._build_from_interactions()

    def _build_from_row(self, row: pd.Series) -> ItemProfile:
        profile = ItemProfile(item_id=str(row['item_id']))

        if 'title' in row and pd.notna(row['title']):
            profile.title = str(row['title'])
        if 'category' in row and pd.notna(row['category']):
            profile.category = str(row['category'])
        if 'tags' in row and pd.notna(row['tags']):
            tags = str(row['tags'])
            profile.tags = [t.strip() for t in tags.split(',') if t.strip()]
        if 'description' in row and pd.notna(row['description']):
            profile.description = str(row['description'])
        if 'popularity' in row and pd.notna(row['popularity']):
            profile.popularity = float(row['popularity'])
        if 'create_time' in row and pd.notna(row['create_time']):
            profile.create_time = str(row['create_time'])
        if 'extra' in row and pd.notna(row['extra']):
            try:
                import json
                profile.extra = json.loads(str(row['extra']))
            except (json.JSONDecodeError, TypeError):
                profile.extra = {'raw': str(row['extra'])}

        return profile

    def _build_from_interactions(self):
        item_stats: Dict[str, Dict[str, float]] = {}

        for _, row in self.dataset.interactions.iterrows():
            item_id = str(row['item_id'])
            rating = float(row.get('rating', 1.0))

            if item_id not in item_stats:
                item_stats[item_id] = {'count': 0, 'sum_rating': 0.0}

            item_stats[item_id]['count'] += 1
            item_stats[item_id]['sum_rating'] += rating

        for item_id, stats in item_stats.items():
            if item_id not in self.profiles:
                self.profiles[item_id] = ItemProfile(item_id=item_id)

            profile = self.profiles[item_id]
            profile.is_new = False
            profile.interaction_count = int(stats['count'])
            profile.avg_rating = stats['sum_rating'] / stats['count'] if stats['count'] > 0 else 0.0

            if profile.popularity == 0.0:
                profile.popularity = stats['count']

    def get_profile(self, item_id: str) -> ItemProfile:
        if item_id in self.profiles:
            return self.profiles[item_id]

        if item_id in self.dataset.item2idx:
            profile = ItemProfile(item_id=item_id)
            self._build_single_from_interactions(profile)
            self.profiles[item_id] = profile
            return profile

        return ItemProfile(item_id=item_id, is_new=True)

    def _build_single_from_interactions(self, profile: ItemProfile):
        item_id = profile.item_id
        item_interactions = self.dataset.interactions[
            self.dataset.interactions['item_id'].astype(str) == item_id
        ]

        if len(item_interactions) > 0:
            profile.is_new = False
            profile.interaction_count = len(item_interactions)
            if 'rating' in item_interactions.columns:
                profile.avg_rating = item_interactions['rating'].mean()
            profile.popularity = max(profile.popularity, len(item_interactions))

    def get_all_items(self) -> List[ItemProfile]:
        return list(self.profiles.values())

    def get_items_by_category(self, category: str) -> List[ItemProfile]:
        return [p for p in self.profiles.values() if p.category == category]

    def get_items_by_tag(self, tag: str) -> List[ItemProfile]:
        return [p for p in self.profiles.values() if tag in p.tags]

    def get_popular_items(self, top_n: int = 100) -> List[ItemProfile]:
        sorted_items = sorted(
            self.profiles.values(),
            key=lambda x: x.popularity,
            reverse=True
        )
        return sorted_items[:top_n]

    def update_profile(self, item_id: str, updates: Dict[str, Any]):
        profile = self.get_profile(item_id)
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
            else:
                profile.extra[key] = value
