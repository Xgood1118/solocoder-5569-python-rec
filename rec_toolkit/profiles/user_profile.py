import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field


@dataclass
class UserProfile:
    user_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    history_items: List[str] = field(default_factory=list)
    history_ratings: Dict[str, float] = field(default_factory=dict)
    category_preferences: Dict[str, float] = field(default_factory=dict)
    tag_preferences: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    is_new: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'age': self.age,
            'gender': self.gender,
            'location': self.location,
            'interests': self.interests,
            'history_count': len(self.history_items),
            'category_preferences': self.category_preferences,
            'tag_preferences': self.tag_preferences,
            'is_new': self.is_new,
            'extra': self.extra,
        }


class UserProfileBuilder:
    def __init__(self, dataset):
        self.dataset = dataset
        self.profiles: Dict[str, UserProfile] = {}

    def build_all(self):
        self.profiles = {}
        for _, row in self.dataset.users.iterrows():
            profile = self._build_from_row(row)
            self.profiles[profile.user_id] = profile

        self._build_from_interactions()

    def _build_from_row(self, row: pd.Series) -> UserProfile:
        profile = UserProfile(user_id=str(row['user_id']))

        if 'age' in row and pd.notna(row['age']):
            profile.age = int(row['age'])
        if 'gender' in row and pd.notna(row['gender']):
            profile.gender = str(row['gender'])
        if 'location' in row and pd.notna(row['location']):
            profile.location = str(row['location'])
        if 'interests' in row and pd.notna(row['interests']):
            interests = str(row['interests'])
            profile.interests = [i.strip() for i in interests.split(',') if i.strip()]
        if 'extra' in row and pd.notna(row['extra']):
            try:
                import json
                profile.extra = json.loads(str(row['extra']))
            except (json.JSONDecodeError, TypeError):
                profile.extra = {'raw': str(row['extra'])}

        return profile

    def _build_from_interactions(self):
        items_df = self.dataset.items.set_index('item_id') if 'item_id' in self.dataset.items.columns else pd.DataFrame()

        for _, row in self.dataset.interactions.iterrows():
            user_id = str(row['user_id'])
            item_id = str(row['item_id'])
            rating = float(row.get('rating', 1.0))

            if user_id not in self.profiles:
                self.profiles[user_id] = UserProfile(user_id=user_id)

            profile = self.profiles[user_id]
            profile.is_new = False
            profile.history_items.append(item_id)
            profile.history_ratings[item_id] = rating

            if not items_df.empty and item_id in items_df.index:
                item_row = items_df.loc[item_id]

                if 'category' in item_row and pd.notna(item_row['category']):
                    cat = str(item_row['category'])
                    profile.category_preferences[cat] = profile.category_preferences.get(cat, 0) + rating

                if 'tags' in item_row and pd.notna(item_row['tags']):
                    tags = [t.strip() for t in str(item_row['tags']).split(',') if t.strip()]
                    for tag in tags:
                        profile.tag_preferences[tag] = profile.tag_preferences.get(tag, 0) + rating

    def get_profile(self, user_id: str) -> UserProfile:
        if user_id in self.profiles:
            return self.profiles[user_id]

        if user_id in self.dataset.user2idx:
            profile = UserProfile(user_id=user_id)
            self._build_single_from_interactions(profile)
            self.profiles[user_id] = profile
            return profile

        return UserProfile(user_id=user_id, is_new=True)

    def _build_single_from_interactions(self, profile: UserProfile):
        user_id = profile.user_id
        user_interactions = self.dataset.interactions[
            self.dataset.interactions['user_id'].astype(str) == user_id
        ]

        items_df = self.dataset.items.set_index('item_id') if 'item_id' in self.dataset.items.columns else pd.DataFrame()

        for _, row in user_interactions.iterrows():
            item_id = str(row['item_id'])
            rating = float(row.get('rating', 1.0))

            profile.history_items.append(item_id)
            profile.history_ratings[item_id] = rating
            profile.is_new = False

            if not items_df.empty and item_id in items_df.index:
                item_row = items_df.loc[item_id]

                if 'category' in item_row and pd.notna(item_row['category']):
                    cat = str(item_row['category'])
                    profile.category_preferences[cat] = profile.category_preferences.get(cat, 0) + rating

                if 'tags' in item_row and pd.notna(item_row['tags']):
                    tags = [t.strip() for t in str(item_row['tags']).split(',') if t.strip()]
                    for tag in tags:
                        profile.tag_preferences[tag] = profile.tag_preferences.get(tag, 0) + rating

    def add_field(self, field_name: str, field_type: str = 'str'):
        pass

    def update_profile(self, user_id: str, updates: Dict[str, Any]):
        profile = self.get_profile(user_id)
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
            else:
                profile.extra[key] = value
