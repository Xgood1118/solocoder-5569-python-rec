import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ColdStartConfig:
    new_user_strategy: str = 'popular_then_interest'
    new_item_strategy: str = 'content_similar'
    force_exposure_ratio: float = 0.1


class ColdStartHandler:
    def __init__(self, config: Optional[ColdStartConfig] = None):
        self.config = config or ColdStartConfig()

    def handle_new_user(self, user_profile, popular_items: List[str],
                        content_recommender=None, n_items: int = 10,
                        item_profiles: Optional[Dict] = None) -> Tuple[List[str], List[str]]:
        item_ids = []
        reasons = []

        if self.config.new_user_strategy == 'popular_then_interest':
            n_popular = int(n_items * 0.6)
            n_interest = n_items - n_popular

            popular_part = popular_items[:n_popular]
            item_ids.extend(popular_part)
            for pid in popular_part:
                if item_profiles and pid in item_profiles:
                    p = item_profiles[pid]
                    cat = getattr(p, 'category', None) or ''
                    title = getattr(p, 'title', None) or ''
                    if title:
                        reasons.append(f'全网热门：{title}')
                    elif cat:
                        reasons.append(f'热门{cat}')
                    else:
                        reasons.append('全网热门')
                else:
                    reasons.append('全网热门')

            if content_recommender is not None and hasattr(user_profile, 'interests') and user_profile.interests:
                interest_items, _ = content_recommender.recommend(
                    user_profile, n_interest, exclude_items=item_ids
                )
                for iid in interest_items:
                    item_ids.append(iid)
                    interest_label = user_profile.interests[0] if user_profile.interests else ''
                    if item_profiles and iid in item_profiles:
                        p = item_profiles[iid]
                        title = getattr(p, 'title', None) or ''
                        cat = getattr(p, 'category', None) or ''
                        if title:
                            reasons.append(f'因为你对「{interest_label}」感兴趣 → {title}')
                        elif cat:
                            reasons.append(f'兴趣匹配{cat}：{interest_label}')
                        else:
                            reasons.append(f'兴趣匹配：{interest_label}')
                    else:
                        reasons.append(f'兴趣匹配：{interest_label}')

            remaining = n_items - len(item_ids)
            if remaining > 0 and len(popular_items) > n_popular:
                extra = popular_items[n_popular:n_popular + remaining]
                for pid in extra:
                    item_ids.append(pid)
                    if item_profiles and pid in item_profiles:
                        p = item_profiles[pid]
                        title = getattr(p, 'title', None) or ''
                        if title:
                            reasons.append(f'热门推荐：{title}')
                        else:
                            reasons.append('热门推荐')
                    else:
                        reasons.append('热门推荐')

        elif self.config.new_user_strategy == 'popular_only':
            item_ids = popular_items[:n_items]
            for pid in item_ids:
                if item_profiles and pid in item_profiles:
                    p = item_profiles[pid]
                    title = getattr(p, 'title', None) or ''
                    reasons.append(f'全网热门：{title}' if title else '全网热门')
                else:
                    reasons.append('全网热门')

        elif self.config.new_user_strategy == 'interest_only':
            if content_recommender is not None and hasattr(user_profile, 'interests') and user_profile.interests:
                interest_items, _ = content_recommender.recommend(user_profile, n_items)
                for iid in interest_items:
                    item_ids.append(iid)
                    interest_label = user_profile.interests[0] if user_profile.interests else ''
                    if item_profiles and iid in item_profiles:
                        p = item_profiles[iid]
                        title = getattr(p, 'title', None) or ''
                        reasons.append(f'兴趣推荐：{title}' if title else f'兴趣匹配：{interest_label}')
                    else:
                        reasons.append(f'兴趣匹配：{interest_label}')
            else:
                item_ids = popular_items[:n_items]
                reasons = ['全网热门'] * len(item_ids)

        return item_ids[:n_items], reasons[:n_items]

    def handle_new_item(self, item_id: str, item_profile,
                        content_recommender=None,
                        popular_items: Optional[List[str]] = None,
                        n_similar: int = 10) -> Dict:
        result = {
            'item_id': item_id,
            'strategy': self.config.new_item_strategy,
            'similar_items': [],
            'force_exposure': self.config.force_exposure_ratio,
        }

        if self.config.new_item_strategy == 'content_similar' and content_recommender:
            similar_items, scores = content_recommender.get_similar_items(item_id, n_similar)
            result['similar_items'] = list(zip(similar_items, scores))

        elif self.config.new_item_strategy == 'popular_boost':
            if popular_items:
                result['boost_position'] = int(len(popular_items) * self.config.force_exposure_ratio)

        return result

    def mix_new_items(self, recommendations: List[str], new_items: List[str],
                      ratio: Optional[float] = None) -> List[str]:
        if ratio is None:
            ratio = self.config.force_exposure_ratio

        n_new = int(len(recommendations) * ratio)
        if n_new == 0 and new_items:
            n_new = 1

        new_items_to_add = new_items[:n_new]

        mixed = []
        step = max(1, len(recommendations) // (len(new_items_to_add) + 1))

        rec_idx = 0
        new_idx = 0

        for i in range(len(recommendations) + len(new_items_to_add)):
            if new_idx < len(new_items_to_add) and (i + 1) % step == 0:
                if new_items_to_add[new_idx] not in mixed:
                    mixed.append(new_items_to_add[new_idx])
                new_idx += 1
            elif rec_idx < len(recommendations):
                if recommendations[rec_idx] not in mixed:
                    mixed.append(recommendations[rec_idx])
                rec_idx += 1

        return mixed
