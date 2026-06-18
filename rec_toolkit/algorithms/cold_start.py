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

        def _match_by_interests(interests: List[str], exclude: List[str], n: int) -> Tuple[List[str], List[str]]:
            matched = []
            matched_reasons = []
            used_items = set(exclude)

            for interest in interests:
                if len(matched) >= n:
                    break
                interest_lower = interest.lower()
                for item_id, profile in (item_profiles or {}).items():
                    if len(matched) >= n:
                        break
                    if item_id in used_items:
                        continue
                    hit = False
                    hit_field = ''
                    if profile.category and interest_lower in str(profile.category).lower():
                        hit = True
                        hit_field = f'分类「{profile.category}」'
                    if not hit:
                        for tag in profile.tags or []:
                            if interest_lower in str(tag).lower():
                                hit = True
                                hit_field = f'标签「{tag}」'
                                break
                    if hit:
                        matched.append(item_id)
                        used_items.add(item_id)
                        matched_reasons.append(f'根据你感兴趣的「{interest}」{hit_field}推荐')

            return matched, matched_reasons

        def _gen_interest_reason(interests: List[str]) -> str:
            if interests:
                preview = '、'.join(interests[:2])
                if len(interests) > 2:
                    preview += '等'
                return f'新人专享：匹配你的兴趣「{preview}」'
            return '新人专享推荐'

        if self.config.new_user_strategy == 'popular_then_interest':
            n_popular = int(n_items * 0.6)
            n_interest = n_items - n_popular

            popular_part = popular_items[:n_popular]
            item_ids.extend(popular_part)
            reasons.extend(['全网热门'] * len(popular_part))

            interest_items = []
            interest_reasons = []

            if content_recommender is not None and user_profile.interests:
                interest_items, _ = content_recommender.recommend(
                    user_profile, n_interest, exclude_items=item_ids
                )
                if interest_items:
                    for iid in interest_items:
                        matched_interest = None
                        for interest in user_profile.interests:
                            profile = item_profiles.get(iid) if item_profiles else None
                            if profile:
                                if profile.category and interest.lower() in str(profile.category).lower():
                                    matched_interest = interest
                                    break
                                for tag in profile.tags or []:
                                    if interest.lower() in str(tag).lower():
                                        matched_interest = interest
                                        break
                        if matched_interest:
                            interest_reasons.append(f'根据你感兴趣的「{matched_interest}」推荐')
                        else:
                            interest_reasons.append(_gen_interest_reason(user_profile.interests))

            if len(interest_items) < n_interest and user_profile.interests and item_profiles:
                remaining_need = n_interest - len(interest_items)
                extra_items, extra_reasons = _match_by_interests(
                    user_profile.interests,
                    exclude=item_ids + interest_items,
                    n=remaining_need
                )
                interest_items.extend(extra_items)
                interest_reasons.extend(extra_reasons)

            if len(interest_items) < n_interest and user_profile.interests:
                remaining_need = n_interest - len(interest_items)
                extra_popular = [p for p in popular_items if p not in item_ids and p not in interest_items]
                interest_items.extend(extra_popular[:remaining_need])
                interest_reasons.extend([_gen_interest_reason(user_profile.interests)] * min(remaining_need, len(extra_popular)))

            item_ids.extend(interest_items)
            reasons.extend(interest_reasons)

            remaining = n_items - len(item_ids)
            if remaining > 0:
                extra_popular = [p for p in popular_items if p not in item_ids]
                extra = extra_popular[:remaining]
                item_ids.extend(extra)
                reasons.extend(['全网热门'] * len(extra))

        elif self.config.new_user_strategy == 'popular_only':
            item_ids = popular_items[:n_items]
            reasons = ['全网热门'] * len(item_ids)

        elif self.config.new_user_strategy == 'interest_only':
            if content_recommender is not None and user_profile.interests:
                item_ids, _ = content_recommender.recommend(user_profile, n_items)
                reasons = [_gen_interest_reason(user_profile.interests)] * len(item_ids)
            if len(item_ids) < n_items and user_profile.interests and item_profiles:
                remaining_need = n_items - len(item_ids)
                extra_items, extra_reasons = _match_by_interests(
                    user_profile.interests,
                    exclude=item_ids,
                    n=remaining_need
                )
                item_ids.extend(extra_items)
                reasons.extend(extra_reasons)
            if len(item_ids) < n_items:
                remaining_need = n_items - len(item_ids)
                extra_popular = [p for p in popular_items if p not in item_ids]
                item_ids.extend(extra_popular[:remaining_need])
                tag = '全网热门' if not user_profile.interests else _gen_interest_reason(user_profile.interests)
                reasons.extend([tag] * min(remaining_need, len(extra_popular)))

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
