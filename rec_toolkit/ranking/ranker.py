import numpy as np
from typing import List, Dict, Optional, Any
from ..recall import RecallResult
from dataclasses import dataclass


@dataclass
class RankedResult:
    item_id: str
    final_score: float
    recall_score: float
    context_score: float
    channels: List[str]
    reason: str = ''


class Ranker:
    def __init__(self, method: str = 'weighted'):
        self.method = method
        self.context_weights: Dict[str, float] = {
            'time': 0.1,
            'location': 0.1,
            'device': 0.05,
        }

    def rank(self, recall_results: List[RecallResult],
             context: Optional[Dict] = None,
             item_profiles: Optional[Dict] = None,
             user_profile=None) -> List[RankedResult]:
        if not recall_results:
            return []

        merged: Dict[str, RankedResult] = {}

        for result in recall_results:
            context_score = self._compute_context_score(result, context, item_profiles)
            recall_score = result.score

            if self.method == 'weighted':
                final_score = recall_score + context_score
            elif self.method == 'multiply':
                final_score = recall_score * (1 + context_score)
            else:
                final_score = recall_score

            if result.item_id in merged:
                existing = merged[result.item_id]
                existing.final_score += final_score
                existing.recall_score += recall_score
                existing.context_score += context_score
                for ch in result.channel.split(','):
                    if ch and ch not in existing.channels:
                        existing.channels.append(ch)
            else:
                merged[result.item_id] = RankedResult(
                    item_id=result.item_id,
                    final_score=final_score,
                    recall_score=recall_score,
                    context_score=context_score,
                    channels=[ch for ch in result.channel.split(',') if ch],
                    reason=result.reason,
                )

        ranked = list(merged.values())
        ranked.sort(key=lambda x: x.final_score, reverse=True)

        return ranked

    def _compute_context_score(self, recall_result: RecallResult,
                               context: Optional[Dict],
                               item_profiles: Optional[Dict]) -> float:
        if not context:
            return 0.0

        score = 0.0

        if 'time' in context and self.context_weights.get('time', 0) > 0:
            score += self._time_context_score(context['time'], item_profiles,
                                              recall_result.item_id) * self.context_weights['time']

        if 'location' in context and self.context_weights.get('location', 0) > 0:
            score += self._location_context_score(context['location'], item_profiles,
                                                  recall_result.item_id) * self.context_weights['location']

        if 'device' in context and self.context_weights.get('device', 0) > 0:
            score += self._device_context_score(context['device']) * self.context_weights['device']

        return score

    def _time_context_score(self, time_context: str, item_profiles: Optional[Dict],
                            item_id: str) -> float:
        if item_profiles and item_id in item_profiles:
            item_profile = item_profiles[item_id]
            if hasattr(item_profile, 'extra') and 'time_preference' in item_profile.extra:
                time_prefs = item_profile.extra['time_preference']
                if time_context in time_prefs:
                    return 0.5
        return 0.0

    def _location_context_score(self, location: str, item_profiles: Optional[Dict],
                                item_id: str) -> float:
        if item_profiles and item_id in item_profiles:
            item_profile = item_profiles[item_id]
            if hasattr(item_profile, 'extra') and 'locations' in item_profile.extra:
                locations = item_profile.extra['locations']
                if location in locations:
                    return 0.8
        return 0.0

    def _device_context_score(self, device: str) -> float:
        device_factor = {
            'mobile': 0.1,
            'desktop': 0.0,
            'tablet': 0.05,
        }
        return device_factor.get(device, 0.0)

    def dedup(self, results: List[RankedResult]) -> List[RankedResult]:
        seen = set()
        unique = []
        for r in results:
            if r.item_id not in seen:
                seen.add(r.item_id)
                unique.append(r)
        return unique
