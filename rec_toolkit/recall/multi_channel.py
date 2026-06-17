from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class RecallResult:
    item_id: str
    score: float
    channel: str
    reason: str = ''


class MultiChannelRecall:
    def __init__(self, channels: Optional[Dict[str, float]] = None, top_n: int = 50):
        self.channels = channels or {
            'user_cf': 0.3,
            'item_cf': 0.3,
            'content': 0.2,
            'popular': 0.1,
            'svd': 0.1,
        }
        self.top_n = top_n
        self.recall_funcs: Dict[str, callable] = {}

    def register_channel(self, name: str, weight: float, func: callable):
        self.channels[name] = weight
        self.recall_funcs[name] = func

    def recall(self, user_id: str, user_idx: Optional[int] = None,
               n_items: int = 50, context: Optional[Dict] = None) -> List[RecallResult]:
        item_channels: Dict[str, Dict[str, Any]] = {}

        for channel_name, weight in self.channels.items():
            if channel_name not in self.recall_funcs:
                continue

            try:
                items, scores = self.recall_funcs[channel_name](
                    user_id, user_idx=user_idx, n_items=n_items, context=context
                )

                for i, item_id in enumerate(items):
                    score = scores[i] if i < len(scores) else 0.0
                    weighted_score = score * weight

                    if item_id in item_channels:
                        item_channels[item_id]['channels'].append(channel_name)
                        item_channels[item_id]['scores'].append((channel_name, weighted_score))
                        item_channels[item_id]['total_score'] += weighted_score
                    else:
                        item_channels[item_id] = {
                            'channels': [channel_name],
                            'scores': [(channel_name, weighted_score)],
                            'total_score': weighted_score,
                            'best_channel': channel_name,
                            'best_score': weighted_score,
                        }

                    if weighted_score > item_channels[item_id]['best_score']:
                        item_channels[item_id]['best_channel'] = channel_name
                        item_channels[item_id]['best_score'] = weighted_score

            except Exception as e:
                print(f"召回通道 {channel_name} 执行失败: {e}")
                continue

        results = []
        for item_id, info in item_channels.items():
            results.append(RecallResult(
                item_id=item_id,
                score=info['total_score'],
                channel=','.join(info['channels']),
                reason='',
            ))

        results.sort(key=lambda x: x.score, reverse=True)

        return results[:n_items]

    def update_channel_weights(self, weights: Dict[str, float]):
        self.channels.update(weights)

    def get_channel_weights(self) -> Dict[str, float]:
        return self.channels.copy()
