import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class BucketConfig:
    name: str
    weight: float = 1.0
    config: Dict[str, Any] = field(default_factory=dict)


class ABTestManager:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.buckets: List[BucketConfig] = []
        self.total_weight: float = 0.0

    def add_bucket(self, name: str, weight: float = 1.0, config: Optional[Dict] = None):
        config = config or {}
        bucket = BucketConfig(name=name, weight=weight, config=config)
        self.buckets.append(bucket)
        self.total_weight += weight

    def set_buckets(self, buckets: List[Dict]):
        self.buckets = []
        self.total_weight = 0.0
        for b in buckets:
            self.add_bucket(b['name'], b.get('weight', 1.0), b.get('config', {}))

    def get_bucket(self, user_id: str) -> str:
        if not self.enabled or not self.buckets:
            return self.buckets[0].name if self.buckets else 'default'

        hash_value = self._hash_user_id(user_id)
        cumulative = 0.0

        for bucket in self.buckets:
            cumulative += bucket.weight
            if hash_value <= cumulative / self.total_weight:
                return bucket.name

        return self.buckets[-1].name

    def _hash_user_id(self, user_id: str) -> float:
        hash_str = f"ab_test_{user_id}"
        hash_hex = hashlib.md5(hash_str.encode('utf-8')).hexdigest()
        hash_int = int(hash_hex[:8], 16)
        return hash_int / 0xFFFFFFFF

    def get_bucket_config(self, user_id: str) -> Dict[str, Any]:
        bucket_name = self.get_bucket(user_id)
        for bucket in self.buckets:
            if bucket.name == bucket_name:
                return bucket.config
        return {}

    def is_in_bucket(self, user_id: str, bucket_name: str) -> bool:
        return self.get_bucket(user_id) == bucket_name

    def get_all_buckets(self) -> List[str]:
        return [b.name for b in self.buckets]

    def get_bucket_stats(self, user_ids: List[str]) -> Dict[str, int]:
        stats = {b.name: 0 for b in self.buckets}
        for uid in user_ids:
            bucket = self.get_bucket(uid)
            if bucket in stats:
                stats[bucket] += 1
        return stats
