import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, List


DEFAULT_CONFIG = {
    'data': {
        'data_dir': 'data',
        'user_file': 'users.csv',
        'item_file': 'items.csv',
        'interaction_file': 'interactions.csv',
    },
    'cf': {
        'top_k': 50,
        'similarity': 'cosine',
        'min_common_items': 3,
    },
    'svd': {
        'n_factors': 100,
        'n_epochs': 20,
        'lr_all': 0.005,
        'reg_all': 0.02,
        'incremental': True,
    },
    'als': {
        'n_factors': 100,
        'n_epochs': 20,
        'reg': 0.1,
        'alpha': 40,
        'implicit': True,
    },
    'content': {
        'method': 'tfidf',
        'embedding_dim': 128,
        'pretrained_model': None,
        'top_k': 50,
    },
    'popular': {
        'half_life_days': 30,
        'time_granularity': 'day',
        'top_k': 100,
    },
    'cold_start': {
        'new_user_strategy': 'popular_then_interest',
        'new_item_strategy': 'content_similar',
        'force_exposure_ratio': 0.1,
    },
    'recall': {
        'channels': {
            'user_cf': 0.3,
            'item_cf': 0.3,
            'content': 0.2,
            'popular': 0.1,
            'svd': 0.1,
        },
        'top_n': 50,
    },
    'bandit': {
        'strategy': 'epsilon_greedy',
        'epsilon': 0.1,
        'ucb_c': 2.0,
    },
    'evaluation': {
        'top_ks': [5, 10, 20],
        'metrics': ['precision', 'recall', 'f1', 'map', 'ndcg', 'hit_rate', 'coverage', 'diversity'],
    },
    'association': {
        'algorithm': 'apriori',
        'min_support': 0.01,
        'min_confidence': 0.5,
        'output_file': 'association_rules.txt',
    },
    'ab_test': {
        'enabled': False,
        'buckets': ['A', 'B'],
        'bucket_configs': {},
    },
    'scheduler': {
        'incremental_interval': 3600,
        'retrain_cron': '0 2 * * *',
    },
    'server': {
        'host': '0.0.0.0',
        'port': 5000,
        'debug': True,
    },
}


@dataclass
class Config:
    _config: Dict[str, Any] = field(default_factory=lambda: DEFAULT_CONFIG.copy())

    def __getitem__(self, key):
        return self._config[key]

    def get(self, key, default=None):
        return self._config.get(key, default)

    def update(self, config_dict: Dict[str, Any]):
        def deep_update(base, updates):
            for k, v in updates.items():
                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    deep_update(base[k], v)
                else:
                    base[k] = v

        deep_update(self._config, config_dict)

    def load_yaml(self, path: str):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            if config_dict:
                self.update(config_dict)

    def save_yaml(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)


config = Config()
