import os
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from .config import config
from .data import Dataset, DataLoader
from .profiles import UserProfileBuilder, ItemProfileBuilder
from .algorithms.cf import UserCF, ItemCF
from .algorithms.svd import SVDRecommender
from .algorithms.als import ALSRecommender
from .algorithms.content import ContentBasedRecommender
from .algorithms.popular import PopularRecommender
from .algorithms.cold_start import ColdStartHandler, ColdStartConfig
from .recall import MultiChannelRecall, RecallResult
from .ranking import Ranker, RankedResult
from .evaluation import Evaluator
from .association import AssociationRuleMiner
from .bandit import BanditWeightOptimizer
from .ab_test import ABTestManager
from .explanation import ReasonGenerator
from .scheduler import Scheduler, IncrementalUpdater


@dataclass
class RecommendationItem:
    item_id: str
    score: float
    reason: str
    channel: str
    rank: int = 0


class RecommenderSystem:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.dataset = Dataset(data_dir)
        self.user_profile_builder: Optional[UserProfileBuilder] = None
        self.item_profile_builder: Optional[ItemProfileBuilder] = None

        self.user_cf: Optional[UserCF] = None
        self.item_cf: Optional[ItemCF] = None
        self.svd: Optional[SVDRecommender] = None
        self.als: Optional[ALSRecommender] = None
        self.content_rec: Optional[ContentBasedRecommender] = None
        self.popular_rec: Optional[PopularRecommender] = None
        self.cold_start: Optional[ColdStartHandler] = None
        self.association_miner: Optional[AssociationRuleMiner] = None

        self.multi_recall: Optional[MultiChannelRecall] = None
        self.ranker: Optional[Ranker] = None
        self.evaluator: Optional[Evaluator] = None
        self.bandit_optimizer: Optional[BanditWeightOptimizer] = None
        self.ab_test: Optional[ABTestManager] = None
        self.reason_generator: Optional[ReasonGenerator] = None

        self.scheduler: Optional[Scheduler] = None
        self.incremental_updater: Optional[IncrementalUpdater] = None

        self.is_trained = False

    def load_data(self):
        self.dataset.load_all()
        self.user_profile_builder = UserProfileBuilder(self.dataset)
        self.user_profile_builder.build_all()
        self.item_profile_builder = ItemProfileBuilder(self.dataset)
        self.item_profile_builder.build_all()

    def train_all(self):
        if self.dataset.user_item_matrix is None:
            self.load_data()

        cf_config = config['cf']
        self.user_cf = UserCF(
            top_k=cf_config['top_k'],
            similarity=cf_config['similarity'],
            min_common_items=cf_config['min_common_items'],
        )
        self.user_cf.fit(self.dataset.user_item_matrix)

        self.item_cf = ItemCF(
            top_k=cf_config['top_k'],
            similarity=cf_config['similarity'],
            min_common_items=cf_config['min_common_items'],
        )
        self.item_cf.fit(self.dataset.user_item_matrix)

        svd_config = config['svd']
        self.svd = SVDRecommender(
            n_factors=svd_config['n_factors'],
            n_epochs=svd_config['n_epochs'],
            lr_all=svd_config['lr_all'],
            reg_all=svd_config['reg_all'],
            incremental=svd_config['incremental'],
        )
        self.svd.fit(self.dataset.user_item_matrix)

        als_config = config['als']
        self.als = ALSRecommender(
            n_factors=als_config['n_factors'],
            n_epochs=als_config['n_epochs'],
            reg=als_config['reg'],
            alpha=als_config['alpha'],
            implicit=als_config['implicit'],
        )
        self.als.fit(self.dataset.user_item_matrix)

        content_config = config['content']
        self.content_rec = ContentBasedRecommender(
            method=content_config['method'],
            embedding_dim=content_config['embedding_dim'],
            pretrained_model=content_config['pretrained_model'],
            top_k=content_config['top_k'],
        )
        if not self.dataset.items.empty:
            self.content_rec.fit(self.dataset.items)

        popular_config = config['popular']
        self.popular_rec = PopularRecommender(
            half_life_days=popular_config['half_life_days'],
            time_granularity=popular_config['time_granularity'],
            top_k=popular_config['top_k'],
        )
        self.popular_rec.fit(self.dataset.interactions, self.dataset.items)

        cold_start_config = ColdStartConfig(**config['cold_start'])
        self.cold_start = ColdStartHandler(cold_start_config)

        assoc_config = config['association']
        self.association_miner = AssociationRuleMiner(
            algorithm=assoc_config['algorithm'],
            min_support=assoc_config['min_support'],
            min_confidence=assoc_config['min_confidence'],
            output_file=os.path.join(self.data_dir, assoc_config['output_file']),
        )
        self.association_miner.fit_from_interactions(self.dataset.interactions)

        self._setup_recall()
        self.ranker = Ranker()

        eval_config = config['evaluation']
        self.evaluator = Evaluator(top_ks=eval_config['top_ks'])

        bandit_config = config['bandit']
        recall_channels = list(config['recall']['channels'].keys())
        self.bandit_optimizer = BanditWeightOptimizer(
            channels=recall_channels,
            strategy=bandit_config['strategy'],
            epsilon=bandit_config['epsilon'],
            ucb_c=bandit_config['ucb_c'],
        )

        ab_config = config['ab_test']
        self.ab_test = ABTestManager(enabled=ab_config['enabled'])
        if ab_config['buckets']:
            for i, bucket in enumerate(ab_config['buckets']):
                weight = 1.0 / len(ab_config['buckets'])
                bucket_config = ab_config.get('bucket_configs', {}).get(bucket, {})
                self.ab_test.add_bucket(bucket, weight=weight, config=bucket_config)

        self.reason_generator = ReasonGenerator()

        self._setup_scheduler()

        self.is_trained = True

    def _setup_recall(self):
        recall_config = config['recall']
        self.multi_recall = MultiChannelRecall(
            channels=recall_config['channels'],
            top_n=recall_config['top_n'],
        )

        self.multi_recall.register_channel('user_cf', recall_config['channels'].get('user_cf', 0),
                                            self._user_cf_recall)
        self.multi_recall.register_channel('item_cf', recall_config['channels'].get('item_cf', 0),
                                            self._item_cf_recall)
        self.multi_recall.register_channel('content', recall_config['channels'].get('content', 0),
                                            self._content_recall)
        self.multi_recall.register_channel('popular', recall_config['channels'].get('popular', 0),
                                            self._popular_recall)
        self.multi_recall.register_channel('svd', recall_config['channels'].get('svd', 0),
                                            self._svd_recall)

    def _setup_scheduler(self):
        self.scheduler = Scheduler()
        self.incremental_updater = IncrementalUpdater(self)

        sched_config = config['scheduler']
        self.scheduler.add_job(
            'incremental_update',
            self.incremental_updater.update,
            interval_seconds=sched_config['incremental_interval'],
        )
        self.scheduler.add_job(
            'retrain',
            self.incremental_updater.retrain,
            cron_expr=sched_config['retrain_cron'],
        )

    def _user_cf_recall(self, user_id: str, user_idx=None, n_items=50, context=None):
        if not self.user_cf or user_idx is None:
            return [], []

        seen_items = self.dataset.get_user_items(user_idx)
        item_indices, scores = self.user_cf.recommend(
            user_idx, n_items=n_items, exclude_seen=True
        )
        item_ids = [self.dataset.idx2item[i] for i in item_indices]
        return item_ids, list(scores)

    def _item_cf_recall(self, user_id: str, user_idx=None, n_items=50, context=None):
        if not self.item_cf or user_idx is None:
            return [], []

        item_indices, scores = self.item_cf.recommend(
            user_idx, n_items=n_items, exclude_seen=True
        )
        item_ids = [self.dataset.idx2item[i] for i in item_indices]
        return item_ids, list(scores)

    def _content_recall(self, user_id: str, user_idx=None, n_items=50, context=None):
        if not self.content_rec or not self.user_profile_builder:
            return [], []

        user_profile = self.user_profile_builder.get_profile(user_id)
        exclude = list(user_profile.history_items)
        item_ids, scores = self.content_rec.recommend(
            user_profile, n_items=n_items, exclude_items=exclude
        )
        return item_ids, scores

    def _popular_recall(self, user_id: str, user_idx=None, n_items=50, context=None):
        if not self.popular_rec:
            return [], []

        user_profile = self.user_profile_builder.get_profile(user_id) if self.user_profile_builder else None
        exclude = list(user_profile.history_items) if user_profile else []
        item_ids, scores = self.popular_rec.recommend(
            n_items=n_items, exclude_items=exclude
        )
        return item_ids, scores

    def _svd_recall(self, user_id: str, user_idx=None, n_items=50, context=None):
        if not self.svd or user_idx is None:
            return [], []

        seen_items = self.dataset.get_user_items(user_idx)
        item_indices, scores = self.svd.recommend(
            user_idx, n_items=n_items, exclude_seen=True, seen_items=seen_items
        )
        item_ids = [self.dataset.idx2item[i] for i in item_indices]
        return item_ids, list(scores)

    def recommend(self, user_id: str, n_items: int = 10,
                  context: Optional[Dict] = None) -> List[RecommendationItem]:
        if not self.is_trained:
            self.train_all()

        is_new_user = self.dataset.is_new_user(user_id)

        if is_new_user:
            return self._cold_start_recommend(user_id, n_items, context)

        user_idx = self.dataset.user2idx.get(user_id)

        recall_results = self.multi_recall.recall(
            user_id, user_idx=user_idx, n_items=n_items * 2, context=context
        )

        item_profiles = None
        if self.item_profile_builder:
            item_profiles = {p.item_id: p for p in self.item_profile_builder.get_all_items()}

        user_profile = self.user_profile_builder.get_profile(user_id) if self.user_profile_builder else None

        ranked = self.ranker.rank(
            recall_results,
            context=context,
            item_profiles=item_profiles,
            user_profile=user_profile,
        )

        ranked = self.ranker.dedup(ranked)

        top_ranked = ranked[:n_items]

        results = []
        for i, r in enumerate(top_ranked):
            channels = r.channels if r.channels else ['popular']
            reasons = []
            for ch in channels:
                reason = self.reason_generator.generate_reason(
                    ch,
                    context={'item_id': r.item_id, 'user_id': user_id}
                )
                reasons.append(reason)
            combined_reason = ' + '.join(reasons) if len(reasons) > 1 else (reasons[0] if reasons else '')

            results.append(RecommendationItem(
                item_id=r.item_id,
                score=r.final_score,
                reason=combined_reason,
                channel=','.join(channels),
                rank=i + 1,
            ))

        return results

    def _cold_start_recommend(self, user_id: str, n_items: int,
                              context: Optional[Dict]) -> List[RecommendationItem]:
        user_profile = self.user_profile_builder.get_profile(user_id) if self.user_profile_builder else None
        popular_items = self.popular_rec.get_popular_items(n_items * 2) if self.popular_rec else []

        item_profiles = None
        if self.item_profile_builder:
            item_profiles = {p.item_id: p for p in self.item_profile_builder.get_all_items()}

        item_ids, reasons = self.cold_start.handle_new_user(
            user_profile, popular_items, self.content_rec, n_items,
            item_profiles=item_profiles
        )

        results = []
        for i, item_id in enumerate(item_ids):
            results.append(RecommendationItem(
                item_id=item_id,
                score=1.0 - i * 0.01,
                reason=reasons[i] if i < len(reasons) else '新人推荐',
                channel='cold_start',
                rank=i + 1,
            ))

        return results

    def get_similar_items(self, item_id: str, n: int = 10) -> List[Dict]:
        if not self.item_cf and not self.content_rec:
            return []

        results = []

        if self.content_rec:
            item_ids, scores = self.content_rec.get_similar_items(item_id, n)
            for iid, score in zip(item_ids, scores):
                results.append({'item_id': iid, 'score': score, 'method': 'content'})

        if self.item_cf and item_id in self.dataset.item2idx:
            item_idx = self.dataset.item2idx[item_id]
            indices, scores = self.item_cf.get_similar_items(item_idx, n)
            for idx, score in zip(indices, scores):
                iid = self.dataset.idx2item[idx]
                results.append({'item_id': iid, 'score': float(score), 'method': 'item_cf'})

        return results[:n]

    def incremental_update(self):
        if not self.is_trained:
            return

        self.dataset.load_all()

        if self.svd and self.dataset.user_item_matrix is not None:
            rows, cols, ratings = [], [], []
            for _, row in self.dataset.interactions.iterrows():
                uid = str(row['user_id'])
                iid = str(row['item_id'])
                if uid in self.dataset.user2idx and iid in self.dataset.item2idx:
                    rows.append(self.dataset.user2idx[uid])
                    cols.append(self.dataset.item2idx[iid])
                    ratings.append(float(row.get('rating', 1.0)))

            new_interactions = list(zip(rows, cols, ratings))
            self.svd.incremental_update(new_interactions)

        if self.popular_rec:
            self.popular_rec.fit(self.dataset.interactions, self.dataset.items)

        if self.user_profile_builder:
            self.user_profile_builder.build_all()
        if self.item_profile_builder:
            self.item_profile_builder.build_all()

    def retrain(self):
        self.load_data()
        self.train_all()

    def record_feedback(self, user_id: str, item_id: str, reward: float, channel: str):
        if self.bandit_optimizer:
            self.bandit_optimizer.record_feedback(channel, reward)
            weights = self.bandit_optimizer.get_weights()
            if self.multi_recall:
                self.multi_recall.update_channel_weights(weights)

    def evaluate(self, test_users: Dict[str, List[str]]) -> Dict:
        if not self.evaluator:
            return {}

        predictions = {}
        for user_id in test_users:
            recs = self.recommend(user_id, n_items=20)
            predictions[user_id] = [r.item_id for r in recs]

        all_items = [p.item_id for p in self.item_profile_builder.get_all_items()] if self.item_profile_builder else None

        item_categories = None
        if self.item_profile_builder:
            item_categories = {}
            for p in self.item_profile_builder.get_all_items():
                if p.category:
                    item_categories[p.item_id] = p.category

        return self.evaluator.evaluate(
            predictions, test_users, all_items=all_items,
            item_categories=item_categories
        )

    def start_scheduler(self):
        if self.scheduler:
            self.scheduler.start()

    def stop_scheduler(self):
        if self.scheduler:
            self.scheduler.stop()

    def get_scheduler_status(self) -> dict:
        if self.scheduler:
            return self.scheduler.get_job_status()
        return {}

    def get_stats(self) -> Dict[str, Any]:
        return {
            'n_users': self.dataset.n_users,
            'n_items': self.dataset.n_items,
            'n_interactions': len(self.dataset.interactions),
            'is_trained': self.is_trained,
            'recall_channels': list(self.multi_recall.channels.keys()) if self.multi_recall else [],
            'bandit_weights': self.bandit_optimizer.get_weights() if self.bandit_optimizer else {},
            'scheduler_status': self.get_scheduler_status(),
        }
