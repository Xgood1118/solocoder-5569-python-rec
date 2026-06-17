import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


class BanditStrategy(ABC):
    @abstractmethod
    def select_arm(self) -> str:
        pass

    @abstractmethod
    def update(self, arm: str, reward: float):
        pass

    @abstractmethod
    def get_weights(self) -> Dict[str, float]:
        pass


class EpsilonGreedy(BanditStrategy):
    def __init__(self, arms: List[str], epsilon: float = 0.1):
        self.arms = arms
        self.epsilon = epsilon
        self.counts = {arm: 0 for arm in arms}
        self.values = {arm: 0.0 for arm in arms}

    def select_arm(self) -> str:
        if np.random.random() < self.epsilon:
            return np.random.choice(self.arms)
        else:
            return max(self.values, key=self.values.get)

    def update(self, arm: str, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward

    def get_weights(self) -> Dict[str, float]:
        total = sum(self.values.values())
        if total == 0:
            return {arm: 1.0 / len(self.arms) for arm in self.arms}
        return {arm: max(0.01, self.values[arm] / total) for arm in self.arms}

    def get_weights_normalized(self) -> Dict[str, float]:
        weights = self.get_weights()
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}


class UCB(BanditStrategy):
    def __init__(self, arms: List[str], c: float = 2.0):
        self.arms = arms
        self.c = c
        self.counts = {arm: 0 for arm in arms}
        self.values = {arm: 0.0 for arm in arms}
        self.total_counts = 0

    def select_arm(self) -> str:
        for arm in self.arms:
            if self.counts[arm] == 0:
                return arm

        ucb_values = {}
        for arm in self.arms:
            bonus = self.c * np.sqrt(np.log(self.total_counts) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus

        return max(ucb_values, key=ucb_values.get)

    def update(self, arm: str, reward: float):
        self.counts[arm] += 1
        self.total_counts += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward

    def get_weights(self) -> Dict[str, float]:
        if self.total_counts == 0:
            return {arm: 1.0 / len(self.arms) for arm in self.arms}

        values = {}
        for arm in self.arms:
            bonus = self.c * np.sqrt(np.log(self.total_counts) / max(1, self.counts[arm]))
            values[arm] = max(0.01, self.values[arm] + bonus)

        total = sum(values.values())
        return {k: v / total for k, v in values.items()}

    def get_weights_normalized(self) -> Dict[str, float]:
        weights = self.get_weights()
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}


class BanditWeightOptimizer:
    def __init__(self, channels: List[str], strategy: str = 'epsilon_greedy',
                 epsilon: float = 0.1, ucb_c: float = 2.0):
        self.channels = channels

        if strategy == 'epsilon_greedy':
            self.strategy = EpsilonGreedy(channels, epsilon)
        elif strategy == 'ucb':
            self.strategy = UCB(channels, ucb_c)
        else:
            raise ValueError(f"未知的策略: {strategy}")

    def select_channel(self) -> str:
        return self.strategy.select_arm()

    def record_feedback(self, channel: str, reward: float):
        self.strategy.update(channel, reward)

    def record_click(self, channel: str):
        self.strategy.update(channel, 1.0)

    def record_impression(self, channel: str):
        self.strategy.update(channel, 0.0)

    def get_weights(self) -> Dict[str, float]:
        return self.strategy.get_weights_normalized()

    def update_recall_weights(self, recall) -> Dict[str, float]:
        weights = self.get_weights()
        recall.update_channel_weights(weights)
        return weights

    def get_stats(self) -> Dict[str, Dict]:
        stats = {}
        if hasattr(self.strategy, 'counts'):
            for arm in self.channels:
                stats[arm] = {
                    'count': self.strategy.counts.get(arm, 0),
                    'value': self.strategy.values.get(arm, 0.0),
                }
        return stats
