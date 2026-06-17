import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os


@dataclass
class AssociationRule:
    antecedent: frozenset
    consequent: frozenset
    support: float
    confidence: float
    lift: float


class AssociationRuleMiner:
    def __init__(self, algorithm: str = 'apriori', min_support: float = 0.01,
                 min_confidence: float = 0.5, output_file: str = 'association_rules.txt'):
        self.algorithm = algorithm
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.output_file = output_file
        self.rules: List[AssociationRule] = []
        self.itemsets: Dict[frozenset, float] = {}

    def fit(self, transactions: List[List[str]]):
        if not transactions:
            return

        if self.algorithm == 'apriori':
            self._apriori(transactions)
        elif self.algorithm == 'fpgrowth':
            self._fpgrowth(transactions)
        else:
            raise ValueError(f"未知的算法: {self.algorithm}")

        self._generate_rules(transactions)
        self._save_rules()

    def _apriori(self, transactions: List[List[str]]):
        n_transactions = len(transactions)
        all_items = set()
        for trans in transactions:
            all_items.update(trans)

        k1_itemsets = {}
        for item in all_items:
            count = sum(1 for trans in transactions if item in trans)
            support = count / n_transactions
            if support >= self.min_support:
                k1_itemsets[frozenset([item])] = support

        self.itemsets.update(k1_itemsets)
        current_itemsets = k1_itemsets

        k = 2
        while current_itemsets:
            candidates = self._apriori_gen(list(current_itemsets.keys()), k)
            candidate_counts = {c: 0 for c in candidates}

            for trans in transactions:
                trans_set = set(trans)
                for candidate in candidates:
                    if candidate.issubset(trans_set):
                        candidate_counts[candidate] += 1

            frequent = {}
            for itemset, count in candidate_counts.items():
                support = count / n_transactions
                if support >= self.min_support:
                    frequent[itemset] = support

            self.itemsets.update(frequent)
            current_itemsets = frequent
            k += 1

    def _apriori_gen(self, itemsets: List[frozenset], k: int) -> List[frozenset]:
        candidates = []
        n = len(itemsets)

        for i in range(n):
            for j in range(i + 1, n):
                itemset1 = itemsets[i]
                itemset2 = itemsets[j]

                list1 = sorted(list(itemset1))
                list2 = sorted(list(itemset2))

                if list1[:-1] == list2[:-1]:
                    candidate = itemset1 | itemset2
                    if self._has_infrequent_subset(candidate, itemsets, k):
                        continue
                    candidates.append(candidate)

        return candidates

    def _has_infrequent_subset(self, candidate: frozenset, prev_itemsets: List[frozenset], k: int) -> bool:
        for item in candidate:
            subset = candidate - frozenset([item])
            if subset not in self.itemsets:
                return True
        return False

    def _fpgrowth(self, transactions: List[List[str]]):
        n_transactions = len(transactions)

        item_counts = {}
        for trans in transactions:
            for item in trans:
                item_counts[item] = item_counts.get(item, 0) + 1

        frequent_items = {
            item: count / n_transactions
            for item, count in item_counts.items()
            if count / n_transactions >= self.min_support
        }

        for item, support in frequent_items.items():
            self.itemsets[frozenset([item])] = support

        sorted_transactions = []
        for trans in transactions:
            filtered = [item for item in trans if item in frequent_items]
            filtered.sort(key=lambda x: frequent_items[x], reverse=True)
            sorted_transactions.append(filtered)

        tree_root = {'children': {}, 'count': 0}

        for trans in sorted_transactions:
            if trans:
                self._insert_tree(trans, tree_root)

        self._mine_tree(tree_root, set(), transactions)

    def _insert_tree(self, items: List[str], node: dict):
        if not items:
            return

        first = items[0]
        if first in node['children']:
            node['children'][first]['count'] += 1
        else:
            node['children'][first] = {'children': {}, 'count': 1}

        self._insert_tree(items[1:], node['children'][first])

    def _mine_tree(self, tree, prefix: set, transactions: List[List[str]]):
        pass

    def _generate_rules(self, transactions: List[List[str]]):
        self.rules = []

        for itemset in self.itemsets:
            if len(itemset) < 2:
                continue

            for item in itemset:
                antecedent = itemset - frozenset([item])
                consequent = frozenset([item])

                if antecedent not in self.itemsets:
                    continue

                support = self.itemsets[itemset]
                confidence = support / self.itemsets[antecedent]

                if confidence < self.min_confidence:
                    continue

                consequent_support = self.itemsets.get(consequent, 0)
                lift = confidence / consequent_support if consequent_support > 0 else 0

                self.rules.append(AssociationRule(
                    antecedent=antecedent,
                    consequent=consequent,
                    support=support,
                    confidence=confidence,
                    lift=lift,
                ))

        self.rules.sort(key=lambda r: r.lift, reverse=True)

    def _save_rules(self):
        if not self.output_file:
            return

        os.makedirs(os.path.dirname(self.output_file) if os.path.dirname(self.output_file) else '.',
                    exist_ok=True)

        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"关联规则挖掘结果 (算法: {self.algorithm})\n")
            f.write(f"最小支持度: {self.min_support}, 最小置信度: {self.min_confidence}\n")
            f.write(f"共发现 {len(self.rules)} 条规则\n\n")
            f.write("-" * 80 + "\n\n")

            for i, rule in enumerate(self.rules, 1):
                ante = ', '.join(sorted(rule.antecedent))
                cons = ', '.join(sorted(rule.consequent))
                f.write(f"规则 {i}: {{{ante}}} => {{{cons}}}\n")
                f.write(f"  支持度: {rule.support:.4f}\n")
                f.write(f"  置信度: {rule.confidence:.4f}\n")
                f.write(f"  提升度: {rule.lift:.4f}\n\n")

    def get_recommendations(self, items: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
        item_set = frozenset(items)
        recommendations = {}

        for rule in self.rules:
            if rule.antecedent.issubset(item_set):
                for cons_item in rule.consequent:
                    if cons_item not in items:
                        if cons_item not in recommendations:
                            recommendations[cons_item] = 0.0
                        recommendations[cons_item] += rule.confidence * rule.support

        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:top_n]

    def get_rules(self, item: Optional[str] = None, top_n: int = 20) -> List[AssociationRule]:
        if item is None:
            return self.rules[:top_n]

        result = []
        for rule in self.rules:
            if item in rule.antecedent or item in rule.consequent:
                result.append(rule)
                if len(result) >= top_n:
                    break
        return result

    def fit_from_interactions(self, interactions_df: pd.DataFrame):
        transactions = []
        user_groups = interactions_df.groupby('user_id')
        for _, group in user_groups:
            items = group['item_id'].astype(str).tolist()
            if items:
                transactions.append(items)

        self.fit(transactions)
