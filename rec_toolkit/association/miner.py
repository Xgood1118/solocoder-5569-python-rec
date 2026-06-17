import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os
from collections import defaultdict


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
            if not candidates:
                break
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
                    if not self._has_infrequent_subset(candidate, itemsets, k):
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

        item_counts: Dict[str, int] = defaultdict(int)
        for trans in transactions:
            for item in set(trans):
                item_counts[item] += 1

        frequent_items = {
            item: count / n_transactions
            for item, count in item_counts.items()
            if count / n_transactions >= self.min_support
        }

        for item, support in frequent_items.items():
            self.itemsets[frozenset([item])] = support

        header_table: Dict[str, Dict] = {}
        for item, support in frequent_items.items():
            header_table[item] = {'support': support, 'count': item_counts[item], 'nodes': []}

        sorted_items = sorted(frequent_items.keys(), key=lambda x: frequent_items[x])

        root = {'item': None, 'count': 0, 'children': {}, 'parent': None}

        for trans in transactions:
            filtered = [item for item in trans if item in frequent_items]
            filtered.sort(key=lambda x: frequent_items[x], reverse=True)
            self._insert_fp_tree(filtered, root, header_table)

        self._mine_fp_tree(root, header_table, frozenset(), n_transactions, sorted_items)

    def _insert_fp_tree(self, items: List[str], node: dict, header_table: dict):
        current = node
        for item in items:
            if item in current['children']:
                current['children'][item]['count'] += 1
            else:
                new_node = {'item': item, 'count': 1, 'children': {}, 'parent': current}
                current['children'][item] = new_node
                if item in header_table:
                    header_table[item]['nodes'].append(new_node)
            current = current['children'][item]

    def _mine_fp_tree(self, root: dict, header_table: dict, prefix: frozenset,
                      n_transactions: int, sorted_items: list):
        for item in sorted_items:
            new_prefix = prefix | frozenset([item])
            prefix_count = header_table[item]['count']
            prefix_support = prefix_count / n_transactions

            if prefix_support >= self.min_support:
                self.itemsets[new_prefix] = prefix_support

            conditional_patterns = self._get_conditional_patterns(item, header_table)

            if conditional_patterns:
                cond_header, cond_n = self._build_conditional_tree(
                    conditional_patterns, n_transactions
                )

                if cond_header:
                    cond_sorted = sorted(
                        cond_header.keys(),
                        key=lambda x: cond_header[x]['support']
                    )
                    self._mine_fp_tree(
                        {'item': None, 'count': 0, 'children': {}, 'parent': None},
                        cond_header, new_prefix, n_transactions, cond_sorted
                    )

    def _get_conditional_patterns(self, item: str, header_table: dict) -> List[Tuple[List[str], int]]:
        patterns = []
        if item not in header_table:
            return patterns

        for node in header_table[item]['nodes']:
            path = []
            count = node['count']
            current = node.get('parent')
            while current and current.get('item') is not None:
                path.append(current['item'])
                current = current.get('parent')
            if path:
                path.reverse()
                patterns.append((path, count))

        return patterns

    def _build_conditional_tree(self, patterns: List[Tuple[List[str], int]],
                                n_transactions: int) -> Tuple[Dict, int]:
        if not patterns:
            return {}, 0

        item_counts: Dict[str, int] = defaultdict(int)
        total = 0
        for path, count in patterns:
            total += count
            for item in set(path):
                item_counts[item] += count

        frequent = {
            item: cnt / n_transactions
            for item, cnt in item_counts.items()
            if cnt / n_transactions >= self.min_support
        }

        if not frequent:
            return {}, 0

        header_table: Dict[str, Dict] = {}
        for item, support in frequent.items():
            header_table[item] = {'support': support, 'count': item_counts[item], 'nodes': []}

        return header_table, total

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

        dir_name = os.path.dirname(self.output_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

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
