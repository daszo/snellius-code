from typing import List


class BaseMetricCalculator:
    """Base class providing modular metric calculation."""

    def calculate_mrr(self, ranks: List[int], k: int) -> float:
        reciprocal_ranks = [1.0 / r if r <= k else 0.0 for r in ranks]
        return sum(reciprocal_ranks) / len(ranks) if ranks else 0.0

    def calculate_hits(self, ranks: List[int], k: int) -> float:
        hits = [1.0 if r <= k else 0.0 for r in ranks]
        return sum(hits) / len(ranks) if ranks else 0.0
