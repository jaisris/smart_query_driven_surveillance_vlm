"""Retrieval evaluation: Precision@K, Recall@K, NDCG for CLIP similarity search."""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np


def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant)


def ndcg_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    top_k = retrieved[:k]
    dcg = sum(
        1.0 / np.log2(i + 2) for i, r in enumerate(top_k) if r in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(retrieved_per_query: List[List[int]], relevant_per_query: List[Set[int]]) -> float:
    rr_list = []
    for retrieved, relevant in zip(retrieved_per_query, relevant_per_query):
        rr = 0.0
        for rank, r in enumerate(retrieved, start=1):
            if r in relevant:
                rr = 1.0 / rank
                break
        rr_list.append(rr)
    return float(np.mean(rr_list)) if rr_list else 0.0


def evaluate_retrieval(
    retrieved_per_query: List[List[int]],
    relevant_per_query: List[Set[int]],
    k_values: List[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Compute mean Precision@K, Recall@K, NDCG@K across all queries.

    Args:
        retrieved_per_query: per query, list of retrieved frame_indices (ranked)
        relevant_per_query:  per query, set of ground-truth relevant frame_indices
    """
    results: Dict[str, float] = {}
    for k in k_values:
        p_list = [precision_at_k(r, rel, k) for r, rel in zip(retrieved_per_query, relevant_per_query)]
        r_list = [recall_at_k(r, rel, k) for r, rel in zip(retrieved_per_query, relevant_per_query)]
        n_list = [ndcg_at_k(r, rel, k) for r, rel in zip(retrieved_per_query, relevant_per_query)]
        results[f"P@{k}"] = round(float(np.mean(p_list)), 4)
        results[f"R@{k}"] = round(float(np.mean(r_list)), 4)
        results[f"NDCG@{k}"] = round(float(np.mean(n_list)), 4)
    results["MRR"] = round(mean_reciprocal_rank(retrieved_per_query, relevant_per_query), 4)
    return results
