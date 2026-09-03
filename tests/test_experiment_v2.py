import numpy as np
import pandas as pd

from kaggle_slaying.experiment_v2 import (
    ExperimentCandidate,
    build_experiment_v2_graph,
    find_best_auc_blend,
)


def _candidate(name: str, score: float) -> ExperimentCandidate:
    return ExperimentCandidate(
        name=name,
        cv_mean=score,
        cv_std=0.0,
        robust_score=score,
        fold_scores=[score, score],
        evaluated_rows=8,
        device="cpu",
    )


def test_experiment_v2_graph_has_parallel_candidate_agents() -> None:
    nodes = build_experiment_v2_graph().get_graph().nodes

    assert {"prepare", "linear", "lightgbm_31", "lightgbm_63", "catboost", "selection"}.issubset(
        nodes
    )


def test_auc_blend_search_uses_shared_out_of_fold_predictions() -> None:
    target = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    splits = [
        (np.array([4, 5, 6, 7]), np.array([0, 1, 2, 3])),
        (np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])),
    ]
    candidates = [_candidate("first", 0.75), _candidate("second", 0.70)]
    predictions = {
        "first": np.array([0.1, 0.7, 0.8, 0.9, 0.2, 0.6, 0.7, 0.8]),
        "second": np.array([0.2, 0.1, 0.4, 0.9, 0.1, 0.2, 0.9, 0.6]),
    }

    blend = find_best_auc_blend(candidates, predictions, target, splits)

    assert blend is not None
    assert blend.components == ["first", "second"]
    assert blend.first_component_weight in {0.25, 0.5, 0.75}
