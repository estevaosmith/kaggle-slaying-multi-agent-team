import numpy as np
import pytest

from kaggle_slaying.tournament import build_tournament_graph, select_ensemble


def test_tournament_graph_nodes() -> None:
    nodes = build_tournament_graph().get_graph().nodes

    assert {"logistic", "extra_trees", "catboost", "xgboost", "ensemble"}.issubset(nodes)


def test_select_ensemble_prefers_combination() -> None:
    target = np.array([0, 0, 1, 1])
    probabilities = {
        "a": np.array([0.1, 0.6, 0.7, 0.8]),
        "b": np.array([0.2, 0.3, 0.4, 0.9]),
    }

    recipe = select_ensemble(probabilities, target)

    assert recipe.oof_score == pytest.approx(1.0)
    assert set(recipe.names) == {"a", "b"}
