from datetime import UTC, datetime
from pathlib import Path

import pytest

from kaggle_slaying.scout import (
    CompetitionSnapshot,
    assess_competition,
    build_scout_graph,
    infer_metric,
    load_scout_policy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tabular_snapshot(team_count: int = 450) -> CompetitionSnapshot:
    return CompetitionSnapshot(
        slug="playground-example",
        url="https://www.kaggle.com/competitions/playground-example",
        deadline="2026-10-01T23:59:00",
        category="Playground",
        reward="Swag",
        team_count=team_count,
        user_has_entered=False,
        file_names=["train.csv", "test.csv", "sample_submission.csv"],
        total_download_bytes=80 * 1024 * 1024,
        pages={
            "evaluation": "Evaluated on area under the ROC curve.",
            "data-description": "train.csv with `Will_Buy` as the target.",
            "abstract": "A lightweight tabular prediction problem.",
            "rules": "Official competition rules.",
        },
        inspection_errors=[],
    )


def test_scout_graph_has_safe_stages() -> None:
    nodes = build_scout_graph().get_graph().nodes

    assert {"discover", "assess", "rank"}.issubset(nodes)


def test_tabular_candidate_is_ranked_but_requires_review() -> None:
    policy = load_scout_policy(PROJECT_ROOT / "config" / "scout_policy.yaml")

    result = assess_competition(
        _tabular_snapshot(),
        policy,
        now=datetime(2026, 9, 2, tzinfo=UTC),
    )

    assert result.modality == "tabular"
    assert result.problem_type == "binary_classification"
    assert result.metric == "roc_auc"
    assert result.target_column == "Will_Buy"
    assert result.score > 0.70
    assert result.decision == "investigate"
    assert "revisao_humana_das_regras" in result.blockers


def test_image_competition_is_rejected() -> None:
    policy = load_scout_policy(PROJECT_ROOT / "config" / "scout_policy.yaml")
    snapshot = CompetitionSnapshot(
        slug="medical-images",
        url="https://www.kaggle.com/competitions/medical-images",
        deadline="2026-10-10T23:59:00",
        category="Research",
        reward="$10,000",
        team_count=800,
        user_has_entered=False,
        file_names=["train.zip", "test.zip"],
        total_download_bytes=2 * 1024**3,
        pages={
            "evaluation": "Submissions use accuracy.",
            "abstract": "Medical imaging and image classification challenge.",
            "rules": "Official competition rules.",
        },
        inspection_errors=[],
    )

    result = assess_competition(
        snapshot,
        policy,
        now=datetime(2026, 9, 2, tzinfo=UTC),
    )

    assert result.decision == "reject"
    assert "modalidade_nao_suportada:image" in result.hard_failures
    assert "submissao_api_indisponivel" in result.hard_failures


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Area under the ROC curve", "roc_auc"),
        ("Root Mean Squared Error", "rmse"),
        ("RMSE between the logarithm of predicted and observed values", "rmsle"),
        ("multiclass log loss", "log_loss"),
    ],
)
def test_metric_inference(description: str, expected: str) -> None:
    assert infer_metric(description) == expected
