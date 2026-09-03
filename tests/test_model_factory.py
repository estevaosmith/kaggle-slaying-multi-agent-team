from pathlib import Path

import pandas as pd

from kaggle_slaying.competition import CompetitionContract
from kaggle_slaying.model_factory import (
    build_model_factory_graph,
    metric_spec,
    run_model_factory,
    screening_frame,
)


def _classification_contract(directory: Path, grouped: bool = False) -> CompetitionContract:
    rows = 120
    values = list(range(rows))
    target = [int(value >= rows / 2) for value in values]
    pd.DataFrame(
        {
            "row_id": values,
            "signal": values,
            "category": ["low" if value < rows / 2 else "high" for value in values],
            "batch_group": [f"g{value // 10}" for value in values],
            "target": target,
        }
    ).to_csv(directory / "train.csv", index=False)
    test_ids = list(range(rows, rows + 10))
    pd.DataFrame(
        {
            "row_id": test_ids,
            "signal": test_ids,
            "category": ["high"] * 10,
            "batch_group": ["future"] * 10,
        }
    ).to_csv(directory / "test.csv", index=False)
    pd.DataFrame({"row_id": test_ids, "target": [0] * 10}).to_csv(
        directory / "sample_submission.csv", index=False
    )
    return CompetitionContract(
        name="Synthetic classification",
        slug="synthetic-classification",
        problem_type="binary_classification",
        target_column="target",
        id_column="row_id",
        metric="accuracy",
        data_directory=directory,
        extracted_subdirectory=".",
        submission_file="submission.csv",
        group_column="batch_group" if grouped else None,
    )


def test_model_factory_graph_has_parallel_model_agents() -> None:
    nodes = build_model_factory_graph().get_graph().nodes

    assert {"benchmark", "linear", "extra_trees", "selection"}.issubset(nodes)


def test_screening_frame_preserves_binary_target_proportion(tmp_path: Path) -> None:
    contract = _classification_contract(tmp_path)
    train = pd.DataFrame(
        {
            "row_id": range(1_000),
            "signal": range(1_000),
            "target": [0] * 900 + [1] * 100,
        }
    )

    screened = screening_frame(train, contract, "stratified_kfold", max_rows=100)

    assert len(screened) == 100
    assert screened["target"].value_counts().to_dict() == {0: 90, 1: 10}


def test_factory_gate_blocks_unreviewed_group_candidate(tmp_path: Path) -> None:
    contract = _classification_contract(tmp_path)

    result = run_model_factory(contract)

    assert result["accepted"] is False
    assert "revisao humana" in " ".join(result["gate_reasons"])
    assert result["submission_path"] is None


def test_factory_generates_classification_submission_after_review(tmp_path: Path) -> None:
    contract = _classification_contract(tmp_path)

    result = run_model_factory(contract, allow_validation_review=True)

    assert result["accepted"] is True
    assert result["selected"]["name"] in {"linear", "extra_trees"}
    submission = pd.read_csv(result["submission_path"])
    assert list(submission.columns) == ["row_id", "target"]
    assert len(submission) == 10


def test_factory_supports_regression(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "row_id": range(100),
            "feature": range(100),
            "target": [2.5 * value + 3 for value in range(100)],
        }
    )
    test = pd.DataFrame({"row_id": range(100, 105), "feature": range(100, 105)})
    sample = pd.DataFrame({"row_id": range(100, 105), "target": [0.0] * 5})
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    sample.to_csv(tmp_path / "sample_submission.csv", index=False)
    contract = CompetitionContract(
        name="Synthetic regression",
        slug="synthetic-regression",
        problem_type="regression",
        target_column="target",
        id_column="row_id",
        metric="rmse",
        data_directory=tmp_path,
        extracted_subdirectory=".",
        submission_file="submission.csv",
    )

    result = run_model_factory(contract)

    assert result["accepted"] is True
    assert metric_spec("RMSE").scorer == "neg_root_mean_squared_error"
    assert Path(result["submission_path"]).exists()


def test_factory_supports_multiclass_probability_submission(tmp_path: Path) -> None:
    values = list(range(150))
    labels = ["a" if value < 50 else "b" if value < 100 else "c" for value in values]
    pd.DataFrame({"row_id": values, "signal": values, "target": labels}).to_csv(
        tmp_path / "train.csv", index=False
    )
    test_ids = list(range(150, 159))
    pd.DataFrame({"row_id": test_ids, "signal": test_ids}).to_csv(
        tmp_path / "test.csv", index=False
    )
    pd.DataFrame({"row_id": test_ids, "a": [0.0] * 9, "b": [0.0] * 9, "c": [0.0] * 9}).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )
    contract = CompetitionContract(
        name="Synthetic multiclass",
        slug="synthetic-multiclass",
        problem_type="multiclass_classification",
        target_column="target",
        id_column="row_id",
        metric="log_loss",
        data_directory=tmp_path,
        extracted_subdirectory=".",
        submission_file="submission.csv",
    )

    result = run_model_factory(contract)

    assert result["accepted"] is True
    submission = pd.read_csv(result["submission_path"])
    assert list(submission.columns) == ["row_id", "a", "b", "c"]
    assert submission[["a", "b", "c"]].sum(axis=1).round(8).eq(1.0).all()
