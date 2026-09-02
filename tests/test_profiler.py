from pathlib import Path

import pandas as pd

from kaggle_slaying.competition import CompetitionContract
from kaggle_slaying.profiler import profile_competition


def _write_competition_files(directory: Path) -> None:
    pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "customer_group": ["a", "a", "b", "b"],
            "amount": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 1, 0, 1],
        }
    ).to_csv(directory / "train.csv", index=False)
    pd.DataFrame(
        {
            "row_id": [5, 6],
            "customer_group": ["c", "a"],
            "amount": [5.0, 6.0],
        }
    ).to_csv(directory / "test.csv", index=False)
    pd.DataFrame({"row_id": [5, 6], "target": [0, 0]}).to_csv(
        directory / "sample_submission.csv", index=False
    )


def test_generic_profiler_recommends_review_for_group_candidate(tmp_path: Path) -> None:
    _write_competition_files(tmp_path)
    contract = CompetitionContract(
        name="Synthetic classification",
        slug="synthetic",
        problem_type="binary_classification",
        target_column="target",
        id_column="row_id",
        metric="accuracy",
        data_directory=tmp_path,
        extracted_subdirectory=".",
        submission_file="submission.csv",
    )

    report = profile_competition(contract)

    assert report.validation.strategy == "stratified_kfold"
    assert report.validation.requires_review is True
    assert report.potential_group_columns == ["customer_group"]


def test_contract_group_column_selects_group_validation(tmp_path: Path) -> None:
    _write_competition_files(tmp_path)
    contract = CompetitionContract(
        name="Synthetic grouped",
        slug="synthetic-grouped",
        problem_type="binary_classification",
        target_column="target",
        id_column="row_id",
        metric="accuracy",
        data_directory=tmp_path,
        extracted_subdirectory=".",
        submission_file="submission.csv",
        group_column="customer_group",
    )

    report = profile_competition(contract)

    assert report.validation.strategy == "stratified_group_kfold"
    assert report.validation.requires_review is False
