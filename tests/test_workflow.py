import json
from pathlib import Path

from kaggle_slaying.competition import CompetitionContract
from kaggle_slaying.workflow import ensure_competition_data, find_submission_receipt


def test_ensure_competition_data_reuses_complete_directory(tmp_path: Path) -> None:
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        (tmp_path / name).write_text("id\n1\n", encoding="utf-8")
    contract = CompetitionContract(
        name="Synthetic",
        slug="synthetic",
        problem_type="binary_classification",
        target_column="target",
        id_column="id",
        metric="roc_auc",
        data_directory=tmp_path,
        extracted_subdirectory=".",
        submission_file="submission.csv",
    )

    assert ensure_competition_data(contract) is True


def test_find_submission_receipt_matches_authorized_hash(tmp_path: Path) -> None:
    (tmp_path / "submission_123.json").write_text(
        json.dumps(
            {
                "submission_ref": 123,
                "sha256": "approved-hash",
                "authorized_by_user": True,
            }
        ),
        encoding="utf-8",
    )

    receipt = find_submission_receipt(tmp_path, "approved-hash")

    assert receipt is not None
    assert receipt["submission_ref"] == 123
    assert find_submission_receipt(tmp_path, "different-hash") is None
