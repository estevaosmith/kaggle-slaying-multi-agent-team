import json
from pathlib import Path

import pandas as pd

from kaggle_slaying.competition import CompetitionContract
from kaggle_slaying.submission_gate import SubmissionStatus, validate_submission_candidate


def _files(directory: Path) -> tuple[CompetitionContract, Path, Path]:
    sample = pd.DataFrame({"id": [10, 11, 12], "target": [0.5, 0.5, 0.5]})
    sample.to_csv(directory / "sample_submission.csv", index=False)
    submission_path = directory / "submission.csv"
    pd.DataFrame({"id": [10, 11, 12], "target": [0.1, 0.8, 0.4]}).to_csv(
        submission_path, index=False
    )
    source_report_path = directory / "experiment_report.json"
    source_report_path.write_text(
        json.dumps(
            {
                "submission_path": str(submission_path),
                "selected": {"name": "blend:model-a:model-b:0.50", "robust_score": 0.91},
            }
        ),
        encoding="utf-8",
    )
    contract = CompetitionContract(
        name="Synthetic",
        slug="synthetic",
        problem_type="binary_classification",
        target_column="target",
        id_column="id",
        metric="roc_auc",
        data_directory=directory,
        extracted_subdirectory=".",
        submission_file="submission.csv",
    )
    return contract, submission_path, source_report_path


def test_submission_gate_accepts_valid_candidate(tmp_path: Path) -> None:
    contract, submission_path, source_report_path = _files(tmp_path)

    report = validate_submission_candidate(
        contract,
        submission_path,
        source_report_path,
        SubmissionStatus(total_submissions=0, submissions_today=0, allowed_now=10),
    )

    assert report.ready is True
    assert report.blockers == []
    assert report.sha256
    assert report.human_approval_required is True


def test_submission_gate_blocks_changed_ids_and_exhausted_limit(tmp_path: Path) -> None:
    contract, submission_path, source_report_path = _files(tmp_path)
    pd.DataFrame({"id": [12, 11, 10], "target": [0.1, 0.8, 0.4]}).to_csv(
        submission_path, index=False
    )

    report = validate_submission_candidate(
        contract,
        submission_path,
        source_report_path,
        SubmissionStatus(total_submissions=10, submissions_today=10, allowed_now=0),
    )

    assert report.ready is False
    assert any("IDs" in blocker for blocker in report.blockers)
    assert any("nao permite" in blocker for blocker in report.blockers)
