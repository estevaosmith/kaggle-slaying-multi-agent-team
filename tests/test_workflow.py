import json
from pathlib import Path

import pandas as pd

import kaggle_slaying.model_factory as model_factory_module
import kaggle_slaying.profiler as profiler_module
import kaggle_slaying.submission_gate as submission_gate_module
import kaggle_slaying.workflow as workflow_module
from kaggle_slaying.competition import CompetitionContract
from kaggle_slaying.submission_gate import SubmissionStatus
from kaggle_slaying.workflow import (
    ensure_competition_data,
    find_submission_receipt,
    load_workflow_state,
)


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


def test_load_workflow_state_reads_local_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(workflow_module, "PROJECT_ROOT", tmp_path)
    state_path = tmp_path / "artifacts" / "demo" / "workflow" / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"competition": "demo", "phase": "submitted"}))

    state, loaded_path = load_workflow_state("demo")

    assert state["phase"] == "submitted"
    assert loaded_path == state_path


def test_workflow_builds_then_reuses_fresh_competition(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    rows = 120
    pd.DataFrame(
        {
            "id": range(rows),
            "signal": range(rows),
            "target": [int(value >= rows / 2) for value in range(rows)],
        }
    ).to_csv(data_directory / "train.csv", index=False)
    test_signals = [10, 20, 30, 40, 50, 70, 80, 90, 100, 110]
    pd.DataFrame({"id": range(rows, rows + 10), "signal": test_signals}).to_csv(
        data_directory / "test.csv", index=False
    )
    pd.DataFrame({"id": range(rows, rows + 10), "target": [0] * 10}).to_csv(
        data_directory / "sample_submission.csv", index=False
    )
    contract = CompetitionContract(
        name="Fresh synthetic",
        slug="fresh-synthetic",
        problem_type="binary_classification",
        target_column="target",
        id_column="id",
        metric="accuracy",
        data_directory=data_directory,
        extracted_subdirectory=".",
        submission_file="submission.csv",
    )
    for module in (
        workflow_module,
        model_factory_module,
        profiler_module,
        submission_gate_module,
    ):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        submission_gate_module,
        "fetch_submission_status",
        lambda _: SubmissionStatus(0, 0, 10),
    )

    first, _ = workflow_module.run_workflow(contract, username="nobody")
    second, _ = workflow_module.run_workflow(contract, username="nobody")

    assert first.phase == "awaiting_approval"
    assert first.training_reused is False
    assert second.phase == "awaiting_approval"
    assert second.training_reused is True
    assert first.submission_sha256 == second.submission_sha256
