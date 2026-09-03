from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from kaggle_slaying.competition import CompetitionContract
from kaggle_slaying.experiment_v2 import run_experiment_v2
from kaggle_slaying.kaggle_cli import configure_project_credentials
from kaggle_slaying.model_factory import run_model_factory
from kaggle_slaying.monitor import refresh_leaderboard_report
from kaggle_slaying.profiler import competition_data_directory, save_dataset_report
from kaggle_slaying.submission_gate import (
    fetch_submission_status,
    run_submission_gate,
    validate_submission_candidate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WorkflowReport:
    competition: str
    phase: str
    data_reused: bool
    training_reused: bool
    training_stage: str
    profile_path: str
    training_report_path: str
    gate_report_path: str
    submission_sha256: str
    submission_ref: int | None
    public_score: float | None
    rank: int | None
    total_teams: int | None
    warnings: list[str]


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_competition_data(contract: CompetitionContract) -> bool:
    data_directory = competition_data_directory(contract)
    required = [contract.train_file, contract.test_file, contract.sample_submission_file]
    if all((data_directory / name).is_file() for name in required):
        return True

    base_directory = _project_path(contract.data_directory)
    base_directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle_slaying.kaggle_cli",
            "competitions",
            "download",
            "-c",
            contract.slug,
            "-p",
            str(base_directory),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    archives = sorted(base_directory.glob("*.zip"), key=lambda path: path.stat().st_mtime)
    if not archives:
        raise FileNotFoundError("O download nao produziu um arquivo ZIP.")
    data_directory.mkdir(parents=True, exist_ok=True)
    with ZipFile(archives[-1]) as archive:
        destination = data_directory.resolve()
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise ValueError("O ZIP contem um caminho inseguro.")
        archive.extractall(data_directory)
    missing = [name for name in required if not (data_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Arquivos ausentes apos o download: {missing}")
    return False


def find_submission_receipt(directory: Path, sha256: str) -> dict[str, Any] | None:
    for path in sorted(directory.glob("submission_*.json"), reverse=True):
        with path.open(encoding="utf-8") as receipt_file:
            receipt = json.load(receipt_file)
        if receipt.get("sha256") == sha256 and receipt.get("authorized_by_user") is True:
            return receipt
    return None


def _training_artifacts(contract: CompetitionContract) -> tuple[str, Path, Path]:
    if contract.problem_type == "binary_classification" and contract.metric in {"auc", "roc_auc"}:
        directory = PROJECT_ROOT / "artifacts" / contract.slug / "model_factory_v2"
        return (
            "experiment_v2",
            directory / "experiment_report.json",
            directory / contract.submission_file,
        )
    directory = PROJECT_ROOT / "artifacts" / contract.slug / "model_factory_v1"
    return (
        "model_factory_v1",
        directory / "factory_report.json",
        directory / contract.submission_file,
    )


def _load_or_train(
    contract: CompetitionContract,
    force_retrain: bool,
) -> tuple[str, dict[str, Any], Path, bool]:
    stage, report_path, submission_path = _training_artifacts(contract)
    if not force_retrain and report_path.is_file() and submission_path.is_file():
        with report_path.open(encoding="utf-8") as report_file:
            return stage, json.load(report_file), report_path, True
    result = (
        run_experiment_v2(contract) if stage == "experiment_v2" else run_model_factory(contract)
    )
    if result.get("accepted") is False:
        raise RuntimeError("A fabrica de modelos bloqueou a candidata.")
    return stage, result, Path(result["report_path"]), False


def run_workflow(
    contract: CompetitionContract,
    username: str,
    force_retrain: bool = False,
) -> tuple[WorkflowReport, Path]:
    data_reused = ensure_competition_data(contract)
    _, profile_path = save_dataset_report(contract)
    stage, training, training_report_path, training_reused = _load_or_train(contract, force_retrain)
    submission_path = Path(training["submission_path"])
    gate, gate_report_path = run_submission_gate(
        contract,
        submission_path=submission_path,
        source_report_path=training_report_path,
    )
    receipt_directory = PROJECT_ROOT / "artifacts" / contract.slug / "submission_gate"
    receipt = find_submission_receipt(receipt_directory, gate.sha256)
    warnings: list[str] = []
    submission_ref: int | None = None
    public_score: float | None = None
    rank: int | None = None
    total_teams: int | None = None
    if receipt:
        phase = "submitted"
        submission_ref = int(receipt["submission_ref"])
        public_score = (
            float(receipt["public_score"]) if receipt.get("public_score") is not None else None
        )
        try:
            leaderboard = refresh_leaderboard_report(contract.slug, username)
            public_score = leaderboard.score
            rank = leaderboard.rank
            total_teams = leaderboard.total_teams
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            warnings.append(f"Leaderboard indisponivel: {error}")
    else:
        phase = "awaiting_approval" if gate.ready else "blocked"
        warnings.extend(gate.blockers)

    report = WorkflowReport(
        competition=contract.slug,
        phase=phase,
        data_reused=data_reused,
        training_reused=training_reused,
        training_stage=stage,
        profile_path=str(profile_path),
        training_report_path=str(training_report_path),
        gate_report_path=str(gate_report_path),
        submission_sha256=gate.sha256,
        submission_ref=submission_ref,
        public_score=public_score,
        rank=rank,
        total_teams=total_teams,
        warnings=warnings,
    )
    workflow_directory = PROJECT_ROOT / "artifacts" / contract.slug / "workflow"
    workflow_directory.mkdir(parents=True, exist_ok=True)
    state_path = workflow_directory / "state.json"
    state_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report, state_path


def load_workflow_state(competition: str) -> tuple[dict[str, Any], Path]:
    state_path = PROJECT_ROOT / "artifacts" / competition / "workflow" / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(
            f"Estado ausente para {competition!r}; execute o comando run primeiro."
        )
    with state_path.open(encoding="utf-8") as state_file:
        return json.load(state_file), state_path


def submit_approved(contract: CompetitionContract, expected_sha256: str) -> dict[str, Any]:
    gate_path = PROJECT_ROOT / "artifacts" / contract.slug / "submission_gate" / "gate_report.json"
    with gate_path.open(encoding="utf-8") as gate_file:
        saved_gate = json.load(gate_file)
    if not saved_gate.get("ready"):
        raise RuntimeError("O gate salvo nao esta pronto.")
    if saved_gate.get("sha256") != expected_sha256:
        raise ValueError("O hash aprovado nao corresponde ao CSV validado.")

    receipt_directory = gate_path.parent
    existing = find_submission_receipt(receipt_directory, expected_sha256)
    if existing:
        return {**existing, "already_submitted": True}

    submission_path = Path(saved_gate["submission_path"])
    source_report_path = Path(saved_gate["source_report_path"])
    current_gate = validate_submission_candidate(
        contract,
        submission_path,
        source_report_path,
        fetch_submission_status(contract.slug),
    )
    if not current_gate.ready or current_gate.sha256 != expected_sha256:
        raise RuntimeError("A candidata mudou ou deixou de passar no gate.")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle_slaying.kaggle_cli",
            "competitions",
            "submit",
            "-c",
            contract.slug,
            "-f",
            str(submission_path),
            "-m",
            current_gate.submission_message,
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())

    configure_project_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    submissions = api.competition_submissions(contract.slug)
    latest = max(submissions, key=lambda item: int(item.ref))
    receipt = {
        "competition": contract.slug,
        "submission_ref": int(latest.ref),
        "file_name": latest.file_name,
        "submitted_at": str(latest.date),
        "description": latest.description,
        "status": str(latest.status),
        "public_score": float(latest.public_score) if latest.public_score else None,
        "private_score": float(latest.private_score) if latest.private_score else None,
        "sha256": expected_sha256,
        "authorized_by_user": True,
        "already_submitted": False,
    }
    receipt_path = receipt_directory / f"submission_{latest.ref}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt
