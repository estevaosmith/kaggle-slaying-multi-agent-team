from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kaggle_slaying.competition import CompetitionContract
from kaggle_slaying.kaggle_cli import configure_project_credentials
from kaggle_slaying.profiler import competition_data_directory

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SubmissionStatus:
    total_submissions: int
    submissions_today: int
    allowed_now: int


@dataclass(frozen=True)
class SubmissionGateReport:
    competition: str
    ready: bool
    checked_at_utc: str
    submission_path: str
    source_report_path: str
    sha256: str
    rows: int
    columns: list[str]
    prediction_min: float | None
    prediction_max: float | None
    prediction_std: float | None
    selected_model: str | None
    local_metric: float | None
    submission_message: str
    status: SubmissionStatus
    blockers: list[str]
    human_approval_required: bool = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_submission_status(competition: str) -> SubmissionStatus:
    configure_project_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    submissions = api.competition_submissions(competition)
    limits = api.competition_get_submission_limits(competition)
    return SubmissionStatus(
        total_submissions=len(submissions),
        submissions_today=int(limits.num_today or 0),
        allowed_now=int(limits.num_allowed_now or 0),
    )


def _load_source_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    if not isinstance(report, dict):
        raise TypeError("O relatorio de origem deve ser um objeto JSON.")
    return report


def validate_submission_candidate(
    contract: CompetitionContract,
    submission_path: Path,
    source_report_path: Path,
    status: SubmissionStatus,
) -> SubmissionGateReport:
    blockers: list[str] = []
    submission = pd.read_csv(submission_path)
    sample = pd.read_csv(competition_data_directory(contract) / contract.sample_submission_file)
    source_report = _load_source_report(source_report_path)

    if list(submission.columns) != list(sample.columns):
        blockers.append("As colunas ou sua ordem diferem do sample_submission.")
    if len(submission) != len(sample):
        blockers.append("A quantidade de linhas difere do sample_submission.")
    if contract.id_column not in submission.columns:
        blockers.append(f"A coluna de ID {contract.id_column!r} esta ausente.")
    elif not submission[contract.id_column].equals(sample[contract.id_column]):
        blockers.append("Os IDs ou sua ordem diferem do sample_submission.")
    if submission.isna().any().any():
        blockers.append("A submissao contem valores ausentes.")

    prediction_columns = [column for column in sample.columns if column != contract.id_column]
    prediction_min: float | None = None
    prediction_max: float | None = None
    prediction_std: float | None = None
    if len(prediction_columns) != 1:
        blockers.append("O gate simples suporta uma unica coluna de predicao.")
    elif prediction_columns[0] in submission.columns:
        predictions = pd.to_numeric(submission[prediction_columns[0]], errors="coerce")
        if predictions.isna().any() or not np.isfinite(predictions).all():
            blockers.append("A coluna de predicao contem valores nao numericos ou infinitos.")
        else:
            prediction_min = float(predictions.min())
            prediction_max = float(predictions.max())
            prediction_std = float(predictions.std())
            if (
                contract.metric in {"auc", "roc_auc", "log_loss"}
                and not predictions.between(0, 1).all()
            ):
                blockers.append("As probabilidades devem estar entre zero e um.")
            if prediction_std == 0:
                blockers.append("A predicao e constante.")

    reported_submission = source_report.get("submission_path")
    if not reported_submission or Path(reported_submission).resolve() != submission_path.resolve():
        blockers.append("O CSV nao corresponde ao artefato declarado no relatorio de origem.")
    if status.allowed_now <= 0:
        blockers.append("O Kaggle nao permite uma nova submissao neste momento.")

    selected = source_report.get("selected") or {}
    selected_model = selected.get("name")
    local_metric = selected.get("robust_score")
    short_model = str(selected_model or "candidate").replace("blend:", "")[:45]
    message = (
        f"agent-v2 {short_model} cv={float(local_metric):.5f}" if local_metric else short_model
    )
    return SubmissionGateReport(
        competition=contract.slug,
        ready=not blockers,
        checked_at_utc=datetime.now(UTC).isoformat(),
        submission_path=str(submission_path.resolve()),
        source_report_path=str(source_report_path.resolve()),
        sha256=sha256_file(submission_path),
        rows=len(submission),
        columns=list(submission.columns),
        prediction_min=prediction_min,
        prediction_max=prediction_max,
        prediction_std=prediction_std,
        selected_model=selected_model,
        local_metric=float(local_metric) if local_metric is not None else None,
        submission_message=message,
        status=status,
        blockers=blockers,
    )


def run_submission_gate(
    contract: CompetitionContract,
    submission_path: Path | None = None,
    source_report_path: Path | None = None,
) -> tuple[SubmissionGateReport, Path]:
    experiment_directory = PROJECT_ROOT / "artifacts" / contract.slug / "model_factory_v2"
    submission_path = submission_path or experiment_directory / contract.submission_file
    source_report_path = source_report_path or experiment_directory / "experiment_report.json"
    status = fetch_submission_status(contract.slug)
    report = validate_submission_candidate(
        contract,
        submission_path,
        source_report_path,
        status,
    )
    gate_directory = PROJECT_ROOT / "artifacts" / contract.slug / "submission_gate"
    gate_directory.mkdir(parents=True, exist_ok=True)
    report_path = gate_directory / "gate_report.json"
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report, report_path
