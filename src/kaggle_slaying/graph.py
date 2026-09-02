from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TypedDict
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

from langgraph.graph import END, START, StateGraph

from kaggle_slaying.baseline import run_titanic_baseline
from kaggle_slaying.competition import CompetitionContract

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BootstrapState(TypedDict, total=False):
    competition_slug: str
    data_directory: str
    environment_ok: bool
    downloaded_archive: str
    extracted_directory: str
    cv_mean: float
    cv_std: float
    metrics_path: str
    submission_path: str
    errors: list[str]


def check_environment(state: BootstrapState) -> BootstrapState:
    """Verifica os recursos essenciais antes de acessar a competicao."""
    errors = list(state.get("errors", []))
    kaggle_executable = Path(sys.executable).parent / "kaggle.exe"
    if not kaggle_executable.is_file():
        errors.append("Kaggle CLI nao encontrado no ambiente virtual.")

    try:
        with urlopen("http://127.0.0.1:11434/api/version", timeout=2) as response:
            if response.status != 200:
                errors.append("Ollama respondeu com status inesperado.")
    except (OSError, URLError) as exc:
        errors.append(f"Servico Ollama indisponivel: {exc}")

    return {"environment_ok": not errors, "errors": errors}


def download_competition_data(state: BootstrapState) -> BootstrapState:
    """Baixa os arquivos da competicao pelo cliente oficial do Kaggle."""
    if not state.get("environment_ok", False):
        return state

    slug = state["competition_slug"]
    data_directory = PROJECT_ROOT / state["data_directory"]
    data_directory.mkdir(parents=True, exist_ok=True)
    existing_archives = sorted(data_directory.glob("*.zip"), key=lambda path: path.stat().st_mtime)
    if existing_archives:
        return {"downloaded_archive": str(existing_archives[-1])}
    result = subprocess.run(
        [
            str(sys.executable),
            "-m",
            "kaggle_slaying.kaggle_cli",
            "competitions",
            "download",
            slug,
            "-p",
            str(data_directory),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        errors = list(state.get("errors", []))
        errors.append((result.stderr or result.stdout).strip())
        return {"errors": errors}

    archives = sorted(data_directory.glob("*.zip"), key=lambda path: path.stat().st_mtime)
    return {"downloaded_archive": str(archives[-1]) if archives else ""}


def extract_competition_data(state: BootstrapState) -> BootstrapState:
    archive_path = Path(state.get("downloaded_archive", ""))
    if not archive_path.is_file():
        errors = list(state.get("errors", []))
        errors.append("Arquivo da competicao nao encontrado para extracao.")
        return {"errors": errors}

    destination = archive_path.parent / "files"
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as archive:
        destination_root = destination.resolve()
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if destination_root not in member_path.parents and member_path != destination_root:
                raise ValueError(f"Caminho inseguro no arquivo ZIP: {member.filename}")
        archive.extractall(destination)
    return {"extracted_directory": str(destination)}


def train_baseline(state: BootstrapState) -> BootstrapState:
    if state.get("errors"):
        return state
    if state["competition_slug"] != "titanic":
        return {"errors": ["Baseline ainda nao implementado para esta competicao."]}

    result = run_titanic_baseline(
        Path(state["extracted_directory"]),
        PROJECT_ROOT / "artifacts" / "titanic" / "baseline_v1",
    )
    return {
        "cv_mean": result.cv_mean,
        "cv_std": result.cv_std,
        "metrics_path": str(result.metrics_path),
        "submission_path": str(result.submission_path),
    }


def build_bootstrap_graph():
    """Constroi o primeiro fluxo vertical do projeto."""
    builder = StateGraph(BootstrapState)
    builder.add_node("check_environment", check_environment)
    builder.add_node("download_data", download_competition_data)
    builder.add_node("extract_data", extract_competition_data)
    builder.add_node("train_baseline", train_baseline)
    builder.add_edge(START, "check_environment")
    builder.add_edge("check_environment", "download_data")
    builder.add_edge("download_data", "extract_data")
    builder.add_edge("extract_data", "train_baseline")
    builder.add_edge("train_baseline", END)
    return builder.compile()


def initial_state(contract: CompetitionContract) -> BootstrapState:
    return {
        "competition_slug": contract.slug,
        "data_directory": str(contract.data_directory),
        "errors": [],
    }
