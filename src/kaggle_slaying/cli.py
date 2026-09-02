from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import typer
from rich.console import Console
from rich.table import Table

from kaggle_slaying.competition import load_competition
from kaggle_slaying.feature_experiment import run_feature_experiment
from kaggle_slaying.graph import build_bootstrap_graph, initial_state
from kaggle_slaying.model_factory import run_model_factory
from kaggle_slaying.monitor import refresh_leaderboard_report
from kaggle_slaying.profiler import save_dataset_report
from kaggle_slaying.tournament import run_tournament
from kaggle_slaying.validation_v2 import run_validation_v2

app = typer.Typer(help="Ferramentas do Kaggle-Slaying Multi-Agent Team.")
console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@app.callback()
def main() -> None:
    """Kaggle-Slaying Multi-Agent Team."""


def _command_version(command: list[str]) -> tuple[bool, str]:
    requested = Path(command[0])
    executable = str(requested) if requested.is_file() else shutil.which(command[0])
    if executable is None:
        return False, "nao encontrado"
    try:
        command_environment = os.environ.copy()
        kaggle_config = PROJECT_ROOT / "work" / "kaggle"
        kaggle_config.mkdir(parents=True, exist_ok=True)
        command_environment.setdefault("KAGGLE_CONFIG_DIR", str(kaggle_config))
        result = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            check=False,
            env=command_environment,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr).strip().splitlines()
    return result.returncode == 0, output[-1] if output else "sem resposta"


def _ollama_server() -> tuple[bool, str]:
    try:
        with urlopen("http://127.0.0.1:11434/api/version", timeout=2) as response:
            return response.status == 200, "respondendo em 127.0.0.1:11434"
    except (OSError, URLError) as exc:
        return False, f"servico indisponivel: {exc}"


@app.command()
def doctor() -> None:
    """Verifica se o ambiente minimo do projeto esta pronto."""
    checks = [
        ("Python", True, sys.version.split()[0]),
        ("Ambiente virtual", sys.prefix != sys.base_prefix, sys.prefix),
    ]
    for label, command in [
        ("Git", ["git", "--version"]),
        ("Kaggle CLI", [str(Path(sys.executable).parent / "kaggle.exe"), "--version"]),
        (
            "Ollama",
            [str(PROJECT_ROOT / "work" / "tools" / "ollama" / "ollama.exe"), "--version"],
        ),
        ("GPU NVIDIA", ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
    ]:
        ok, detail = _command_version(command)
        checks.append((label, ok, detail))
    ollama_ok, ollama_detail = _ollama_server()
    checks.append(("Servico Ollama", ollama_ok, ollama_detail))

    table = Table(title="Diagnostico do ambiente")
    table.add_column("Componente")
    table.add_column("Status")
    table.add_column("Detalhe")
    for label, ok, detail in checks:
        table.add_row(label, "OK" if ok else "PENDENTE", str(detail))
    console.print(table)

    if not all(ok for _, ok, _ in checks):
        raise typer.Exit(code=1)


@app.command("bootstrap")
def bootstrap(competition: str = "titanic") -> None:
    """Verifica o ambiente e baixa os dados de uma competicao configurada."""
    contract_path = PROJECT_ROOT / "config" / "competitions" / f"{competition}.yaml"
    contract = load_competition(contract_path)
    result = build_bootstrap_graph().invoke(initial_state(contract))
    if result.get("errors"):
        for error in result["errors"]:
            console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Dados baixados:[/green] {result.get('downloaded_archive', '')}")
    console.print(
        f"[green]Baseline validado:[/green] "
        f"{result.get('cv_mean', 0):.4f} +/- {result.get('cv_std', 0):.4f}"
    )
    console.print(f"[green]Submissao gerada:[/green] {result.get('submission_path', '')}")


@app.command("leaderboard-report")
def leaderboard_report(competition: str = "titanic", username: str = "estevaosmith") -> None:
    """Atualiza e resume a posicao da equipe no leaderboard publico."""
    report = refresh_leaderboard_report(competition, username)
    console.print(
        f"[green]Score:[/green] {report.score:.5f} | "
        f"[green]Rank:[/green] {report.rank}/{report.total_teams} | "
        f"[green]Top:[/green] {report.top_percent:.2f}%"
    )
    console.print(
        f"Corte top {report.target_percent:.0f}%: {report.target_score:.5f} "
        f"(gap {report.score_gap:.5f})"
    )


@app.command("tournament")
def tournament() -> None:
    """Executa Model Agents em paralelo e seleciona o melhor ensemble local."""
    result = run_tournament()
    if result.get("errors"):
        for error in result["errors"]:
            console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1)
    candidates = sorted(result["candidates"], key=lambda item: item["cv_mean"], reverse=True)
    for candidate in candidates:
        console.print(
            f"{candidate['name']}: {candidate['cv_mean']:.4f} +/- {candidate['cv_std']:.4f}"
        )
    ensemble = result["ensemble"]
    console.print(
        f"[green]Selecionado:[/green] {ensemble['names']} "
        f"pesos={ensemble['weights']} OOF={ensemble['oof_score']:.4f}"
    )
    console.print(f"[green]Submissao candidata:[/green] {ensemble['submission_path']}")


@app.command("validate-v2")
def validate_v2() -> None:
    """Compara CV repetida e agrupada e seleciona o modelo mais robusto."""
    result = run_validation_v2()
    report = result["selected"]
    for profile in report["ranking"]:
        console.print(
            f"{profile['name']}: repetida={profile['repeated_mean']:.4f} "
            f"agrupada={profile['grouped_mean']:.4f} robusta={profile['robust_score']:.4f}"
        )
    console.print(f"[green]Mais robusto:[/green] {report['selected']['name']}")


@app.command("feature-experiment")
def feature_experiment() -> None:
    """Testa features sem target sobre o modelo Logistic robusto."""
    report = run_feature_experiment()
    selected = report["selected"]
    console.print(
        f"C={selected['regularization_c']}: repetida={selected['repeated_mean']:.4f} "
        f"agrupada={selected['grouped_mean']:.4f} robusta={selected['robust_score']:.4f}"
    )
    status = "ACEITO" if report["accepted"] else "REJEITADO"
    console.print(f"[green]Gate local:[/green] {status}")


@app.command("profile")
def profile(competition: str = "titanic") -> None:
    """Gera perfil de dados e plano de validacao para uma competicao tabular."""
    contract_path = PROJECT_ROOT / "config" / "competitions" / f"{competition}.yaml"
    contract = load_competition(contract_path)
    report, report_path = save_dataset_report(contract)
    console.print(
        f"[green]{report.competition}:[/green] {report.train_rows} treino, "
        f"{report.test_rows} teste, {report.feature_count} features"
    )
    console.print(
        f"Validacao: {report.validation.strategy} | "
        f"revisao={'sim' if report.validation.requires_review else 'nao'}"
    )
    console.print(f"Relatorio: {report_path}")


@app.command("model-factory")
def model_factory(
    competition: str = "titanic",
    approve_validation_review: bool = False,
) -> None:
    """Avalia modelos genericos e gera uma submissao candidata se o gate passar."""
    contract_path = PROJECT_ROOT / "config" / "competitions" / f"{competition}.yaml"
    contract = load_competition(contract_path)
    result = run_model_factory(contract, approve_validation_review)
    for candidate in result["candidates"]:
        console.print(
            f"{candidate['name']}: media={candidate['cv_mean']:.4f} "
            f"robusta={candidate['robust_score']:.4f}"
        )
    if not result["accepted"]:
        console.print("[yellow]Gate: REVISAO NECESSARIA[/yellow]")
        for reason in result["gate_reasons"]:
            console.print(f"- {reason}")
        raise typer.Exit(code=2)
    console.print(f"[green]Gate: ACEITO[/green] | modelo={result['selected']['name']}")
    console.print(f"Submissao candidata: {result['submission_path']}")


if __name__ == "__main__":
    app()
