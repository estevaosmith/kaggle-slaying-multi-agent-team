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
from kaggle_slaying.experiment_v2 import run_experiment_v2
from kaggle_slaying.feature_experiment import run_feature_experiment
from kaggle_slaying.graph import build_bootstrap_graph, initial_state
from kaggle_slaying.model_factory import run_model_factory
from kaggle_slaying.monitor import refresh_leaderboard_report
from kaggle_slaying.profiler import save_dataset_report
from kaggle_slaying.scout import run_scout
from kaggle_slaying.submission_gate import run_submission_gate
from kaggle_slaying.tournament import run_tournament
from kaggle_slaying.validation_v2 import run_validation_v2
from kaggle_slaying.workflow import run_workflow, submit_approved

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


@app.command("experiment-v2")
def experiment_v2(competition: str = "playground-series-s6e9") -> None:
    """Compara boosting e ensemble com triagem economica e gate humano."""
    contract_path = PROJECT_ROOT / "config" / "competitions" / f"{competition}.yaml"
    contract = load_competition(contract_path)
    result = run_experiment_v2(contract)
    for candidate in result["candidates"]:
        console.print(
            f"{candidate['name']}: media={candidate['cv_mean']:.4f} "
            f"robusta={candidate['robust_score']:.4f} dispositivo={candidate['device']}"
        )
    console.print(
        f"[green]Vencedor v2:[/green] {result['selected']['name']} | "
        f"ganho vs linear={result['improvement_vs_linear']:.5f}"
    )
    console.print(f"Submissao candidata: {result['submission_path']}")
    console.print("[yellow]Envio ao Kaggle continua bloqueado por aprovacao humana.[/yellow]")


@app.command("submission-gate")
def submission_gate(competition: str = "playground-series-s6e9") -> None:
    """Valida e registra uma candidata sem envia-la ao Kaggle."""
    contract_path = PROJECT_ROOT / "config" / "competitions" / f"{competition}.yaml"
    contract = load_competition(contract_path)
    report, report_path = run_submission_gate(contract)
    status = "PRONTA" if report.ready else "BLOQUEADA"
    console.print(f"Gate de submissao: {status}")
    console.print(
        f"Submissoes: total={report.status.total_submissions}, "
        f"hoje={report.status.submissions_today}, permitidas agora={report.status.allowed_now}"
    )
    console.print(f"SHA-256: {report.sha256}")
    for blocker in report.blockers:
        console.print(f"- {blocker}")
    console.print(f"Relatorio: {report_path}")
    console.print("[yellow]Nenhuma submissao foi enviada.[/yellow]")


@app.command("run")
def run_mvp(
    competition: str = "playground-series-s6e9",
    username: str = "estevaosmith",
    force_retrain: bool = False,
) -> None:
    """Executa o fluxo do MVP e para no gate de aprovacao."""
    contract_path = PROJECT_ROOT / "config" / "competitions" / f"{competition}.yaml"
    contract = load_competition(contract_path)
    report, state_path = run_workflow(contract, username, force_retrain)
    console.print(f"Fase: [green]{report.phase}[/green]")
    console.print(
        f"Dados reutilizados={'sim' if report.data_reused else 'nao'} | "
        f"treino reutilizado={'sim' if report.training_reused else 'nao'}"
    )
    if report.submission_ref is not None:
        console.print(
            f"Submissao={report.submission_ref} | score={report.public_score} | "
            f"rank={report.rank}/{report.total_teams}"
        )
    else:
        console.print(f"Aguardando aprovacao do hash: {report.submission_sha256}")
    console.print(f"Estado: {state_path}")


@app.command("submit-approved")
def submit_approved_command(
    sha256: str = typer.Option(..., "--sha256"),
    competition: str = "playground-series-s6e9",
) -> None:
    """Envia uma unica vez o CSV cujo hash foi aprovado pelo usuario."""
    contract_path = PROJECT_ROOT / "config" / "competitions" / f"{competition}.yaml"
    contract = load_competition(contract_path)
    receipt = submit_approved(contract, sha256)
    if receipt["already_submitted"]:
        console.print(f"Ja enviada anteriormente: {receipt['submission_ref']}")
    else:
        console.print(f"Submissao enviada: {receipt['submission_ref']}")


@app.command("scout")
def scout(limit: int = 20, top: int = 5) -> None:
    """Ranqueia competicoes ativas sem entrar nelas ou baixar seus dados."""
    report = run_scout(limit=limit)
    recommendations = report["recommendations"][:top]
    table = Table(title="Competition Scout v1")
    table.add_column("Competicao")
    table.add_column("Decisao")
    table.add_column("Nota", justify="right")
    table.add_column("Tipo")
    table.add_column("Metrica")
    table.add_column("Equipes", justify="right")
    table.add_column("Dias", justify="right")
    for item in recommendations:
        table.add_row(
            item["slug"],
            item["decision"],
            f"{item['score']:.3f}",
            item["problem_type"],
            item["metric"],
            str(item["team_count"]),
            f"{item['days_remaining']:.0f}",
        )
    console.print(table)
    console.print(
        f"Inspecionadas={report['summary']['inspected']} | "
        f"investigar={report['summary']['investigate']} | "
        f"rejeitadas={report['summary']['reject']}"
    )
    console.print(f"Relatorio: {report['report_path']}")


if __name__ == "__main__":
    app()
