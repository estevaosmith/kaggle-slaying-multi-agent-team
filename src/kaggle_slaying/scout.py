from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import yaml
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from kaggle_slaying.kaggle_cli import configure_project_credentials
from kaggle_slaying.model_factory import METRICS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GIBIBYTE = 1024**3


class HardFilters(BaseModel):
    modalities: list[str]
    problem_types: list[str]
    submission_types: list[str]
    minimum_days_remaining: int
    maximum_download_gb: float
    maximum_train_rows: int
    rules_must_be_reviewed: bool
    api_submission_required: bool


class DecisionThresholds(BaseModel):
    enter: float
    investigate: float
    reject_below: float


class ScoutPolicy(BaseModel):
    version: int
    hard_filters: HardFilters
    score_weights: dict[str, float]
    risk_penalties: dict[str, float]
    decision_thresholds: DecisionThresholds


@dataclass(frozen=True)
class CompetitionSnapshot:
    slug: str
    url: str
    deadline: str
    category: str
    reward: str
    team_count: int
    user_has_entered: bool
    file_names: list[str]
    total_download_bytes: int | None
    pages: dict[str, str]
    inspection_errors: list[str]


@dataclass(frozen=True)
class ScoutAssessment:
    slug: str
    url: str
    deadline: str
    days_remaining: float
    category: str
    reward: str
    team_count: int
    top_20_slots: int
    modality: str
    problem_type: str
    metric: str
    target_column: str | None
    submission_type: str
    download_gb: float | None
    score: float
    decision: str
    score_components: dict[str, float]
    risk_flags: list[str]
    hard_failures: list[str]
    blockers: list[str]
    evidence: list[str]


class ScoutState(TypedDict, total=False):
    policy_path: str
    artifact_directory: str
    limit: int
    now: str
    snapshots: list[dict[str, Any]]
    assessments: list[dict[str, Any]]
    report: dict[str, Any]
    errors: list[str]


def load_scout_policy(path: Path) -> ScoutPolicy:
    with path.open(encoding="utf-8") as policy_file:
        return ScoutPolicy.model_validate(yaml.safe_load(policy_file))


def _to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    raise TypeError(f"Resposta Kaggle inesperada: {type(value).__name__}")


def _slug_from_ref(reference: str) -> str:
    return reference.rstrip("/").split("/")[-1]


def collect_live_competitions(limit: int = 20) -> list[CompetitionSnapshot]:
    """Coleta metadados publicos sem entrar em competicoes ou baixar datasets."""
    configure_project_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    response = api.competitions_list(
        group="general",
        category="all",
        sort_by="earliestDeadline",
        page_size=200,
    )
    listings = list(response.competitions or []) if response else []
    snapshots: list[CompetitionSnapshot] = []
    for raw_listing in listings:
        listing = _to_dict(raw_listing)
        if listing.get("userHasEntered"):
            continue
        reference = str(listing.get("ref", ""))
        slug = _slug_from_ref(reference)
        errors: list[str] = []
        file_names: list[str] = []
        total_bytes: int | None = None
        pages: dict[str, str] = {}
        try:
            file_response = api.competition_list_files(slug, page_size=200)
            file_records = [_to_dict(item) for item in (file_response.files or [])]
            file_names = [str(item.get("name") or item.get("ref")) for item in file_records]
            total_bytes = sum(
                int(item.get("totalBytes") or item.get("size") or 0) for item in file_records
            )
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"files: {exc}")
        try:
            page_records = api.competition_list_pages(slug) or []
            pages = {
                str(item.name).strip().lower(): str(item.content or "") for item in page_records
            }
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"pages: {exc}")
        snapshots.append(
            CompetitionSnapshot(
                slug=slug,
                url=(
                    reference
                    if reference.startswith("http")
                    else f"https://www.kaggle.com/competitions/{slug}"
                ),
                deadline=str(listing.get("deadline", "")),
                category=str(listing.get("category", "Unknown")),
                reward=str(listing.get("reward", "")),
                team_count=int(listing.get("teamCount") or 0),
                user_has_entered=bool(listing.get("userHasEntered")),
                file_names=file_names,
                total_download_bytes=total_bytes,
                pages=pages,
                inspection_errors=errors,
            )
        )
        if len(snapshots) >= limit:
            break
    return snapshots


def _page_text(snapshot: CompetitionSnapshot, *names: str) -> str:
    requested = {name.lower() for name in names}
    return "\n".join(
        content for name, content in snapshot.pages.items() if name.lower() in requested
    )


def infer_metric(text: str) -> str:
    lowered = text.lower()
    if (
        "root mean squared logarithmic" in lowered
        or "rmsle" in lowered
        or (
            (
                "root-mean-squared-error" in lowered
                or "root mean squared error" in lowered
                or "rmse" in lowered
            )
            and "between the logarithm" in lowered
        )
    ):
        return "rmsle"
    patterns = [
        ("rmse", ("root mean squared error", "rmse")),
        ("mae", ("mean absolute error", "mae")),
        ("roc_auc", ("area under the roc", "roc auc", "roc-auc", "auc")),
        ("log_loss", ("log loss", "logarithmic loss")),
        ("f1_macro", ("macro f1", "macro-averaged f1")),
        ("f1", ("f1 score", "f1-score")),
        ("accuracy", ("accuracy",)),
        ("r2", ("r-squared", "r²", "r2 score")),
    ]
    for metric, terms in patterns:
        if any(term in lowered for term in terms):
            return metric
    return "unknown"


def infer_problem_type(metric: str, text: str) -> str:
    lowered = text.lower()
    if metric in {"rmse", "rmsle", "mae", "r2"}:
        return "regression"
    if metric == "accuracy" and any(
        term in lowered
        for term in (
            "multiclass",
            "multi-class",
            "zero through nine",
            "0 through 9",
            "class probabilities",
        )
    ):
        return "multiclass_classification"
    if metric in {"roc_auc", "f1", "accuracy"}:
        return "binary_classification"
    if metric in {"f1_macro", "log_loss"}:
        return (
            "multiclass_classification"
            if any(term in lowered for term in ("multiclass", "multi-class", "class probabilities"))
            else "binary_classification"
        )
    return "unknown"


def infer_modality(snapshot: CompetitionSnapshot, text: str) -> str:
    lowered = text.lower()
    file_names = [name.lower() for name in snapshot.file_names]
    if any(term in lowered for term in ("time series", "forecasting", "forecast")):
        return "time_series"
    if any(
        term in lowered
        for term in (
            "image segmentation",
            "image classification",
            "medical imaging",
            "object detection",
            "cell tracking",
            "radiograph",
            "computer vision",
            "gray-scale images",
            "grayscale images",
            "image of",
            "images of",
            "pixel-value",
        )
    ) or any(name.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")) for name in file_names):
        return "image"
    if any(
        term in lowered
        for term in ("natural language", "text classification", "translation", "language model")
    ):
        return "text"
    required_csv = {"train.csv", "test.csv", "sample_submission.csv"}
    if required_csv.issubset(file_names) or "tabular" in lowered:
        return "tabular"
    return "unknown"


def infer_target(text: str) -> str | None:
    patterns = (
        r"with\s+`?([A-Za-z][A-Za-z0-9_]*)`?\s+as\s+(?:the\s+)?target",
        r"for\s+the\s+`?([A-Za-z][A-Za-z0-9_]*)`?\s+variable",
        r"value\s+of\s+(?:the\s+)?`?([A-Za-z][A-Za-z0-9_]*)`?\s+variable",
        r"first\s+column,?\s+called\s+[\"'`]?([A-Za-z][A-Za-z0-9_]*)",
        r"predict(?:ing)?\s+(?:the\s+)?`?([A-Za-z][A-Za-z0-9_]*)`?\s+(?:column|target)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _days_remaining(deadline: str, now: datetime) -> float:
    parsed = datetime.fromisoformat(deadline)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (parsed - now.astimezone(UTC)).total_seconds() / 86_400


def _compute_fit(download_gb: float | None) -> float:
    if download_gb is None:
        return 0.30
    if download_gb <= 0.10:
        return 1.00
    if download_gb <= 0.50:
        return 0.90
    if download_gb <= 1.00:
        return 0.75
    if download_gb <= 5.00:
        return 0.50
    return 0.0


def _participation_opportunity(team_count: int, category: str) -> float:
    if team_count < 40:
        base = 0.45
    elif team_count < 100:
        base = 0.75
    elif team_count <= 1_200:
        base = 1.00
    elif team_count <= 3_000:
        base = 0.65
    else:
        base = 0.40
    category_factor = {
        "playground": 1.00,
        "getting started": 0.80,
        "featured": 0.70,
        "research": 0.60,
    }.get(category.lower(), 0.65)
    return base * category_factor


def _time_score(days_remaining: float) -> float:
    if days_remaining < 7:
        return 0.0
    if days_remaining < 14:
        return 0.60
    if days_remaining <= 90:
        return 1.00
    return 0.55


def assess_competition(
    snapshot: CompetitionSnapshot,
    policy: ScoutPolicy,
    now: datetime | None = None,
) -> ScoutAssessment:
    now = now or datetime.now(UTC)
    evaluation = _page_text(snapshot, "evaluation")
    descriptive = _page_text(
        snapshot,
        "evaluation",
        "data-description",
        "abstract",
        "description",
        "overview",
        "about the tabular playground series",
    )
    rules = _page_text(snapshot, "rules")
    metric = infer_metric(evaluation)
    problem_type = infer_problem_type(metric, descriptive)
    modality = infer_modality(snapshot, descriptive)
    target = infer_target(descriptive)
    lowered = descriptive.lower()
    file_names = {name.lower() for name in snapshot.file_names}
    submission_type = "csv" if "sample_submission.csv" in file_names else "unknown"
    code_only = any(
        term in lowered
        for term in ("code competition", "code submission", "notebook submission only")
    )
    api_submission_supported = submission_type == "csv" and not code_only
    download_gb = (
        snapshot.total_download_bytes / GIBIBYTE
        if snapshot.total_download_bytes is not None
        else None
    )
    days = _days_remaining(snapshot.deadline, now)
    validation_risk = 0.20
    if modality == "time_series":
        validation_risk = 0.80
    elif any(term in lowered for term in ("patient", "subject", "grouped", "same user")):
        validation_risk = 0.65
    validation_confidence = 0.90 if modality == "tabular" else 0.35
    if metric != "unknown":
        validation_confidence = min(1.0, validation_confidence + 0.05)
    validation_confidence *= 1 - 0.35 * validation_risk
    domain_fit = {
        "playground": 0.95,
        "getting started": 0.85,
        "featured": 0.55,
        "research": 0.35,
    }.get(snapshot.category.lower(), 0.45)
    components = {
        "validation_confidence": validation_confidence,
        "compute_fit": _compute_fit(download_gb),
        "participation_opportunity": _participation_opportunity(
            snapshot.team_count, snapshot.category
        ),
        "metric_support": 1.0 if metric in METRICS else 0.0,
        "domain_fit": domain_fit,
        "time_remaining": _time_score(days),
    }
    risk_flags: list[str] = []
    if code_only:
        risk_flags.append("code_only_submission")
    if any(term in lowered for term in ("external data required", "external data is expected")):
        risk_flags.append("external_data_expected")
    if modality in {"image", "text", "unknown"} or snapshot.category.lower() == "research":
        risk_flags.append("high_domain_complexity")
    if metric == "unknown" or metric not in METRICS:
        risk_flags.append("unstable_or_custom_metric")
    if validation_risk >= 0.60:
        risk_flags.append("suspected_leakage_or_group_structure")

    weighted_score = sum(
        components.get(name, 0.0) * weight for name, weight in policy.score_weights.items()
    )
    penalty = sum(policy.risk_penalties.get(flag, 0.0) for flag in risk_flags)
    score = max(0.0, min(1.0, weighted_score - penalty))
    failures: list[str] = []
    filters = policy.hard_filters
    if modality not in filters.modalities:
        failures.append(f"modalidade_nao_suportada:{modality}")
    if problem_type not in filters.problem_types:
        failures.append(f"problema_nao_suportado:{problem_type}")
    if submission_type not in filters.submission_types:
        failures.append(f"submissao_nao_suportada:{submission_type}")
    if days < filters.minimum_days_remaining:
        failures.append("prazo_insuficiente")
    if download_gb is not None and download_gb > filters.maximum_download_gb:
        failures.append("download_excede_limite")
    if filters.api_submission_required and not api_submission_supported:
        failures.append("submissao_api_indisponivel")
    if metric not in METRICS:
        failures.append(f"metrica_nao_suportada:{metric}")

    blockers = list(snapshot.inspection_errors)
    if filters.rules_must_be_reviewed and rules:
        blockers.append("revisao_humana_das_regras")
    elif filters.rules_must_be_reviewed:
        blockers.append("regras_indisponiveis_para_revisao")
    blockers.append(f"linhas_de_treino_a_confirmar:<={filters.maximum_train_rows}")
    if target is None:
        blockers.append("coluna_alvo_a_confirmar")

    thresholds = policy.decision_thresholds
    if failures or score < thresholds.reject_below:
        decision = "reject"
    elif score >= thresholds.enter and not blockers:
        decision = "enter"
    else:
        decision = "investigate"
    evidence = [
        f"arquivos={len(snapshot.file_names)}",
        f"download={download_gb:.3f}GB" if download_gb is not None else "download=desconhecido",
        f"top20={max(1, int(snapshot.team_count * 0.20))} posicoes",
        f"avaliacao={metric}",
    ]
    return ScoutAssessment(
        slug=snapshot.slug,
        url=snapshot.url,
        deadline=snapshot.deadline,
        days_remaining=days,
        category=snapshot.category,
        reward=snapshot.reward,
        team_count=snapshot.team_count,
        top_20_slots=max(1, int(snapshot.team_count * 0.20)),
        modality=modality,
        problem_type=problem_type,
        metric=metric,
        target_column=target,
        submission_type=submission_type,
        download_gb=download_gb,
        score=score,
        decision=decision,
        score_components=components,
        risk_flags=risk_flags,
        hard_failures=failures,
        blockers=blockers,
        evidence=evidence,
    )


def discovery_agent(state: ScoutState) -> ScoutState:
    snapshots = collect_live_competitions(limit=state.get("limit", 20))
    return {"snapshots": [asdict(snapshot) for snapshot in snapshots]}


def assessment_agent(state: ScoutState) -> ScoutState:
    policy = load_scout_policy(Path(state["policy_path"]))
    now = datetime.fromisoformat(state["now"])
    assessments = [
        assess_competition(CompetitionSnapshot(**snapshot), policy, now=now)
        for snapshot in state.get("snapshots", [])
    ]
    return {"assessments": [asdict(item) for item in assessments]}


def ranking_agent(state: ScoutState) -> ScoutState:
    ranked = sorted(
        state.get("assessments", []),
        key=lambda item: (
            item["decision"] != "reject",
            item["score"],
            item["team_count"],
        ),
        reverse=True,
    )
    report = {
        "generated_at": state["now"],
        "source": "Kaggle API public competition metadata",
        "safe_mode": True,
        "side_effects": [],
        "recommendations": ranked,
        "summary": {
            "inspected": len(ranked),
            "investigate": sum(item["decision"] == "investigate" for item in ranked),
            "enter": sum(item["decision"] == "enter" for item in ranked),
            "reject": sum(item["decision"] == "reject" for item in ranked),
        },
    }
    artifact_directory = Path(state["artifact_directory"])
    artifact_directory.mkdir(parents=True, exist_ok=True)
    report_path = artifact_directory / "scout_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"report": report}


def build_scout_graph():
    builder = StateGraph(ScoutState)
    builder.add_node("discover", discovery_agent)
    builder.add_node("assess", assessment_agent)
    builder.add_node("rank", ranking_agent)
    builder.add_edge(START, "discover")
    builder.add_edge("discover", "assess")
    builder.add_edge("assess", "rank")
    builder.add_edge("rank", END)
    return builder.compile()


def run_scout(limit: int = 20, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    result = build_scout_graph().invoke(
        {
            "policy_path": str(PROJECT_ROOT / "config" / "scout_policy.yaml"),
            "artifact_directory": str(PROJECT_ROOT / "artifacts" / "scout" / "v1"),
            "limit": limit,
            "now": now.isoformat(),
            "errors": [],
        }
    )
    return result["report"]
