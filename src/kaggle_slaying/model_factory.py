from __future__ import annotations

import json
import operator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict

import joblib
import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kaggle_slaying.competition import CompetitionContract
from kaggle_slaying.profiler import (
    CLASSIFICATION_TYPES,
    DatasetReport,
    competition_data_directory,
    profile_competition,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCREENING_MAX_ROWS = 100_000
SCREENING_MAX_FOLDS = 3
SCREENING_TREE_ESTIMATORS = 100
FINAL_TREE_ESTIMATORS = 250


@dataclass(frozen=True)
class MetricSpec:
    scorer: str
    prediction_kind: str


@dataclass(frozen=True)
class CandidateScore:
    name: str
    cv_mean: float
    cv_std: float
    robust_score: float
    fold_scores: list[float]
    benchmark: bool
    evaluated_rows: int


class FactoryState(TypedDict, total=False):
    contract: dict[str, Any]
    artifact_directory: str
    allow_validation_review: bool
    candidates: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    result: dict[str, Any]


METRICS = {
    "accuracy": MetricSpec("accuracy", "label"),
    "roc_auc": MetricSpec("roc_auc", "probability"),
    "auc": MetricSpec("roc_auc", "probability"),
    "f1": MetricSpec("f1", "label"),
    "f1_macro": MetricSpec("f1_macro", "label"),
    "log_loss": MetricSpec("neg_log_loss", "probability"),
    "rmse": MetricSpec("neg_root_mean_squared_error", "value"),
    "root_mean_squared_error": MetricSpec("neg_root_mean_squared_error", "value"),
    "mae": MetricSpec("neg_mean_absolute_error", "value"),
    "mean_absolute_error": MetricSpec("neg_mean_absolute_error", "value"),
    "r2": MetricSpec("r2", "value"),
}


def metric_spec(metric: str) -> MetricSpec:
    normalized = metric.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized not in METRICS:
        supported = ", ".join(sorted(METRICS))
        raise ValueError(f"Metrica {metric!r} ainda nao suportada. Opcoes: {supported}")
    return METRICS[normalized]


def selected_features(report: DatasetReport, contract: CompetitionContract) -> list[str]:
    """Seleciona features seguras sem conhecimento especifico da competicao."""
    selected: list[str] = []
    for column in report.columns:
        risky_cardinality = column.high_cardinality and (
            column.unseen_test_rate is None or column.unseen_test_rate > 0.20
        )
        if (
            column.name == contract.id_column
            or column.name == contract.group_column
            or column.name == contract.time_column
            or column.constant
            or (column.id_like and column.kind != "numeric")
            or risky_cardinality
        ):
            continue
        selected.append(column.name)
    if not selected:
        raise ValueError("Nenhuma feature segura restou depois do perfilamento.")
    return selected


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = list(frame.select_dtypes(include=["number", "bool"]).columns)
    categorical = [column for column in frame.columns if column not in numeric]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=0.01,
                                max_categories=100,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def candidate_pipeline(
    name: str,
    problem_type: str,
    feature_frame: pd.DataFrame,
    random_state: int = 42,
    tree_estimators: int = FINAL_TREE_ESTIMATORS,
) -> Pipeline:
    classification = problem_type in CLASSIFICATION_TYPES
    if name == "benchmark":
        estimator = (
            DummyClassifier(strategy="prior") if classification else DummyRegressor(strategy="mean")
        )
    elif name == "linear":
        estimator = (
            LogisticRegression(max_iter=1_000, random_state=random_state)
            if classification
            else Ridge(alpha=1.0)
        )
    elif name == "extra_trees":
        estimator = (
            ExtraTreesClassifier(
                n_estimators=tree_estimators,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=2,
                random_state=random_state,
            )
            if classification
            else ExtraTreesRegressor(
                n_estimators=tree_estimators,
                min_samples_leaf=2,
                max_features=0.8,
                n_jobs=2,
                random_state=random_state,
            )
        )
    else:
        raise ValueError(f"Model Agent generico desconhecido: {name}")
    return Pipeline([("preprocessor", build_preprocessor(feature_frame)), ("model", estimator)])


def _fold_count(
    report: DatasetReport,
    target: pd.Series,
    groups: pd.Series | None,
    fold_limit: int | None = None,
) -> int:
    requested = report.validation.folds
    if fold_limit is not None:
        requested = min(requested, fold_limit)
    if groups is not None:
        available = int(groups.nunique())
    elif report.problem_type in CLASSIFICATION_TYPES:
        available = int(target.value_counts().min())
    else:
        available = len(target)
    folds = min(requested, available)
    if folds < 2:
        raise ValueError("Sao necessarios ao menos dois folds independentes.")
    return folds


def validation_splits(
    train: pd.DataFrame,
    target: pd.Series,
    report: DatasetReport,
    fold_limit: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    plan = report.validation
    groups = train[plan.group_column] if plan.group_column else None
    folds = _fold_count(report, target, groups, fold_limit)
    row_positions = np.arange(len(train))
    if plan.strategy == "stratified_group_kfold":
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
        return list(splitter.split(row_positions, target, groups))
    if plan.strategy == "group_kfold":
        return list(GroupKFold(n_splits=folds).split(row_positions, target, groups))
    if plan.strategy == "time_series_split":
        time_values = pd.to_datetime(train[plan.time_column], errors="coerce")
        if time_values.isna().all():
            time_values = train[plan.time_column]
        ordered = np.asarray(time_values.sort_values(kind="stable").index)
        splitter = TimeSeriesSplit(n_splits=folds)
        return [(ordered[fit], ordered[valid]) for fit, valid in splitter.split(ordered)]
    if plan.strategy == "stratified_kfold":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        return list(splitter.split(row_positions, target))
    return list(KFold(n_splits=folds, shuffle=True, random_state=42).split(row_positions))


def screening_frame(
    train: pd.DataFrame,
    contract: CompetitionContract,
    validation_strategy: str,
    max_rows: int = SCREENING_MAX_ROWS,
) -> pd.DataFrame:
    """Cria uma amostra deterministica apenas para a triagem de modelos."""
    if len(train) <= max_rows or validation_strategy not in {"stratified_kfold", "kfold"}:
        return train
    if contract.problem_type in CLASSIFICATION_TYPES:
        sampled, _ = train_test_split(
            train,
            train_size=max_rows,
            stratify=train[contract.target_column],
            random_state=42,
        )
        return sampled.sort_index()
    return train.sample(n=max_rows, random_state=42).sort_index()


def evaluate_candidate(contract: CompetitionContract, name: str) -> CandidateScore:
    report = profile_competition(contract)
    data_directory = competition_data_directory(contract)
    train = pd.read_csv(data_directory / contract.train_file)
    train = screening_frame(train, contract, report.validation.strategy)
    features = selected_features(report, contract)
    feature_frame = train[features]
    target = train[contract.target_column]
    pipeline = candidate_pipeline(
        name,
        contract.problem_type,
        feature_frame,
        tree_estimators=SCREENING_TREE_ESTIMATORS,
    )
    scorer = get_scorer(metric_spec(contract.metric).scorer)
    scores: list[float] = []
    for fit_indices, validation_indices in validation_splits(
        train,
        target,
        report,
        fold_limit=SCREENING_MAX_FOLDS,
    ):
        fold_model = clone(pipeline)
        fold_model.fit(feature_frame.iloc[fit_indices], target.iloc[fit_indices])
        score = scorer(
            fold_model,
            feature_frame.iloc[validation_indices],
            target.iloc[validation_indices],
        )
        scores.append(float(score))
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    return CandidateScore(
        name=name,
        cv_mean=mean,
        cv_std=std,
        robust_score=mean - 0.25 * std,
        fold_scores=scores,
        benchmark=name == "benchmark",
        evaluated_rows=len(train),
    )


def model_agent(name: str):
    def run(state: FactoryState) -> FactoryState:
        contract = CompetitionContract.model_validate(state["contract"])
        try:
            result = evaluate_candidate(contract, name)
            return {"candidates": [asdict(result)]}
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return {"errors": [f"{name}: {exc}"]}

    return run


def _prediction_frame(
    model: Pipeline,
    test_features: pd.DataFrame,
    sample: pd.DataFrame,
    contract: CompetitionContract,
) -> pd.DataFrame:
    output = sample.copy()
    prediction_columns = [column for column in sample.columns if column != contract.id_column]
    spec = metric_spec(contract.metric)
    if spec.prediction_kind in {"label", "value"}:
        if len(prediction_columns) != 1:
            raise ValueError("Submissao de labels/valores deve ter uma coluna de predicao.")
        output[prediction_columns[0]] = model.predict(test_features)
        return output

    probabilities = model.predict_proba(test_features)
    classes = list(model.named_steps["model"].classes_)
    if len(prediction_columns) == 1 and probabilities.shape[1] == 2:
        output[prediction_columns[0]] = probabilities[:, 1]
        return output
    class_positions = {str(label): position for position, label in enumerate(classes)}
    if not set(prediction_columns).issubset(class_positions):
        raise ValueError("Colunas da submissao nao correspondem as classes aprendidas.")
    for column in prediction_columns:
        output[column] = probabilities[:, class_positions[column]]
    return output


def selection_agent(state: FactoryState) -> FactoryState:
    contract = CompetitionContract.model_validate(state["contract"])
    report = profile_competition(contract)
    candidates = [CandidateScore(**candidate) for candidate in state.get("candidates", [])]
    benchmark = next((item for item in candidates if item.benchmark), None)
    models = [item for item in candidates if not item.benchmark]
    gate_reasons = list(state.get("errors", []))
    if report.validation.requires_review and not state.get("allow_validation_review", False):
        gate_reasons.append("O plano de validacao exige revisao humana.")
    if benchmark is None:
        gate_reasons.append("O benchmark nao foi calculado.")
    if not models:
        gate_reasons.append("Nenhum modelo candidato foi avaliado.")
    selected = max(models, key=lambda item: item.robust_score) if models else None
    if selected and benchmark and selected.robust_score <= benchmark.robust_score:
        gate_reasons.append("O melhor modelo nao superou o benchmark ingenuo.")

    artifact_directory = Path(state["artifact_directory"])
    artifact_directory.mkdir(parents=True, exist_ok=True)
    accepted = not gate_reasons
    result: dict[str, Any] = {
        "competition": contract.slug,
        "problem_type": contract.problem_type,
        "metric": contract.metric,
        "screening": {
            "max_rows": SCREENING_MAX_ROWS,
            "max_folds": SCREENING_MAX_FOLDS,
            "tree_estimators": SCREENING_TREE_ESTIMATORS,
        },
        "validation": asdict(report.validation),
        "features": selected_features(report, contract),
        "candidates": [asdict(item) for item in sorted(candidates, key=lambda x: x.name)],
        "accepted": accepted,
        "gate_reasons": gate_reasons,
        "selected": asdict(selected) if selected else None,
        "model_path": None,
        "submission_path": None,
    }
    if accepted and selected:
        data_directory = competition_data_directory(contract)
        train = pd.read_csv(data_directory / contract.train_file)
        test = pd.read_csv(data_directory / contract.test_file)
        sample = pd.read_csv(data_directory / contract.sample_submission_file)
        features = result["features"]
        model = candidate_pipeline(
            selected.name,
            contract.problem_type,
            train[features],
            tree_estimators=FINAL_TREE_ESTIMATORS,
        )
        model.fit(train[features], train[contract.target_column])
        submission = _prediction_frame(model, test[features], sample, contract)
        if not submission[contract.id_column].equals(sample[contract.id_column]):
            raise ValueError("IDs da submissao candidata estao fora de ordem.")
        if submission.isna().any().any():
            raise ValueError("A submissao candidata contem valores ausentes.")
        model_path = artifact_directory / "model.joblib"
        submission_path = artifact_directory / contract.submission_file
        joblib.dump(model, model_path)
        submission.to_csv(submission_path, index=False)
        result["model_path"] = str(model_path)
        result["submission_path"] = str(submission_path)

    report_path = artifact_directory / "factory_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["report_path"] = str(report_path)
    return {"result": result}


def build_model_factory_graph():
    builder = StateGraph(FactoryState)
    agent_names = ["benchmark", "linear", "extra_trees"]
    for name in agent_names:
        builder.add_node(name, model_agent(name))
        builder.add_edge(START, name)
    builder.add_node("selection", selection_agent)
    builder.add_edge(agent_names, "selection")
    builder.add_edge("selection", END)
    return builder.compile()


def run_model_factory(
    contract: CompetitionContract,
    allow_validation_review: bool = False,
) -> dict[str, Any]:
    artifact_directory = PROJECT_ROOT / "artifacts" / contract.slug / "model_factory_v1"
    state = build_model_factory_graph().invoke(
        {
            "contract": contract.model_dump(mode="json"),
            "artifact_directory": str(artifact_directory),
            "allow_validation_review": allow_validation_review,
            "candidates": [],
            "errors": [],
        }
    )
    return state["result"]
