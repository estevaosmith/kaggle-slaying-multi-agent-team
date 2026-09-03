from __future__ import annotations

import itertools
import json
import operator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError
from catboost.utils import get_gpu_device_count
from langgraph.graph import END, START, StateGraph
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from kaggle_slaying.competition import CompetitionContract
from kaggle_slaying.model_factory import (
    SCREENING_MAX_FOLDS,
    build_preprocessor,
    candidate_pipeline,
    screening_frame,
    selected_features,
    validation_splits,
)
from kaggle_slaying.profiler import competition_data_directory, profile_competition

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCREENING_BOOSTING_ITERATIONS = 300
FINAL_BOOSTING_ITERATIONS = 600
BLEND_WEIGHTS = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class ExperimentCandidate:
    name: str
    cv_mean: float
    cv_std: float
    robust_score: float
    fold_scores: list[float]
    evaluated_rows: int
    device: str
    oof_path: str | None = None
    components: list[str] | None = None
    first_component_weight: float | None = None


class ExperimentState(TypedDict, total=False):
    contract: dict[str, Any]
    artifact_directory: str
    use_gpu: bool
    prepared_path: str
    candidates: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    result: dict[str, Any]


def _advanced_pipeline(
    name: str,
    contract: CompetitionContract,
    feature_frame: pd.DataFrame,
    *,
    iterations: int,
    use_gpu: bool,
) -> Pipeline:
    if name == "linear":
        return candidate_pipeline(name, contract.problem_type, feature_frame)
    if contract.problem_type != "binary_classification":
        raise ValueError("Experiment Agent v2 suporta inicialmente classificacao binaria.")
    if name == "lightgbm_31":
        estimator = LGBMClassifier(
            n_estimators=iterations,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=2,
            random_state=42,
            verbosity=-1,
        )
    elif name == "lightgbm_63":
        estimator = LGBMClassifier(
            n_estimators=iterations,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=50,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            n_jobs=2,
            random_state=42,
            verbosity=-1,
        )
    elif name == "catboost":
        estimator = CatBoostClassifier(
            iterations=iterations,
            depth=7,
            learning_rate=0.06,
            loss_function="Logloss",
            eval_metric="AUC",
            task_type="GPU" if use_gpu else "CPU",
            devices="0" if use_gpu else None,
            thread_count=2,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
    else:
        raise ValueError(f"Candidato v2 desconhecido: {name}")
    return Pipeline([("preprocessor", build_preprocessor(feature_frame)), ("model", estimator)])


def _probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    if probabilities.shape[1] != 2:
        raise ValueError("O ensemble v2 requer exatamente duas classes.")
    return np.asarray(probabilities[:, 1], dtype=float)


def _score_folds(
    target: pd.Series,
    predictions: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    return [
        float(roc_auc_score(target.iloc[validation_indices], predictions[validation_indices]))
        for _, validation_indices in splits
    ]


def _candidate_from_scores(
    name: str,
    scores: list[float],
    rows: int,
    device: str,
    **kwargs: Any,
) -> ExperimentCandidate:
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    return ExperimentCandidate(
        name=name,
        cv_mean=mean,
        cv_std=std,
        robust_score=mean - 0.25 * std,
        fold_scores=scores,
        evaluated_rows=rows,
        device=device,
        **kwargs,
    )


def find_best_auc_blend(
    candidates: list[ExperimentCandidate],
    predictions: dict[str, np.ndarray],
    target: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> ExperimentCandidate | None:
    eligible = sorted(candidates, key=lambda item: item.robust_score, reverse=True)[:3]
    blends: list[ExperimentCandidate] = []
    for first, second in itertools.combinations(eligible, 2):
        for weight in BLEND_WEIGHTS:
            blended = weight * predictions[first.name] + (1.0 - weight) * predictions[second.name]
            scores = _score_folds(target, blended, splits)
            blends.append(
                _candidate_from_scores(
                    name=f"blend:{first.name}:{second.name}:{weight:.2f}",
                    scores=scores,
                    rows=len(target),
                    device="mixed" if first.device != second.device else first.device,
                    components=[first.name, second.name],
                    first_component_weight=weight,
                )
            )
    return max(blends, key=lambda item: item.robust_score) if blends else None


def prepare_agent(state: ExperimentState) -> ExperimentState:
    contract = CompetitionContract.model_validate(state["contract"])
    if contract.problem_type != "binary_classification" or contract.metric not in {
        "auc",
        "roc_auc",
    }:
        raise ValueError("Experiment Agent v2 requer classificacao binaria com AUC.")
    report = profile_competition(contract)
    if report.validation.requires_review:
        raise ValueError("O plano de validacao precisa de revisao antes do experimento v2.")
    data_directory = competition_data_directory(contract)
    train = pd.read_csv(data_directory / contract.train_file)
    train = screening_frame(train, contract, report.validation.strategy)
    features = selected_features(report, contract)
    splits = validation_splits(
        train,
        train[contract.target_column],
        report,
        fold_limit=SCREENING_MAX_FOLDS,
    )
    prepared_path = Path(state["artifact_directory"]) / "screening.joblib"
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"train": train, "features": features, "splits": splits}, prepared_path)
    return {"prepared_path": str(prepared_path)}


def _evaluate_candidate(
    name: str,
    contract: CompetitionContract,
    prepared_path: Path,
    artifact_directory: Path,
    use_gpu: bool,
) -> ExperimentCandidate:
    prepared = joblib.load(prepared_path)
    train: pd.DataFrame = prepared["train"]
    features: list[str] = prepared["features"]
    splits: list[tuple[np.ndarray, np.ndarray]] = prepared["splits"]
    feature_frame = train[features]
    target = train[contract.target_column]
    oof = np.full(len(train), np.nan, dtype=float)
    scores: list[float] = []
    for fit_indices, validation_indices in splits:
        model = _advanced_pipeline(
            name,
            contract,
            feature_frame,
            iterations=SCREENING_BOOSTING_ITERATIONS,
            use_gpu=use_gpu,
        )
        model.fit(feature_frame.iloc[fit_indices], target.iloc[fit_indices])
        predictions = _probability(model, feature_frame.iloc[validation_indices])
        oof[validation_indices] = predictions
        scores.append(float(roc_auc_score(target.iloc[validation_indices], predictions)))
    if np.isnan(oof).any():
        raise ValueError(f"Predicoes OOF incompletas para {name}.")
    oof_path = artifact_directory / f"oof_{name}.npy"
    np.save(oof_path, oof)
    return _candidate_from_scores(
        name=name,
        scores=scores,
        rows=len(train),
        device="gpu" if use_gpu and name == "catboost" else "cpu",
        oof_path=str(oof_path),
    )


def candidate_agent(name: str):
    def run(state: ExperimentState) -> ExperimentState:
        contract = CompetitionContract.model_validate(state["contract"])
        gpu_requested = bool(state.get("use_gpu")) and name == "catboost"
        try:
            candidate = _evaluate_candidate(
                name,
                contract,
                Path(state["prepared_path"]),
                Path(state["artifact_directory"]),
                gpu_requested,
            )
        except CatBoostError as gpu_error:
            if not gpu_requested:
                return {"errors": [f"{name}: {gpu_error}"]}
            try:
                candidate = _evaluate_candidate(
                    name,
                    contract,
                    Path(state["prepared_path"]),
                    Path(state["artifact_directory"]),
                    False,
                )
            except (OSError, RuntimeError, TypeError, ValueError, CatBoostError) as cpu_error:
                return {"errors": [f"{name}: GPU={gpu_error}; CPU={cpu_error}"]}
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            return {"errors": [f"{name}: {error}"]}
        return {"candidates": [asdict(candidate)]}

    return run


def selection_agent(state: ExperimentState) -> ExperimentState:
    contract = CompetitionContract.model_validate(state["contract"])
    artifact_directory = Path(state["artifact_directory"])
    prepared = joblib.load(state["prepared_path"])
    screening_train: pd.DataFrame = prepared["train"]
    splits: list[tuple[np.ndarray, np.ndarray]] = prepared["splits"]
    candidates = [ExperimentCandidate(**item) for item in state.get("candidates", [])]
    if not candidates:
        raise ValueError("Nenhum candidato v2 foi avaliado com sucesso.")
    predictions = {
        candidate.name: np.load(candidate.oof_path)
        for candidate in candidates
        if candidate.oof_path
    }
    blend = find_best_auc_blend(
        candidates,
        predictions,
        screening_train[contract.target_column],
        splits,
    )
    options = candidates + ([blend] if blend else [])
    selected = max(options, key=lambda item: item.robust_score)
    linear = next((item for item in candidates if item.name == "linear"), None)

    data_directory = competition_data_directory(contract)
    train = pd.read_csv(data_directory / contract.train_file)
    test = pd.read_csv(data_directory / contract.test_file)
    sample = pd.read_csv(data_directory / contract.sample_submission_file)
    features: list[str] = prepared["features"]
    component_names = selected.components or [selected.name]
    devices = {candidate.name: candidate.device for candidate in candidates}
    final_models: dict[str, Pipeline] = {}
    final_predictions: dict[str, np.ndarray] = {}
    for name in component_names:
        model = _advanced_pipeline(
            name,
            contract,
            train[features],
            iterations=FINAL_BOOSTING_ITERATIONS,
            use_gpu=devices.get(name) == "gpu",
        )
        model.fit(train[features], train[contract.target_column])
        final_models[name] = model
        final_predictions[name] = _probability(model, test[features])

    if selected.components:
        weight = float(selected.first_component_weight)
        values = weight * final_predictions[component_names[0]]
        values += (1.0 - weight) * final_predictions[component_names[1]]
    else:
        values = final_predictions[selected.name]
    prediction_columns = [column for column in sample.columns if column != contract.id_column]
    if len(prediction_columns) != 1:
        raise ValueError("A submissao binaria deve ter uma coluna de predicao.")
    submission = sample.copy()
    submission[prediction_columns[0]] = values
    if not submission[contract.id_column].equals(sample[contract.id_column]):
        raise ValueError("IDs da submissao v2 estao fora de ordem.")
    if submission.isna().any().any() or not submission[prediction_columns[0]].between(0, 1).all():
        raise ValueError("Probabilidades invalidas na submissao v2.")

    model_path = artifact_directory / "models.joblib"
    submission_path = artifact_directory / contract.submission_file
    joblib.dump(final_models, model_path)
    submission.to_csv(submission_path, index=False)
    result = {
        "competition": contract.slug,
        "metric": contract.metric,
        "screening_rows": len(screening_train),
        "screening_folds": len(splits),
        "gpu_available": bool(state.get("use_gpu")),
        "candidates": [asdict(item) for item in sorted(options, key=lambda x: -x.robust_score)],
        "selected": asdict(selected),
        "improvement_vs_linear": (
            selected.robust_score - linear.robust_score if linear is not None else None
        ),
        "warnings": state.get("errors", []),
        "model_path": str(model_path),
        "submission_path": str(submission_path),
        "human_submission_approval_required": True,
    }
    report_path = artifact_directory / "experiment_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["report_path"] = str(report_path)
    return {"result": result}


def build_experiment_v2_graph():
    builder = StateGraph(ExperimentState)
    builder.add_node("prepare", prepare_agent)
    builder.add_edge(START, "prepare")
    candidate_names = ["linear", "lightgbm_31", "lightgbm_63", "catboost"]
    for name in candidate_names:
        builder.add_node(name, candidate_agent(name))
        builder.add_edge("prepare", name)
    builder.add_node("selection", selection_agent)
    builder.add_edge(candidate_names, "selection")
    builder.add_edge("selection", END)
    return builder.compile()


def run_experiment_v2(contract: CompetitionContract) -> dict[str, Any]:
    artifact_directory = PROJECT_ROOT / "artifacts" / contract.slug / "model_factory_v2"
    gpu_available = get_gpu_device_count() > 0
    state = build_experiment_v2_graph().invoke(
        {
            "contract": contract.model_dump(mode="json"),
            "artifact_directory": str(artifact_directory),
            "use_gpu": gpu_available,
            "candidates": [],
            "errors": [],
        }
    )
    return state["result"]
