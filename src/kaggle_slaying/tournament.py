from __future__ import annotations

import json
import operator
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Annotated, Any, TypedDict

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from langgraph.graph import END, START, StateGraph
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from kaggle_slaying.baseline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_titanic_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CandidateResult:
    name: str
    cv_mean: float
    cv_std: float
    fold_scores: list[float]
    model_path: str
    oof_path: str
    test_probabilities_path: str


@dataclass(frozen=True)
class EnsembleRecipe:
    names: list[str]
    weights: list[float]
    oof_score: float


class TournamentState(TypedDict, total=False):
    data_directory: str
    artifact_directory: str
    candidates: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    ensemble: dict[str, Any]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def candidate_pipeline(name: str, random_state: int = 42) -> Pipeline:
    estimators = {
        "logistic": LogisticRegression(C=0.8, max_iter=1_000, random_state=random_state),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=2,
            max_features=0.8,
            class_weight="balanced",
            n_jobs=1,
            random_state=random_state,
        ),
        "catboost": CatBoostClassifier(
            iterations=500,
            depth=5,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="Accuracy",
            l2_leaf_reg=5,
            allow_writing_files=False,
            verbose=False,
            thread_count=2,
            random_seed=random_state,
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=3,
            min_child_weight=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5,
            eval_metric="logloss",
            n_jobs=2,
            random_state=random_state,
        ),
    }
    if name not in estimators:
        raise ValueError(f"Model Agent desconhecido: {name}")
    return Pipeline([("preprocessor", build_preprocessor()), ("model", estimators[name])])


def evaluate_candidate(
    name: str, data_directory: Path, artifact_directory: Path
) -> CandidateResult:
    train = pd.read_csv(data_directory / "train.csv")
    test = pd.read_csv(data_directory / "test.csv")
    features = build_titanic_features(train)
    test_features = build_titanic_features(test)
    target = train["Survived"].to_numpy()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = candidate_pipeline(name)
    oof_probabilities = np.zeros(len(train), dtype=float)
    fold_scores: list[float] = []

    for train_indices, validation_indices in folds.split(features, target):
        fold_model = clone(pipeline)
        fold_model.fit(features.iloc[train_indices], target[train_indices])
        probabilities = fold_model.predict_proba(features.iloc[validation_indices])[:, 1]
        oof_probabilities[validation_indices] = probabilities
        predictions = (probabilities >= 0.5).astype(int)
        fold_scores.append(float(accuracy_score(target[validation_indices], predictions)))

    pipeline.fit(features, target)
    test_probabilities = pipeline.predict_proba(test_features)[:, 1]
    candidate_directory = artifact_directory / name
    candidate_directory.mkdir(parents=True, exist_ok=True)
    model_path = candidate_directory / "model.joblib"
    oof_path = candidate_directory / "oof.csv"
    test_path = candidate_directory / "test_probabilities.csv"
    joblib.dump(pipeline, model_path)
    pd.DataFrame({"row_index": np.arange(len(train)), "probability": oof_probabilities}).to_csv(
        oof_path, index=False
    )
    pd.DataFrame({"PassengerId": test["PassengerId"], "probability": test_probabilities}).to_csv(
        test_path, index=False
    )
    result = CandidateResult(
        name=name,
        cv_mean=float(np.mean(fold_scores)),
        cv_std=float(np.std(fold_scores)),
        fold_scores=fold_scores,
        model_path=str(model_path),
        oof_path=str(oof_path),
        test_probabilities_path=str(test_path),
    )
    (candidate_directory / "metrics.json").write_text(
        json.dumps(asdict(result), indent=2), encoding="utf-8"
    )
    return result


def select_ensemble(probabilities: dict[str, np.ndarray], target: np.ndarray) -> EnsembleRecipe:
    recipes: list[EnsembleRecipe] = []
    names = sorted(probabilities)
    for name in names:
        score = accuracy_score(target, probabilities[name] >= 0.5)
        recipes.append(EnsembleRecipe([name], [1.0], float(score)))

    for size in range(2, len(names) + 1):
        for selected_names in combinations(names, size):
            weights = np.full(size, 1 / size)
            blended = sum(
                probabilities[name] * weight
                for name, weight in zip(selected_names, weights, strict=True)
            )
            score = accuracy_score(target, blended >= 0.5)
            recipes.append(EnsembleRecipe(list(selected_names), weights.tolist(), float(score)))

    for first, second in combinations(names, 2):
        for first_weight in (0.25, 0.50, 0.75):
            weights = [first_weight, 1 - first_weight]
            blended = probabilities[first] * weights[0] + probabilities[second] * weights[1]
            score = accuracy_score(target, blended >= 0.5)
            recipes.append(EnsembleRecipe([first, second], weights, float(score)))

    return max(recipes, key=lambda recipe: (recipe.oof_score, len(recipe.names)))


def model_agent(name: str):
    def run(state: TournamentState) -> TournamentState:
        result = evaluate_candidate(
            name,
            Path(state["data_directory"]),
            Path(state["artifact_directory"]),
        )
        return {"candidates": [asdict(result)]}

    return run


def ensemble_agent(state: TournamentState) -> TournamentState:
    if state.get("errors"):
        return state
    candidates = [CandidateResult(**candidate) for candidate in state["candidates"]]
    data_directory = Path(state["data_directory"])
    artifact_directory = Path(state["artifact_directory"])
    train = pd.read_csv(data_directory / "train.csv")
    test = pd.read_csv(data_directory / "test.csv")
    sample = pd.read_csv(data_directory / "gender_submission.csv")
    target = train["Survived"].to_numpy()
    oof_probabilities = {
        candidate.name: pd.read_csv(candidate.oof_path)["probability"].to_numpy()
        for candidate in candidates
    }
    test_probabilities = {
        candidate.name: pd.read_csv(candidate.test_probabilities_path)["probability"].to_numpy()
        for candidate in candidates
    }
    recipe = select_ensemble(oof_probabilities, target)
    blended_test = sum(
        test_probabilities[name] * weight
        for name, weight in zip(recipe.names, recipe.weights, strict=True)
    )
    submission = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Survived": (blended_test >= 0.5).astype(int)}
    )
    if list(submission.columns) != list(sample.columns):
        raise ValueError("Schema da submissao do ensemble esta incorreto.")
    if not submission["PassengerId"].equals(sample["PassengerId"]):
        raise ValueError("IDs da submissao do ensemble estao fora de ordem.")

    correlation = pd.DataFrame(oof_probabilities).corr().round(6).to_dict()
    submission_path = artifact_directory / "submission.csv"
    report_path = artifact_directory / "tournament_report.json"
    submission.to_csv(submission_path, index=False)
    ensemble = {
        "names": recipe.names,
        "weights": recipe.weights,
        "oof_score": recipe.oof_score,
        "submission_path": str(submission_path),
        "correlation": correlation,
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    report_path.write_text(json.dumps(ensemble, indent=2), encoding="utf-8")
    return {"ensemble": ensemble}


def build_tournament_graph():
    builder = StateGraph(TournamentState)
    agent_names = ["logistic", "extra_trees", "catboost", "xgboost"]
    for name in agent_names:
        builder.add_node(name, model_agent(name))
        builder.add_edge(START, name)
    builder.add_node("ensemble", ensemble_agent)
    builder.add_edge(agent_names, "ensemble")
    builder.add_edge("ensemble", END)
    return builder.compile()


def run_tournament() -> TournamentState:
    return build_tournament_graph().invoke(
        {
            "data_directory": str(PROJECT_ROOT / "data" / "raw" / "titanic" / "files"),
            "artifact_directory": str(PROJECT_ROOT / "artifacts" / "titanic" / "tournament_v1"),
            "candidates": [],
            "errors": [],
        }
    )
