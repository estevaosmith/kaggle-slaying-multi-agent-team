from __future__ import annotations

import json
import operator
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from kaggle_slaying.baseline import build_titanic_features
from kaggle_slaying.tournament import candidate_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_SEEDS = (42, 2026, 31415)
MODEL_NAMES = ("logistic", "extra_trees", "catboost", "xgboost")


@dataclass(frozen=True)
class ValidationProfile:
    name: str
    repeated_scores: list[float]
    repeated_mean: float
    repeated_std: float
    grouped_scores: list[float]
    grouped_mean: float
    grouped_std: float
    worst_mean: float
    robust_score: float


class ValidationState(TypedDict, total=False):
    data_directory: str
    artifact_directory: str
    profiles: Annotated[list[dict[str, Any]], operator.add]
    selected: dict[str, Any]


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _union_repeated_values(disjoint_set: DisjointSet, values: pd.Series) -> None:
    grouped_indices: dict[str, list[int]] = {}
    for index, value in enumerate(values.fillna("").astype(str)):
        normalized = value.strip().upper()
        if normalized:
            grouped_indices.setdefault(normalized, []).append(index)
    for indices in grouped_indices.values():
        if len(indices) < 2:
            continue
        anchor = indices[0]
        for index in indices[1:]:
            disjoint_set.union(anchor, index)


def build_family_ticket_groups(frame: pd.DataFrame) -> np.ndarray:
    """Agrupa passageiros ligados por ticket ou sobrenome+tamanho da familia."""
    disjoint_set = DisjointSet(len(frame))
    family_size = frame["SibSp"] + frame["Parch"] + 1
    surname = frame["Name"].str.split(",", n=1).str[0].str.strip().str.upper()
    family_key = surname + "|" + family_size.astype(str)
    _union_repeated_values(disjoint_set, frame["Ticket"])
    _union_repeated_values(disjoint_set, family_key.where(family_size > 1, ""))
    return np.array([disjoint_set.find(index) for index in range(len(frame))])


def _score_splits(
    name: str,
    features: pd.DataFrame,
    target: np.ndarray,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> float:
    predictions = np.full(len(target), -1, dtype=int)
    for train_indices, validation_indices in splits:
        model = candidate_pipeline(name, random_state=seed)
        model.fit(features.iloc[train_indices], target[train_indices])
        probabilities = model.predict_proba(features.iloc[validation_indices])[:, 1]
        predictions[validation_indices] = (probabilities >= 0.5).astype(int)
    if np.any(predictions < 0):
        raise ValueError("Nem todas as linhas receberam predicao out-of-fold.")
    return float(accuracy_score(target, predictions))


def evaluate_validation_profile(name: str, data_directory: Path) -> ValidationProfile:
    train = pd.read_csv(data_directory / "train.csv")
    features = build_titanic_features(train)
    target = train["Survived"].to_numpy()
    groups = build_family_ticket_groups(train)
    repeated_scores: list[float] = []
    grouped_scores: list[float] = []

    for seed in VALIDATION_SEEDS:
        repeated_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        repeated_scores.append(
            _score_splits(name, features, target, repeated_splitter.split(features, target), seed)
        )
        grouped_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        grouped_scores.append(
            _score_splits(
                name,
                features,
                target,
                grouped_splitter.split(features, target, groups),
                seed,
            )
        )

    repeated_mean = float(np.mean(repeated_scores))
    repeated_std = float(np.std(repeated_scores))
    grouped_mean = float(np.mean(grouped_scores))
    grouped_std = float(np.std(grouped_scores))
    worst_mean = min(repeated_mean, grouped_mean)
    robust_score = worst_mean - 0.5 * max(repeated_std, grouped_std)
    return ValidationProfile(
        name=name,
        repeated_scores=repeated_scores,
        repeated_mean=repeated_mean,
        repeated_std=repeated_std,
        grouped_scores=grouped_scores,
        grouped_mean=grouped_mean,
        grouped_std=grouped_std,
        worst_mean=worst_mean,
        robust_score=robust_score,
    )


def validation_agent(name: str):
    def run(state: ValidationState) -> ValidationState:
        profile = evaluate_validation_profile(name, Path(state["data_directory"]))
        artifact_directory = Path(state["artifact_directory"]) / name
        artifact_directory.mkdir(parents=True, exist_ok=True)
        (artifact_directory / "profile.json").write_text(
            json.dumps(asdict(profile), indent=2), encoding="utf-8"
        )
        return {"profiles": [asdict(profile)]}

    return run


def robustness_selector(state: ValidationState) -> ValidationState:
    profiles = [ValidationProfile(**profile) for profile in state["profiles"]]
    selected = max(profiles, key=lambda profile: profile.robust_score)
    report = {
        "selected": asdict(selected),
        "ranking": [
            asdict(profile)
            for profile in sorted(profiles, key=lambda item: item.robust_score, reverse=True)
        ],
    }
    artifact_directory = Path(state["artifact_directory"])
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return {"selected": report}


def build_validation_graph():
    builder = StateGraph(ValidationState)
    for name in MODEL_NAMES:
        builder.add_node(name, validation_agent(name))
        builder.add_edge(START, name)
    builder.add_node("robustness_selector", robustness_selector)
    builder.add_edge(list(MODEL_NAMES), "robustness_selector")
    builder.add_edge("robustness_selector", END)
    return builder.compile()


def run_validation_v2() -> ValidationState:
    return build_validation_graph().invoke(
        {
            "data_directory": str(PROJECT_ROOT / "data" / "raw" / "titanic" / "files"),
            "artifact_directory": str(PROJECT_ROOT / "artifacts" / "titanic" / "validation_v2"),
            "profiles": [],
        }
    )
