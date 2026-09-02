from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kaggle_slaying.baseline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_titanic_features,
)
from kaggle_slaying.validation_v2 import VALIDATION_SEEDS, build_family_ticket_groups

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NUMERIC_FEATURES_V2 = [
    *NUMERIC_FEATURES,
    "TicketGroupSize",
    "SurnameGroupSize",
    "FarePerPerson",
    "Child",
    "Mother",
    "AgeMissing",
    "CabinKnown",
]
CATEGORICAL_FEATURES_V2 = [*CATEGORICAL_FEATURES, "FamilySizeBand", "PassengerRole"]
REGULARIZATION_VALUES = (0.05, 0.10, 0.25, 0.50, 1.0, 2.0)


@dataclass(frozen=True)
class FeatureProfile:
    regularization_c: float
    repeated_scores: list[float]
    repeated_mean: float
    repeated_std: float
    grouped_scores: list[float]
    grouped_mean: float
    grouped_std: float
    robust_score: float


def _surname(frame: pd.DataFrame) -> pd.Series:
    return frame["Name"].str.split(",", n=1).str[0].str.strip().str.upper()


def build_titanic_features_v2(
    frame: pd.DataFrame, reference_frame: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Features de grupo sem usar Survived ou qualquer outro target."""
    reference = frame if reference_frame is None else reference_frame
    features = build_titanic_features(frame)
    ticket_counts = reference["Ticket"].fillna("UNKNOWN").astype(str).value_counts()
    surname_counts = _surname(reference).value_counts()
    ticket = frame["Ticket"].fillna("UNKNOWN").astype(str)
    surname = _surname(frame)
    features["TicketGroupSize"] = ticket.map(ticket_counts).fillna(1).astype(float)
    features["SurnameGroupSize"] = surname.map(surname_counts).fillna(1).astype(float)
    features["FarePerPerson"] = frame["Fare"] / features["TicketGroupSize"].clip(lower=1)
    features["Child"] = frame["Age"].lt(14).fillna(False).astype(int)
    features["Mother"] = (
        frame["Sex"].eq("female")
        & frame["Age"].gt(18)
        & frame["Parch"].gt(0)
        & features["Title"].ne("Miss")
    ).astype(int)
    features["AgeMissing"] = frame["Age"].isna().astype(int)
    features["CabinKnown"] = frame["Cabin"].notna().astype(int)
    features["FamilySizeBand"] = np.select(
        [features["FamilySize"].eq(1), features["FamilySize"].le(4)],
        ["alone", "small"],
        default="large",
    )
    features["PassengerRole"] = np.select(
        [features["Child"].eq(1), features["Mother"].eq(1), frame["Sex"].eq("female")],
        ["child", "mother", "woman"],
        default="man",
    )
    return features[NUMERIC_FEATURES_V2 + CATEGORICAL_FEATURES_V2]


def build_feature_pipeline(regularization_c: float, random_state: int) -> Pipeline:
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
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES_V2),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES_V2),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=regularization_c,
                    max_iter=1_000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _score_splitter(
    features: pd.DataFrame,
    target: np.ndarray,
    splitter,
    regularization_c: float,
    seed: int,
) -> float:
    predictions = np.full(len(target), -1, dtype=int)
    for train_indices, validation_indices in splitter:
        pipeline = build_feature_pipeline(regularization_c, seed)
        pipeline.fit(features.iloc[train_indices], target[train_indices])
        predictions[validation_indices] = pipeline.predict(features.iloc[validation_indices])
    return float(accuracy_score(target, predictions))


def evaluate_feature_profile(
    regularization_c: float,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> FeatureProfile:
    reference = pd.concat([train.drop(columns=["Survived"]), test], ignore_index=True)
    features = build_titanic_features_v2(train, reference)
    target = train["Survived"].to_numpy()
    groups = build_family_ticket_groups(train)
    repeated_scores: list[float] = []
    grouped_scores: list[float] = []
    for seed in VALIDATION_SEEDS:
        repeated_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        repeated_scores.append(
            _score_splitter(
                features,
                target,
                repeated_splitter.split(features, target),
                regularization_c,
                seed,
            )
        )
        grouped_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        grouped_scores.append(
            _score_splitter(
                features,
                target,
                grouped_splitter.split(features, target, groups),
                regularization_c,
                seed,
            )
        )
    repeated_mean = float(np.mean(repeated_scores))
    repeated_std = float(np.std(repeated_scores))
    grouped_mean = float(np.mean(grouped_scores))
    grouped_std = float(np.std(grouped_scores))
    robust_score = min(repeated_mean, grouped_mean) - 0.5 * max(repeated_std, grouped_std)
    return FeatureProfile(
        regularization_c=regularization_c,
        repeated_scores=repeated_scores,
        repeated_mean=repeated_mean,
        repeated_std=repeated_std,
        grouped_scores=grouped_scores,
        grouped_mean=grouped_mean,
        grouped_std=grouped_std,
        robust_score=robust_score,
    )


def run_feature_experiment() -> dict:
    data_directory = PROJECT_ROOT / "data" / "raw" / "titanic" / "files"
    artifact_directory = PROJECT_ROOT / "artifacts" / "titanic" / "feature_v2"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(data_directory / "train.csv")
    test = pd.read_csv(data_directory / "test.csv")
    sample = pd.read_csv(data_directory / "gender_submission.csv")
    profiles = [
        evaluate_feature_profile(regularization_c, train, test)
        for regularization_c in REGULARIZATION_VALUES
    ]
    selected = max(profiles, key=lambda profile: profile.robust_score)

    reference = pd.concat([train.drop(columns=["Survived"]), test], ignore_index=True)
    train_features = build_titanic_features_v2(train, reference)
    test_features = build_titanic_features_v2(test, reference)
    pipeline = build_feature_pipeline(selected.regularization_c, random_state=42)
    pipeline.fit(train_features, train["Survived"])
    submission = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Survived": pipeline.predict(test_features)}
    )
    if list(submission.columns) != list(sample.columns):
        raise ValueError("Schema invalido na submissao do Feature Agent.")
    if not submission["PassengerId"].equals(sample["PassengerId"]):
        raise ValueError("IDs fora de ordem na submissao do Feature Agent.")

    validation_report = json.loads(
        (
            PROJECT_ROOT / "artifacts" / "titanic" / "validation_v2" / "validation_report.json"
        ).read_text(encoding="utf-8")
    )
    logistic_v1 = next(
        profile for profile in validation_report["ranking"] if profile["name"] == "logistic"
    )
    accepted = (
        selected.robust_score >= logistic_v1["robust_score"] + 0.002
        and selected.grouped_mean >= logistic_v1["grouped_mean"]
    )
    model_path = artifact_directory / "model.joblib"
    submission_path = artifact_directory / "submission.csv"
    joblib.dump(pipeline, model_path)
    submission.to_csv(submission_path, index=False)
    report = {
        "accepted": accepted,
        "selected": asdict(selected),
        "baseline_logistic_v1": logistic_v1,
        "profiles": [asdict(profile) for profile in profiles],
        "submission_path": str(submission_path),
    }
    (artifact_directory / "feature_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
