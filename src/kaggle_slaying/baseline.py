from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class BaselineResult:
    cv_mean: float
    cv_std: float
    metrics_path: Path
    model_path: Path
    submission_path: Path


NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title", "CabinDeck"]


def build_titanic_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Cria features deterministicas sem utilizar o target."""
    features = frame.copy()
    features["FamilySize"] = features["SibSp"] + features["Parch"] + 1
    features["IsAlone"] = (features["FamilySize"] == 1).astype(int)
    features["Title"] = features["Name"].str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
    rare_titles = ~features["Title"].isin(["Mr", "Miss", "Mrs", "Master"])
    features.loc[rare_titles, "Title"] = "Rare"
    features["CabinDeck"] = features["Cabin"].fillna("Unknown").str[0]
    return features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1_000, random_state=42)),
        ]
    )


def run_titanic_baseline(data_directory: Path, artifact_directory: Path) -> BaselineResult:
    train = pd.read_csv(data_directory / "train.csv")
    test = pd.read_csv(data_directory / "test.csv")
    sample_submission = pd.read_csv(data_directory / "gender_submission.csv")

    train_features = build_titanic_features(train)
    test_features = build_titanic_features(test)
    target = train["Survived"]
    pipeline = build_pipeline()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, train_features, target, cv=folds, scoring="accuracy")
    pipeline.fit(train_features, target)
    predictions = pipeline.predict(test_features).astype(int)

    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
    if list(submission.columns) != list(sample_submission.columns):
        raise ValueError("Schema da submissao difere do arquivo de exemplo.")
    if not submission["PassengerId"].equals(sample_submission["PassengerId"]):
        raise ValueError("A ordem dos IDs da submissao esta incorreta.")
    if submission.isna().any().any():
        raise ValueError("A submissao contem valores ausentes.")

    artifact_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_directory / "metrics.json"
    model_path = artifact_directory / "model.joblib"
    submission_path = artifact_directory / "submission.csv"
    result = BaselineResult(
        cv_mean=float(scores.mean()),
        cv_std=float(scores.std()),
        metrics_path=metrics_path,
        model_path=model_path,
        submission_path=submission_path,
    )
    metrics_path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    joblib.dump(pipeline, model_path)
    submission.to_csv(submission_path, index=False)
    return result
