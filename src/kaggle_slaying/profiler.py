from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from kaggle_slaying.competition import CompetitionContract

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSIFICATION_TYPES = {"binary_classification", "multiclass_classification"}
GROUP_NAME_HINTS = ("group", "customer", "user", "patient", "entity", "ticket", "family")
TIME_NAME_HINTS = ("date", "time", "timestamp", "year", "month", "week")


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    kind: str
    train_missing_rate: float
    test_missing_rate: float
    unique_count: int
    unique_ratio: float
    constant: bool
    id_like: bool
    high_cardinality: bool
    unseen_test_rate: float | None
    numeric_mean_shift: float | None


@dataclass(frozen=True)
class ValidationPlan:
    strategy: str
    folds: int
    shuffle: bool
    group_column: str | None
    time_column: str | None
    metric: str
    requires_review: bool
    reason: str


@dataclass(frozen=True)
class DatasetReport:
    competition: str
    modality: str
    problem_type: str
    train_rows: int
    test_rows: int
    feature_count: int
    duplicate_train_rows: int
    target_distribution: dict[str, float]
    potential_group_columns: list[str]
    potential_time_columns: list[str]
    warnings: list[str]
    columns: list[ColumnProfile]
    validation: ValidationPlan


def competition_data_directory(contract: CompetitionContract) -> Path:
    base_directory = contract.data_directory
    if not base_directory.is_absolute():
        base_directory = PROJECT_ROOT / base_directory
    return base_directory / contract.extracted_subdirectory


def _column_kind(series: pd.Series) -> str:
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _categorical_unseen_rate(train: pd.Series, test: pd.Series) -> float:
    train_values = set(train.dropna().astype(str).unique())
    test_values = test.dropna().astype(str)
    if test_values.empty:
        return 0.0
    return float((~test_values.isin(train_values)).mean())


def _numeric_mean_shift(train: pd.Series, test: pd.Series) -> float:
    train_numeric = pd.to_numeric(train, errors="coerce")
    test_numeric = pd.to_numeric(test, errors="coerce")
    train_std = float(train_numeric.std())
    if not train_std or pd.isna(train_std):
        return 0.0
    return float(abs(test_numeric.mean() - train_numeric.mean()) / train_std)


def build_validation_plan(
    contract: CompetitionContract,
    potential_group_columns: list[str],
    potential_time_columns: list[str],
) -> ValidationPlan:
    classification = contract.problem_type in CLASSIFICATION_TYPES
    if contract.time_column:
        return ValidationPlan(
            strategy="time_series_split",
            folds=5,
            shuffle=False,
            group_column=None,
            time_column=contract.time_column,
            metric=contract.metric,
            requires_review=False,
            reason="O contrato declara uma coluna temporal.",
        )
    if contract.group_column:
        strategy = "stratified_group_kfold" if classification else "group_kfold"
        return ValidationPlan(
            strategy=strategy,
            folds=5,
            shuffle=True,
            group_column=contract.group_column,
            time_column=None,
            metric=contract.metric,
            requires_review=False,
            reason="O contrato declara uma coluna de grupo.",
        )
    strategy = "stratified_kfold" if classification else "kfold"
    review = bool(potential_group_columns or potential_time_columns)
    return ValidationPlan(
        strategy=strategy,
        folds=5,
        shuffle=True,
        group_column=None,
        time_column=None,
        metric=contract.metric,
        requires_review=review,
        reason=(
            "Split padrao selecionado; candidatos a grupo/tempo exigem revisao."
            if review
            else "Split padrao compativel com o tipo do problema."
        ),
    )


def profile_competition(contract: CompetitionContract) -> DatasetReport:
    data_directory = competition_data_directory(contract)
    train = pd.read_csv(data_directory / contract.train_file)
    test = pd.read_csv(data_directory / contract.test_file)
    sample = pd.read_csv(data_directory / contract.sample_submission_file)
    if contract.target_column not in train.columns:
        raise ValueError(f"Target {contract.target_column!r} ausente no treino.")
    if contract.target_column in test.columns:
        raise ValueError("O target nao deve estar presente no teste.")
    if contract.id_column not in train.columns or contract.id_column not in test.columns:
        raise ValueError(f"ID {contract.id_column!r} deve existir em treino e teste.")
    if contract.id_column not in sample.columns:
        raise ValueError("O arquivo de submissao nao contem a coluna de ID.")

    feature_names = [column for column in train.columns if column != contract.target_column]
    missing_in_test = sorted(set(feature_names) - set(test.columns))
    if missing_in_test:
        raise ValueError(f"Features ausentes no teste: {missing_in_test}")

    warnings: list[str] = []
    columns: list[ColumnProfile] = []
    potential_groups: list[str] = []
    potential_times: list[str] = []
    for name in feature_names:
        train_column = train[name]
        test_column = test[name]
        kind = _column_kind(train_column)
        unique_count = int(train_column.nunique(dropna=True))
        unique_ratio = unique_count / max(len(train), 1)
        id_like = unique_ratio >= 0.98
        high_cardinality = kind == "categorical" and unique_ratio >= 0.20
        unseen_rate = (
            _categorical_unseen_rate(train_column, test_column) if kind == "categorical" else None
        )
        mean_shift = _numeric_mean_shift(train_column, test_column) if kind == "numeric" else None
        profile = ColumnProfile(
            name=name,
            kind=kind,
            train_missing_rate=float(train_column.isna().mean()),
            test_missing_rate=float(test_column.isna().mean()),
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            constant=unique_count <= 1,
            id_like=id_like,
            high_cardinality=high_cardinality,
            unseen_test_rate=unseen_rate,
            numeric_mean_shift=mean_shift,
        )
        columns.append(profile)
        lowered_name = name.lower()
        if (
            name != contract.id_column
            and unique_count > 1
            and unique_ratio < 0.80
            and any(hint in lowered_name for hint in GROUP_NAME_HINTS)
        ):
            potential_groups.append(name)
        if kind == "datetime" or any(hint in lowered_name for hint in TIME_NAME_HINTS):
            potential_times.append(name)
        if profile.constant:
            warnings.append(f"Coluna constante: {name}")
        if unseen_rate is not None and unseen_rate > 0.10:
            warnings.append(f"Categorias novas no teste em {name}: {unseen_rate:.1%}")
        if mean_shift is not None and mean_shift > 0.50:
            warnings.append(f"Possivel drift numerico em {name}: {mean_shift:.2f} desvios")
        missing_gap = abs(profile.train_missing_rate - profile.test_missing_rate)
        if missing_gap > 0.20:
            warnings.append(f"Diferenca de missing em {name}: {missing_gap:.1%}")

    target = train[contract.target_column]
    if contract.problem_type in CLASSIFICATION_TYPES:
        target_distribution = {
            str(label): float(rate) for label, rate in target.value_counts(normalize=True).items()
        }
    else:
        target_distribution = {
            "mean": float(target.mean()),
            "std": float(target.std()),
            "min": float(target.min()),
            "max": float(target.max()),
        }
    validation = build_validation_plan(contract, potential_groups, potential_times)
    return DatasetReport(
        competition=contract.slug,
        modality=contract.modality,
        problem_type=contract.problem_type,
        train_rows=len(train),
        test_rows=len(test),
        feature_count=len(feature_names),
        duplicate_train_rows=int(train.duplicated().sum()),
        target_distribution=target_distribution,
        potential_group_columns=potential_groups,
        potential_time_columns=potential_times,
        warnings=warnings,
        columns=columns,
        validation=validation,
    )


def save_dataset_report(contract: CompetitionContract) -> tuple[DatasetReport, Path]:
    report = profile_competition(contract)
    artifact_directory = PROJECT_ROOT / "artifacts" / contract.slug / "profile_v1"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    report_path = artifact_directory / "dataset_report.json"
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report, report_path
