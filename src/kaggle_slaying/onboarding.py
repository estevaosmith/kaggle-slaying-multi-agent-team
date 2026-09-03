from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from kaggle_slaying.competition import CompetitionContract

SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
DEFAULT_METRICS = {
    "binary_classification": "roc_auc",
    "multiclass_classification": "log_loss",
    "regression": "rmse",
}


def create_competition_contract(
    config_directory: Path,
    *,
    name: str,
    slug: str,
    problem_type: str,
    target_column: str,
    id_column: str,
    metric: str | None = None,
) -> tuple[CompetitionContract, Path]:
    normalized_slug = slug.strip().lower()
    if not SLUG_PATTERN.fullmatch(normalized_slug):
        raise ValueError("O slug deve conter apenas letras minusculas, numeros e hifens.")
    values = {
        "name": name.strip(),
        "slug": normalized_slug,
        "modality": "tabular",
        "problem_type": problem_type,
        "target_column": target_column.strip(),
        "id_column": id_column.strip(),
        "metric": metric or DEFAULT_METRICS[problem_type],
        "data_directory": f"data/raw/{normalized_slug}",
        "extracted_subdirectory": "files",
        "train_file": "train.csv",
        "test_file": "test.csv",
        "sample_submission_file": "sample_submission.csv",
        "submission_file": "submission.csv",
        "group_column": None,
        "time_column": None,
        "requires_rule_acceptance": True,
    }
    if not values["name"] or not values["target_column"] or not values["id_column"]:
        raise ValueError("Nome, target e ID nao podem ficar vazios.")
    contract = CompetitionContract.model_validate(values)
    config_directory.mkdir(parents=True, exist_ok=True)
    contract_path = config_directory / f"{normalized_slug}.yaml"
    if contract_path.exists():
        raise FileExistsError(f"O contrato {contract_path.name} ja existe.")
    serialized: dict[str, Any] = contract.model_dump(mode="json")
    contract_path.write_text(
        yaml.safe_dump(serialized, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return contract, contract_path
