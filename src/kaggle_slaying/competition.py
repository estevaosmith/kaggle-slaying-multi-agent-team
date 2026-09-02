from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class CompetitionContract(BaseModel):
    """Configuracao deterministica de uma competicao suportada."""

    name: str
    slug: str
    modality: Literal["tabular", "time_series", "text", "image"] = "tabular"
    problem_type: Literal[
        "binary_classification",
        "multiclass_classification",
        "regression",
    ]
    target_column: str
    id_column: str
    metric: str
    data_directory: Path
    extracted_subdirectory: str = "files"
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    sample_submission_file: str = "sample_submission.csv"
    submission_file: str
    group_column: str | None = None
    time_column: str | None = None
    requires_rule_acceptance: bool = True


def load_competition(path: Path) -> CompetitionContract:
    """Carrega e valida o contrato YAML de uma competicao."""
    with path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    return CompetitionContract.model_validate(raw_config)
