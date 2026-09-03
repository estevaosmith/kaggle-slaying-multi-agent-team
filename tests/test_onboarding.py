from pathlib import Path

import pytest

from kaggle_slaying.competition import load_competition
from kaggle_slaying.onboarding import create_competition_contract


def test_create_competition_contract_writes_valid_yaml(tmp_path: Path) -> None:
    contract, contract_path = create_competition_contract(
        tmp_path,
        name="Demo Competition",
        slug="demo-competition",
        problem_type="regression",
        target_column="price",
        id_column="row_id",
    )

    loaded = load_competition(contract_path)
    assert contract.metric == "rmse"
    assert loaded.slug == "demo-competition"
    assert loaded.data_directory == Path("data/raw/demo-competition")


def test_create_competition_contract_refuses_duplicate(tmp_path: Path) -> None:
    arguments = {
        "name": "Demo",
        "slug": "demo",
        "problem_type": "binary_classification",
        "target_column": "target",
        "id_column": "id",
    }
    create_competition_contract(tmp_path, **arguments)

    with pytest.raises(FileExistsError):
        create_competition_contract(tmp_path, **arguments)


def test_create_competition_contract_validates_slug(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="slug"):
        create_competition_contract(
            tmp_path,
            name="Invalid",
            slug="Invalid slug!",
            problem_type="binary_classification",
            target_column="target",
            id_column="id",
        )
