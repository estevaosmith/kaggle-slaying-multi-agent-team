from pathlib import Path

from kaggle_slaying.competition import load_competition


def test_titanic_contract() -> None:
    contract = load_competition(Path("config/competitions/titanic.yaml"))

    assert contract.slug == "titanic"
    assert contract.target_column == "Survived"
    assert contract.metric == "accuracy"
