from pathlib import Path

from kaggle_slaying.competition import load_competition


def test_titanic_contract() -> None:
    contract = load_competition(Path("config/competitions/titanic.yaml"))

    assert contract.slug == "titanic"
    assert contract.target_column == "Survived"
    assert contract.metric == "accuracy"


def test_playground_ev_contract() -> None:
    contract = load_competition(
        Path("config/competitions/playground-series-s6e9.yaml")
    )

    assert contract.slug == "playground-series-s6e9"
    assert contract.target_column == "Will_Buy_EV"
    assert contract.id_column == "id"
    assert contract.metric == "roc_auc"
    assert contract.problem_type == "binary_classification"
