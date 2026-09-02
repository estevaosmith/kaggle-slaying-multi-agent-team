import pandas as pd

from kaggle_slaying.feature_experiment import build_titanic_features_v2


def test_v2_features_are_target_free_and_group_aware() -> None:
    frame = pd.DataFrame(
        {
            "PassengerId": [1, 2],
            "Survived": [0, 1],
            "Pclass": [3, 3],
            "Name": ["Doe, Mr. A", "Doe, Mrs. B"],
            "Sex": ["male", "female"],
            "Age": [10.0, 30.0],
            "SibSp": [1, 1],
            "Parch": [0, 1],
            "Ticket": ["A/1", "A/1"],
            "Fare": [20.0, 20.0],
            "Cabin": [None, "C12"],
            "Embarked": ["S", "S"],
        }
    )

    features = build_titanic_features_v2(frame)

    assert "Survived" not in features.columns
    assert features["TicketGroupSize"].tolist() == [2.0, 2.0]
    assert features["FarePerPerson"].tolist() == [10.0, 10.0]
    assert features["Child"].tolist() == [1, 0]
