import pandas as pd

from kaggle_slaying.baseline import build_titanic_features


def test_titanic_features_do_not_use_target() -> None:
    frame = pd.DataFrame(
        {
            "Survived": [1],
            "Pclass": [1],
            "Name": ["Doe, Mr. John"],
            "Sex": ["male"],
            "Age": [30.0],
            "SibSp": [0],
            "Parch": [0],
            "Fare": [50.0],
            "Cabin": ["C12"],
            "Embarked": ["S"],
        }
    )

    features = build_titanic_features(frame)

    assert "Survived" not in features.columns
    assert features.loc[0, "FamilySize"] == 1
    assert features.loc[0, "Title"] == "Mr"
    assert features.loc[0, "CabinDeck"] == "C"
