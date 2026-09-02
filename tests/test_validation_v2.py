import pandas as pd

from kaggle_slaying.validation_v2 import build_family_ticket_groups, build_validation_graph


def test_family_ticket_groups_connect_related_passengers() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["Doe, Mr. A", "Doe, Mrs. B", "Other, Mr. C", "Solo, Miss. D"],
            "SibSp": [1, 1, 0, 0],
            "Parch": [0, 0, 0, 0],
            "Ticket": ["A/1", "A/2", "A/2", "Z/9"],
        }
    )

    groups = build_family_ticket_groups(frame)

    assert groups[0] == groups[1]
    assert groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_validation_graph_nodes() -> None:
    nodes = build_validation_graph().get_graph().nodes

    assert {
        "logistic",
        "extra_trees",
        "catboost",
        "xgboost",
        "robustness_selector",
    }.issubset(nodes)
