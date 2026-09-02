import pandas as pd
import pytest

from kaggle_slaying.monitor import summarize_leaderboard


def test_summarize_leaderboard() -> None:
    leaderboard = pd.DataFrame(
        {
            "Rank": [1, 2, 3, 4, 5],
            "Score": [0.90, 0.85, 0.80, 0.75, 0.70],
            "TeamMemberUserNames": ["alpha", "beta", "estevaosmith", "delta", "echo"],
        }
    )

    report = summarize_leaderboard(leaderboard, "demo", "estevaosmith", target_percent=20)

    assert report.rank == 3
    assert report.total_teams == 5
    assert report.target_max_rank == 1
    assert report.target_score == 0.90
    assert report.score_gap == pytest.approx(0.10)
