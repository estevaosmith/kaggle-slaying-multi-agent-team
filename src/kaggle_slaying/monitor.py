from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LeaderboardReport:
    competition: str
    username: str
    score: float
    rank: int
    total_teams: int
    top_percent: float
    percent_beaten: float
    target_percent: float
    target_max_rank: int
    target_score: float
    score_gap: float


def download_leaderboard(competition: str) -> Path:
    output_directory = PROJECT_ROOT / "artifacts" / competition / "leaderboard"
    output_directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(sys.executable),
            "-m",
            "kaggle_slaying.kaggle_cli",
            "competitions",
            "leaderboard",
            competition,
            "--download",
            "-p",
            str(output_directory),
            "-q",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())

    archives = sorted(output_directory.glob("*.zip"), key=lambda path: path.stat().st_mtime)
    if not archives:
        raise FileNotFoundError("Arquivo ZIP do leaderboard nao encontrado.")
    with ZipFile(archives[-1]) as archive:
        csv_members = [member for member in archive.namelist() if member.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError("O ZIP do leaderboard deve conter exatamente um CSV.")
        leaderboard_path = output_directory / "leaderboard.csv"
        leaderboard_path.write_bytes(archive.read(csv_members[0]))
    return leaderboard_path


def summarize_leaderboard(
    leaderboard: pd.DataFrame,
    competition: str,
    username: str,
    target_percent: float = 20.0,
) -> LeaderboardReport:
    usernames = leaderboard["TeamMemberUserNames"].fillna("").astype(str).str.lower()
    matching_rows = leaderboard.loc[usernames.str.contains(username.lower(), regex=False)]
    if matching_rows.empty:
        raise ValueError(f"Usuario {username!r} nao encontrado no leaderboard.")

    team = matching_rows.sort_values("Rank").iloc[0]
    total_teams = len(leaderboard)
    target_max_rank = math.ceil(total_teams * target_percent / 100)
    target_score = float(leaderboard.iloc[target_max_rank - 1]["Score"])
    rank = int(team["Rank"])
    score = float(team["Score"])
    return LeaderboardReport(
        competition=competition,
        username=username,
        score=score,
        rank=rank,
        total_teams=total_teams,
        top_percent=rank / total_teams * 100,
        percent_beaten=(total_teams - rank) / total_teams * 100,
        target_percent=target_percent,
        target_max_rank=target_max_rank,
        target_score=target_score,
        score_gap=target_score - score,
    )


def refresh_leaderboard_report(competition: str, username: str) -> LeaderboardReport:
    leaderboard_path = download_leaderboard(competition)
    leaderboard = pd.read_csv(leaderboard_path)
    report = summarize_leaderboard(leaderboard, competition, username)
    report_path = leaderboard_path.parent / "report.json"
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report
