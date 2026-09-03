from typer.testing import CliRunner

from kaggle_slaying.cli import app, configured_competitions

runner = CliRunner()


def test_configured_competitions_lists_contracts() -> None:
    competitions = configured_competitions()

    assert "playground-series-s6e9" in competitions
    assert "titanic" in competitions


def test_menu_can_exit_without_side_effects() -> None:
    result = runner.invoke(app, ["menu"], input="0\n")

    assert result.exit_code == 0
    assert "Kaggle-Slaying v0.1" in result.stdout
    assert "Ate logo" in result.stdout


def test_menu_lists_competitions_then_exits() -> None:
    result = runner.invoke(app, ["menu"], input="2\n0\n")

    assert result.exit_code == 0
    assert "Competicoes configuradas" in result.stdout
    assert "playground-series-s6e9" in result.stdout
    assert "titanic" in result.stdout
