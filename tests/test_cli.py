from kaggle_slaying.cli import DiagnosticCheck, required_diagnostics_pass


def test_optional_diagnostic_does_not_block_environment() -> None:
    checks = [
        DiagnosticCheck("required", True, True, "ok"),
        DiagnosticCheck("optional", False, False, "not installed"),
    ]

    assert required_diagnostics_pass(checks) is True


def test_required_diagnostic_blocks_environment() -> None:
    checks = [DiagnosticCheck("required", True, False, "missing")]

    assert required_diagnostics_pass(checks) is False
