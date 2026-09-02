from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KAGGLE_DIRECTORY = PROJECT_ROOT / "work" / "kaggle"
OAUTH_CREDENTIALS = KAGGLE_DIRECTORY / "credentials.json"


def configure_project_credentials() -> None:
    """Mantem as credenciais do Kaggle na area privada e ignorada do projeto."""
    KAGGLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLE_CONFIG_DIR"] = str(KAGGLE_DIRECTORY)

    from kagglesdk.kaggle_creds import KaggleCredentials

    original_save = KaggleCredentials.save
    original_delete = KaggleCredentials.delete

    def save_in_project(self, file_path: str | None = None) -> None:
        original_save(self, file_path or str(OAUTH_CREDENTIALS))

    def delete_from_project(self, file_path: str | None = None) -> None:
        original_delete(self, file_path or str(OAUTH_CREDENTIALS))

    KaggleCredentials.DEFAULT_CREDENTIALS_FILE = str(OAUTH_CREDENTIALS)
    KaggleCredentials.save = save_in_project
    KaggleCredentials.delete = delete_from_project


def main() -> None:
    configure_project_credentials()
    sys.argv[0] = "kaggle"
    from kaggle.cli import main as kaggle_main

    kaggle_main()


if __name__ == "__main__":
    main()
