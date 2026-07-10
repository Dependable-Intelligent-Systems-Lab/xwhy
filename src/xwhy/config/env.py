"""Environment variable utilities."""

from pathlib import Path

from dotenv import load_dotenv


def load_environment(env_file: Path | None = None) -> None:
    """Load environment variables from a .env file."""
    if env_file is None:
        load_dotenv()
    else:
        load_dotenv(dotenv_path=env_file)
