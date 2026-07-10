"""Tests for xwhy.config.env."""

from pathlib import Path

from xwhy.config.env import load_environment


def test_load_environment_without_env_file(tmp_path: Path) -> None:
    """Loading environment should not fail if .env file does not exist."""
    missing_env = tmp_path / ".env"

    load_environment(missing_env)

    assert not missing_env.exists()


def test_load_environment_existing_file(tmp_path: Path) -> None:
    """Loading an existing .env file should not raise any exception."""
    env_file = tmp_path / ".env"

    env_file.write_text(
        "OPENAI_API_KEY=test-key\n",
        encoding="utf-8",
    )

    load_environment(env_file)

    assert env_file.exists()
