import nox

# Set uv as the default virtual environment backend for faster resolution
nox.options.default_venv_backend = "uv"

# Define the default execution order (lint first, then tests)
nox.options.sessions = ["lint", "tests"]

@nox.session(python=False)
def lint(session: nox.Session) -> None:
    """Run code quality checks (Ruff, Mypy) via pre-commit."""
    # Using python=False skips virtualenv creation for this session,
    # leveraging the tools already installed by uv in the dev environment.
    session.run("uv", "run", "pre-commit", "run", "--all-files", external=True)

@nox.session(python=["3.12", "3.13"])
def tests(session: nox.Session) -> None:
    """Run matrix tests using pytest and generate coverage reports."""
    env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
    
    # Install the project and development dependencies inside the nox environment
    session.run_install("uv", "sync", "--group", "dev", env=env)
    
    # Run pytest with code coverage flags
    session.run("pytest", "--cov", "--cov-branch", "--cov-report=xml", *session.posargs)
