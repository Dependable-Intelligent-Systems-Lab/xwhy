import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["tests"]

@nox.session(python=["3.12", "3.13"])
def tests(session: nox.Session) -> None:
    env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}

    session.run_install("uv", "sync", "--group", "dev", env=env)

    session.run("pytest", *session.posargs)