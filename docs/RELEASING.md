## xwhy Release Procedures & Checklist

This document outlines the step-by-step workflow for releasing a new semantic version of the `xwhy` package. Following this exact sequence ensures that our automated GitHub Actions pipeline successfully runs multi-version tests via Nox, publishes the distribution to PyPI using Trusted Publishing (OIDC), and generates a clean GitHub Release with attached assets.

---

### Pre-Release Sanity Check

Before updating any versions, make sure your local environment is up to date and your working directory is clean.

* Run `git pull origin main` to ensure you have the latest upstream changes.
* Run `uv run nox` locally to ensure all linting checks and test matrices pass seamlessly.

---

### The Release Sequence

1. **1. Bump the Version:** pyproject.toml.
Open your `pyproject.toml` file and update the `version` field under the `[project]` table following **Semantic Versioning** rules (`MAJOR.MINOR.PATCH`).

```toml
[project]
name = "xwhy"
version = "0.1.0"  # Update this line accordingly

```


2. **2. Commit the Version Change:** Git Commit.
Stage and commit the change to `pyproject.toml`. Use a clear, conventional commit message to keep the git history semantic.

```bash
git add pyproject.toml
git commit -m "chore: bump version to 0.1.0"

```


3. **3. Create a Semantic Git Tag:** Git Tagging.
Create a local Git tag that matches your new version exactly. **Always prefix the tag with a lowercase `v**` (e.g., `v0.1.0`) to match the pattern expected by the deployment workflow.

```bash
git tag v0.1.0

```


4. **4. Push to GitHub:** Git Push.
Push both your commit and the newly created tag to the remote repository. Pushing the tag is the definitive action that triggers the `publish.yml` pipeline.

```bash
git push origin main
git push origin v0.1.0

```


5. **5. Monitor the Automation Pipeline:** GitHub Actions.
Navigate to the **Actions** tab of your GitHub repository and track the progress of the **Publish to PyPI and GitHub Releases** workflow. The automation will:

* Perform a final pre-release sanity check via Nox.
* Build the source distribution and wheel packages using `uv build`.
* Authenticate securely with PyPI via OIDC and upload the package.
* Compile an automated Changelog based on recent commits and PRs.
* Publish an official GitHub Release with the compiled Changelog and attach the built files from `dist/*` as release assets.


---

> **Crucial Rule:** Never modify the `version` string in `pyproject.toml` without immediately tagging that exact commit. If the version inside the code does not match the Git tag, PyPI will reject the deployment due to version conflicts, or your GitHub Releases page will display mismatched metadata.