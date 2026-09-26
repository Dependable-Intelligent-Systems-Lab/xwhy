# JFrog security scanning

## Status and safe activation

The two workflows are prepared for **Frogbot V3 3.7.0**, pinned to the
corresponding upstream action commit and binary version. They are **inactive**
until an XWhy maintainer sets the repository Actions variable
`JFROG_FROGBOT_V3_READY` to the exact string `true`. An unset value produces a
skipped job. Do not set it until the checks below are complete. The first V3
scan permanently switches this Git repository from Frogbot V2 to V3 in JFrog;
JFrog says this change cannot be reversed. Whether XWhy already has V2 scan
history has not been verified. A test scan of XWhy is therefore **not** an
appropriate way to discover its V2 state.

The integration uses GitHub OIDC for short lived JFrog access, never a stored
JFrog access token. It makes no changes to application code, `pyproject.toml`,
`uv.lock`, `.env.example`, or the four existing CI/CD workflows.

### Required checks before setting the readiness variable

1. In the organisation's JFrog administration pages, record the JFrog Platform
   and Xray versions and verify the deployed combination supports Frogbot V3.
   **Xray must be at least 3.143.6.** For self hosted JFrog, also confirm
   Catalog Service and Transitive SBOM are enabled. The actual instance and its
   Platform/Xray versions are not accessible from this GitHub repository.
2. Ask the JFrog administrator whether this exact Git repository is already
   scanned with Frogbot V2 and accept the irreversible V3 transition. Confirm
   the organisation's subscription and enabled features in the JFrog instance;
   the status of each feature in the table below is currently unverified.
3. Pre-create a **Generic local** Artifactory repository named `frogbot`, or
   `<project-key>-frogbot` inside a JFrog Project. Add it to **Administration →
   Xray Settings → Indexed Resources**. Grant the OIDC identity Deploy rights
   to this repository and only the remaining permissions required for scans and
   reading their results. Do not give Actions an Artifactory administrator role.
   Verify repository existence, indexing, and permissions in JFrog; none can be
   inferred from GitHub.
4. In **Administration → Xray Settings → Indexed Resources → Git Repositories**,
   select XWhy's `default` workspace and review its effective configuration
   (including inherited settings). **Turn off Create automated fixes** on the
   Auto-PR tab: JFrog documents that it is enabled by default. Disable any
   policy that fails the PR on an existing finding until a baseline is reviewed.
   Keep secret values out of public PR comments. Configure scanners, PR
   decorations, licence policy, exclusions, and any severity rules there.
   Do not set exclusions merely to silence findings.
5. Set up the OIDC provider and identity mappings described below, then create
   the `frogbot` GitHub Environment with required reviewers. Set repository
   Actions variables `JF_URL` (the full HTTPS JFrog Platform URL) and
   `JFROG_OIDC_PROVIDER_NAME` (the exact configured provider name). These are
   configuration values, not credentials. Test claims and permissions without
   invoking Frogbot on XWhy. Check the settings and version/entitlement matrix
   once more before setting `JFROG_FROGBOT_V3_READY=true`.

### Capabilities to verify in the subscription

| Capability | Intended setting after entitlement check | Current status |
| --- | --- | --- |
| SCA and resolved dependency vulnerabilities | Enable in JFrog | Unverified |
| CVE detection and advisory data | Enable with SCA | Unverified |
| Python SAST | Enable if licensed | Unverified |
| Secrets detection | Enable if licensed; keep values out of public PR comments | Unverified |
| CVE contextual analysis | Enable if licensed and Python scan coverage is confirmed | Unverified |
| Dependency licence policy | Set the approved licence policy in JFrog | Unverified |
| IaC and workflow misconfiguration scans | Enable if licensed; confirm applicable files | Unverified |
| Supply chain and snippet risk analysis | Enable only where entitled and supported | Unverified |
| Remediation/automatic fixes | **Disabled** during baseline; reconsider separately | Licence unverified |

JFrog documents Python SCA, SAST, contextual analysis, and secret scanning as
supported technologies. This does **not** prove that this JFrog subscription
includes them or that this particular lockfile is ingested correctly.

## Workflows and fork behaviour

`.github/workflows/frogbot-scan-pull-request.yml` runs for PRs targeting
`main` on `opened`, `reopened`, and `synchronize`. Frogbot compares the source
and target branches and posts new findings to the PR where supported. Its
`pull_request_target` workflow definition comes from the trusted upstream
default branch. The checkout uses the **upstream base commit**, with Git
credential persistence disabled. There is no `run` step that executes PR code,
no `uv sync`, and no PR-head checkout in the workflow. Frogbot V3 retrieves the
source and target as scan data and uses JFrog's static dependency engine. Its
scan process still parses untrusted content; do not add code execution to this
privileged job. GitHub's `frogbot` Environment requires human approval before
the PR job obtains an OIDC token. Review the PR and any workflow changes before
approval. A PR from a fork into upstream is eligible for this reviewed scan.

`.github/workflows/frogbot-scan-repository.yml` runs daily at **04:23 UTC**
and can be started with **Actions → Frogbot Scan Repository → Run workflow**.
It only runs on upstream `main`, including for manual dispatch. It has no
`push` trigger, avoiding a second scan for every ordinary CI push. This scan
can reveal newly disclosed issues in unchanged dependencies and uploads results
to JFrog's **Xray → Scans List → Git Repositories**. GitHub Dependency Graph
SBOM upload is disabled in this workflow, so no GitHub repository write
permission is needed; JFrog's own SBOM upload remains enabled.

Every JFrog job checks the canonical repository name **before** requesting an
Environment, checking out code, or performing JFrog setup. An independent fork
such as `external-user/xwhy` intentionally reports **skipped**, including if
the fork lacks variables, secrets, OIDC trust, or the Environment. This is not
a CI failure. The repository scanner also checks `github.ref` is `main`.

## OIDC, GitHub permissions, and Environment

In JFrog, create a GitHub OIDC integration with the provider name stored in
`JFROG_OIDC_PROVIDER_NAME`, using issuer/provider URL
`https://token.actions.githubusercontent.com`. Configure the action's explicit
audience `https://github.com/Dependable-Intelligent-Systems-Lab` and validate
the `aud` claim in JFrog's Identity Mapping; JFrog's integration Audience field
is distinct from a mapping rule for the JWT `aud` claim. Match the exact
`repository` and `repository_owner` claims. Use separate, narrow mappings for:

| Job | Claims to restrict beyond repository and audience |
| --- | --- |
| PR | `event_name=pull_request_target`, `sub=repo:Dependable-Intelligent-Systems-Lab/xwhy:environment:frogbot`, `workflow_ref=Dependable-Intelligent-Systems-Lab/xwhy/.github/workflows/frogbot-scan-pull-request.yml@refs/heads/main` |
| Daily | `event_name=schedule`, `ref=refs/heads/main`, `workflow_ref=Dependable-Intelligent-Systems-Lab/xwhy/.github/workflows/frogbot-scan-repository.yml@refs/heads/main` |
| Manual | `event_name=workflow_dispatch`, `ref=refs/heads/main`, the same repository-scan `workflow_ref` |

Check the actual claims against JFrog's OIDC test/diagnostic view before
activation; do not print JWTs in Actions logs. Use a short JFrog token lifetime
that still covers a complete XWhy scan. The identity needs scan and SBOM
upload rights and Deploy permission on the indexed Generic repository, not
administrator rights. If JFrog Projects are used, configure the appropriate
project key centrally and check its policy effects. OIDC failure should fail
the scan, never trigger a fallback to a stored access token.

Create a GitHub Environment named exactly `frogbot` in **Settings →
Environments**. Add at least one authorised maintainer or public team as a
required reviewer, disable self review where available, and limit deployment
to `main` if GitHub offers that protection. The PR job references this
Environment; the daily repository job does not, so scheduled scans do not
wait each day for review. No long lived JFrog secret belongs in either
Environment or repository. `JF_URL`, `JFROG_OIDC_PROVIDER_NAME`, and the
readiness switch belong to repository Actions **Variables**.

The PR job has `contents: read` to read the trusted base and public fork code;
`pull-requests: write` lets Frogbot decorate the upstream PR; and
`id-token: write` requests OIDC only in that job. The repository job has
`contents: read` and `id-token: write`. Frogbot uses the scoped GitHub
`GITHUB_TOKEN` as `JF_GIT_TOKEN`; no personal access token is stored.
Neither job receives `contents: write` or `security-events: write`. Auto-PR
cannot create a branch with these GitHub permissions; ensure it is also
disabled in JFrog. GitHub Code Scanning SARIF upload is not enabled. Enabling
it later requires a separate review of `security-events: write` and the
JFrog V3 PR upload setting.

## Python graph and scan scope

XWhy requires Python **3.12–3.13**, uses `pyproject.toml`, and commits a
`uv.lock` with a resolved transitive dependency graph. The current lockfile
contains 282 package entries, including `openai==3.0.0`,
`anthropic==0.122.0`, `torch==2.13.0`, `transformers==5.15.0`, and
`scikit-learn==1.9.0`. This is a **repository inspection**, not proof they
appear in a JFrog report. JFrog V3 describes build independent static scans,
so these workflows do not install Python dependencies or invoke `uv`. Support
for the exact `uv.lock` format and coverage of optional/development groups
must be confirmed in a real Xray scan. Do not replace the lockfile with a
generated requirements file or claim `uv` coverage from a green workflow
alone. If detection fails, investigate the JFrog scanner version and a safe,
separate export of the locked graph before proposing changes.

No custom scan path exclusions have been introduced. Inspect the effective
central settings to ensure `src/`, `tests/`, `docs/`, notebooks, examples,
generated documentation, static site files, and `data/` are assessed as
appropriate. Record any later exclusions with a reason. `.env.example`
contains **empty** provider key examples (OpenAI, Anthropic, Google, AWS and
others), not key values; keep these examples and investigate actual nonempty
secrets as urgent findings. Treat test/example findings separately from
production library findings when triaging, rather than hiding their paths.

The JFrog **repository/workspace configuration** is authoritative for scanner
selection, exclusions, severity and licence policies, PR decoration, and
Auto-PR. JFrog environment variables override platform settings; these
workflows set only connection data, branch, audience, and
`JF_UPLOAD_SBOM_TO_VCS=false` for the repository job. Do not add a V2
`frogbot-config.yml` or V2-only environment variables.

## Validation and policy after activation

After setup, set the readiness variable, manually dispatch the repository
scan **from `main`**, and save its run URL. Verify OIDC succeeds, the Xray
version and indexing are compatible, the report appears in JFrog, and the
SBOM contains the resolved XWhy packages and their transitive relationships.
Check SCA counts by Critical/High/Medium/Low; SAST, secrets, licence and
contextual findings; false positives; and source versus test/example paths.
Confirm the job logs do not expose tokens or JWTs. Open a normal upstream PR
and a reviewed fork-originated upstream PR; verify PR comments and Xray
results. In an independent fork, trigger the inherited workflow and confirm
the JFrog job is **skipped**. Check the existing `ci.yml` and docs checks as
usual. These live checks and baseline counts **have not yet been performed**.

Initially use findings for visibility. After triage, consider blocking newly
introduced Critical and demonstrably exploitable High findings, with a
documented exception process for disputed results. Configure thresholds in
JFrog after the baseline; do not make historical findings block unrelated
development. A genuine scanner or authentication error should still be fixed,
not silently treated as a clean scan. To rerun, dispatch the repository
workflow on `main` or reopen/update a PR; the Environment reviewer must
approve a new privileged PR run when required.

### Troubleshooting

- **Skipped on upstream:** check the exact readiness variable value and,
  for manual scans, that `main` was selected. **Skipped in a fork is expected.**
- **Waiting on PR review:** an authorised `frogbot` Environment reviewer must
  inspect and approve the run. Check protection settings if the Environment
  is absent or not configured with reviewers.
- **OIDC denied:** check issuer, audience, provider name, exact repository,
  Environment subject, workflow ref, event claim, and token lifetime without
  printing the JWT. There is no access-token fallback.
- **Missing SBOM/index:** check the Generic `frogbot` repository name, Xray
  indexing, and identity Deploy permission. Check `JF_URL` and JFrog versions.
- **Missing dependencies:** confirm `uv.lock` was scanned and compare the Xray
  SBOM with the package examples above; do not treat a successful SAST or
  secret-only scan as proof of SCA coverage.
- **Auto-PR or policy failure:** disable automated fixes and initial fail rules
  in the effective XWhy workspace configuration. The workflow cannot write
  repository contents by design.

Official references: [Frogbot V3](https://docs.jfrog.com/security/docs/frogbot),
[JFrog GitHub integration](https://docs.jfrog.com/security/docs/github),
[V3 configuration](https://docs.jfrog.com/security/docs/advanced-management-and-configuration),
[scan behaviour](https://docs.jfrog.com/security/docs/how-to-commit-scan-and-pr-scan),
and [GitHub's `pull_request_target` security guide](https://docs.github.com/en/actions/reference/security/securely-using-pull_request_target).
