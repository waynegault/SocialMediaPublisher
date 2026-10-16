# Python architecture audit findings

This is a prioritized, evidence-based working list. It records confirmed issues
and candidates still under investigation; it is not a claim that all repository
debt has been found or eliminated.

## Confirmed findings

| Priority | Finding | Evidence | Status |
| --- | --- | --- | --- |
| P1 | The test orchestrator could report success when a registered module had no test entry point, when Ruff timed out, or when Ruff was missing; its final exit code ignored a failed linter result. | `run_tests.py`: `_run_linter`, `_run_module_tests`, and `main`. | Fixed: failure paths are tested and lint gates the exit code. |
| P1 | The standalone indirect-people path ignores configured database selection and opens `content_engine.db` relative to the current directory. The application database is selected through `Config.DB_NAME`/`Database`. | `find_indirect_people.py`; `database.py:348-353`; `config.py:408`. | Fixed: uses `Config.DB_NAME`; targeted tests verify configured-path access and connection closure. |
| P0 | The analytics singleton test called the production constructor and created tables in persistent `content_engine.db`, violating test isolation. | `analytics_engine.py` singleton test and `_ensure_tables`. | Fixed: the test mocks construction, and no `content_engine.db` was created during verification. |
| P2 | The configuration directory test created the configured application image directory instead of using isolated temporary storage. | `config.py:_test_config_ensure_directories`. | Fixed: it points the setting at a temporary directory, restores the setting, and removes the directory after the assertion. |
| P1 | The analytics engine hard-coded a database file beside its source instead of using the configured application database. | `analytics_engine.py` constructor; `config.py` `DB_NAME`; `database.py` constructor. | Fixed: defaults to `Config.DB_NAME`. |
| P1 | SQLite context-manager use in the analytics engine commits/rolls back but does not close connections. This retained file handles on Windows, preventing temporary test databases from being removed. | `analytics_engine.py` database methods; reproduced by the analytics test suite on Windows. | Fixed: connections are explicitly closed after transaction scopes; temporary-database tests pass. |
| P2 | The indirect-people module reimplemented optional browser-driver import and organization-type heuristics, despite the repository having centralized browser availability (`browser.py`) and academic-entity patterns (`ner_engine.py`). Its broad import fallback could conceal an error inside the central module and silently select a second implementation. | `find_indirect_people.py`; `browser.py`; `ner_engine.py`. | Fixed: imports the central availability and academic patterns directly, logs malformed stored JSON, closes the configured database connection, and raises browser/search failures instead of returning success-shaped empty results. |
| P2 | Persistent domain entities use unvalidated dataclasses and ad hoc dictionaries; Pydantic is used only for application settings. External/API and serialized-data boundaries need an inventory before enforcing strict validation. | `models.py`; `database.py` (`Story`, `Person`); `config.py` (`SettingsModel`). | Partially addressed: strict Pydantic validation now protects both story-update routes; domain persistence and other API/LLM boundaries remain open. |
| P1 | Two HTTP handlers independently mutate the same story fields from unvalidated `request.get_json()` dictionaries, so malformed field types reach persisted domain objects and duplicate route behavior can drift. | `web_server.py` story update route; `validation_server.py` story update route. | Fixed for updates: both handlers share `StoryUpdateRequest` and `apply_story_update`; invalid payloads return client errors before persistence. |
| P2 | Oversized modules concentrate unrelated responsibilities and make verification difficult. Largest observed examples include `linkedin_profile_lookup.py` (~5.8k lines), `main.py` (~5.2k), and `company_mention_enricher.py` (~3.6k). | Python module line counts from the audit worktree. | Investigate |
| P2 | The test runner suppressed selected asyncio warnings and cleanup exceptions, then used `os._exit`, bypassing its cleanup callback and discarding stderr. | `run_tests.py` startup/cleanup and entry point. | Fixed: warnings/errors are visible and normal `SystemExit` runs cleanup. |
| P2 | Interpreter guidance and runtime checks accepted the legacy `venv` directory, while the test runner could be invoked with a global interpreter and then use that interpreter for Ruff. | `README.md` installation; `config.py:_check_venv`; `run_tests.py`. | Fixed: documented and enforced the repository `.venv` for tests, removed the legacy directory fallback, and limited configuration initialization handling to Pydantic validation errors. |
| P1 | History-cleanup scripts target an obsolete absolute checkout path; the PowerShell version falls back to global `pip`, and the scripts rewrite history while their placeholder replacement is a no-op. | `cleanup-history.ps1`; `cleanup-history.sh`. | Deferred: not executed or modified because these history-rewriting scripts can destroy or rewrite Git history; they require a dedicated safe migration/removal decision. |
| P2 | The LLM benchmark coverage is incomplete: the only benchmark entry point is for image providers, while text LLM work is performed by multiple pipeline components. | `run_benchmark.py:1-4`; text/LLM pipeline modules include `searcher.py`, `verifier.py`, and `company_mention_enricher.py`. | Open |
| P2 | Source verification and domain extraction caught every exception and returned normal-looking low-credibility/empty-domain results, conflating invalid input with programming/runtime failures. The constructor also used truthiness defaults, silently discarding explicit zero thresholds, and retained an unused duplicate domain matcher. | `source_verifier.py`; `domain_credibility.py:160-176`. | Fixed: invalid source types and URL parse failures propagate, malformed story source lists fail explicitly, configuration imports no longer fall back, zero thresholds are preserved, and the unused matcher is removed. Added regression tests and corrected contradictory expected-outcome text. |
| P2 | Repository tests are embedded in production modules and share a custom runner, rather than using independently discoverable tests. This makes test discovery and outcomes dependent on the bespoke orchestration path. | `run_tests.py` module registry; `_create_module_tests`/`run_comprehensive_tests` repeated across modules. | Investigate |
| P1 | `LOCAL_LLM_SEARCH_PROMPT` in `.env.example` cannot be parsed by python-dotenv: its multi-line quoted value contains a JSON example whose double quotes were not escaped, so the first of them terminated the value and the whole binding was dropped. A copied `.env` silently lost a prompt that `searcher.py` formats, and this file is now what CI copies. | `dotenv.main.parse_stream` reports an error at line 227 with `key=None`; raw `KEY=` names 147 vs 146 parsed, `LOCAL_LLM_SEARCH_PROMPT` missing; the value is used at `searcher.py:874` and `searcher.py:1677`. | Fixed: the inner quotes are escaped, `.env.example` now parses 147/147 with no warning, and `config.py` gained a regression test asserting every binding parses. |
| P2 | A cache test was racy by construction. `test_sqlite_cache_ttl` gave its pre-expiry read only a 0.1 s default TTL, so on a loaded machine the entry could expire between `set()` and the first `get()`, failing an assertion that has nothing to do with expiry. A latent source of random CI failures. | `cache.py` `test_sqlite_cache_ttl`; failed in a 427 s full-suite run whose only failure was `Cache` / "SQLite cache TTL", and again in isolation while the machine was busy; passed three times when idle, and the TTL logic itself was verified correct. | Fixed: the pre-expiry read uses a 30 s default TTL, and expiry is exercised on a second key with `ttl=0.1` — deterministic on an idle or a loaded machine. |

## Still to investigate

- Whether tests can read or mutate the real application database or other persistent
  user data, including on failure before cleanup.
- Pydantic validation gaps at request, persistence, and LLM-output boundaries.
- Duplicated evidence/classification models, non-idempotent public operations, and
  overlapping utility implementations.
- Unfinished or unwired code, unused compatibility layers, and incorrect file names.
- Test assertions that are vacuous, duplicated, or not reflected in process exit
  status; whether LLM pipeline tasks have benchmark coverage.
- Hard-coded Python executables in scripts, unresolved circular imports, suppressed
  errors elsewhere, and relevant log/diagnostic quality.

## Follow-up TODO list

These are proposed next work items, ordered to protect data and improve confidence
before broad refactoring. Treat each item as its own reviewable batch and update
the findings table with evidence, disposition, and targeted test results.

### P0 — Test safety and trustworthy results

- [ ] Audit every test and test helper for writes to the configured database,
  image/output directories, credentials, network services, and other persistent
  state. Replace real resources with temporary directories/databases and mocked
  external services; prove isolation on both success and failure paths.
- [x] Identify and gate tests that contact real APIs, browsers, or LLM providers.
  The `linkedin_profile_lookup` live test now runs only in `--integration` mode,
  and a regression test pins that (`8e3397a`). The gated test is now *named* in the
  summary rather than omitted: `TestSuite.skip()` records it with its reason,
  `finish_suite()` lists skipped tests separately from failures, and the runner
  reports them per module. One part of this item remains open: provider
  selection/fallback is not yet unit-tested with fakes.
- [ ] Make a clean `.venv` setup reproducible, then run the complete unit suite
  and lint/type checks through `.venv` only. Record missing optional dependencies
  separately instead of treating a partial run as full-suite validation.

### P1 — Validate boundaries and cover LLM behavior

- [ ] Inventory all HTTP request, database deserialization, configuration, and
  LLM-output boundaries. Add strict Pydantic models where untrusted or serialized
  data enters domain logic; reject unknown/malformed fields and test invalid
  values before side effects.
- [ ] Add deterministic text-LLM benchmark/evaluation coverage for search
  distillation and story processing, verification, source/people extraction,
  company mentions, and originality assessment. Use fixed cases and expected
  quality criteria; report selected and fallback providers separately, latency,
  failures, and output validity. Keep routine unit tests fully offline.
- [ ] Audit success-shaped fallbacks and broad exception catches. Preserve
  intended resilience, but distinguish unavailable/invalid output from valid
  empty results; log or propagate failures with task/provider context and add
  tests for each failure path.
- [ ] Audit mutating operations for idempotency, especially publishing,
  scheduling, story updates, migrations, and database writes. Define the
  duplicate/retry behavior and verify repeated calls do not create duplicate
  side effects.

### P2 — Simplify structure and consolidate shared logic

- [ ] Replace embedded production-module test suites and the custom test
  registry with independently discoverable tests in dedicated test modules.
  Keep a thin standard test command, preserve meaningful test outcomes, and
  migrate incrementally so each batch remains runnable.
- [ ] Decompose the largest mixed-responsibility modules, starting with
  `linkedin_profile_lookup.py`, `main.py`, and `company_mention_enricher.py`.
  Move cohesive responsibilities behind explicit interfaces; avoid changing
  behavior in the same batch as structural moves and add characterization tests
  before extraction.
- [ ] Compare evidence, source credibility, organization/person classification,
  and entity matching implementations. Consolidate only where semantics match;
  document any intentional distinctions and remove duplicate models/helpers
  once callers and tests are migrated.
- [ ] Search for unfinished stubs, no-op code, dead branches, unused modules,
  unnecessary compatibility shims, poorly named files, and hard-coded
  interpreter paths. Remove or wire each confirmed case and test the resulting
  command/API behavior.
- [ ] Review import dependencies for cycles and oversized public interfaces.
  Extract shared low-level types/helpers only where needed to remove a
  demonstrated cycle or duplicated implementation.
- [ ] Review logging and warning behavior for suppressed diagnostics, vague
  messages, secrets/PII exposure, and swallowed cleanup failures. Keep expected
  fallback warnings observable and add tests that verify important errors reach
  logs or callers.

### Deferred — History cleanup scripts

- [ ] Decide whether to delete or rewrite `cleanup-history.ps1` and
  `cleanup-history.sh`. Do not execute either script as-is: they rewrite Git
  history and contain stale absolute paths. If retained, first make paths
  repository-relative, remove global-Python fallbacks, correct the no-op
  placeholder replacement, and document a dry-run/backup/recovery procedure.

## Verification and disposition

The initial baseline was not valid: imports failed because declared packages
were absent from the newly created repository `.venv`. Installing the complete
requirements pulled in a large optional ML stack and was stopped after more
than 16 minutes; lightweight dependencies were installed for targeted tests.

### Review and verification on `main` (2026-10-15)

The eight fixing commits were landed on `main` by fast-forward (`main` was the
merge base; the branch was 0 behind, 8 ahead). Verification used the complete
repository interpreter
`C:\Users\wayne\GitHub\Python\Projects\SocialMediaPublisher\.venv\Scripts\python.exe`
(Python 3.13.15) rather than the audit worktree `.venv` (Python 3.14.7), which is
missing `google-genai`.

- Targeted re-run of every module the branch changed — `test_run_tests`,
  `test_story_update_api`, `config`, `domain_credibility`, `source_verifier`,
  `find_indirect_people`, `analytics_engine` — passed **7/7**.
- The **full** `run_tests.py` run does **not** complete on `main`. It wedges in
  `linkedin_profile_lookup` (test 12, "Lookup person URN - no driver") on a real
  browser launch followed by a blocking `After logging in, press Enter to
  continue...` prompt; the run was terminated after 600 s. In the audit worktree
  this module failed to import (`google.genai` absent), so its tests never ran
  and the hang was invisible — a complete local `.venv` is what reproduces it.
- The persisted databases were byte-identical before and after the interrupted
  run (`sha256` compared for `content_engine.db`, `cache.db`, `socialmedia.db`,
  `social_publisher.db`), so no user data was touched by the partial run.
- `run_benchmark.py` carried three `F541` violations. Because the linter now
  gates the exit code they became build-breaking at merge time, so they were
  auto-fixed (`ruff check run_benchmark.py --fix`); `ruff check .` now reports
  `All checks passed!`.

#### The lint gate is environment-dependent (new finding)

The gate's outcome depends on the installed Ruff version: `requirements.txt`
pins only `ruff>=0.5.0`, and the repository carries **no** Ruff configuration
(no `pyproject.toml`, `ruff.toml` or `.ruff.toml` in the tree or any parent, and
no user-level config under `%APPDATA%\ruff`).

- Ruff **0.15.0** (the interpreter above): 3 errors, all `F541`.
- Ruff **0.17.0** (the audit worktree): **1104 errors**, even under `--isolated`
  — `BLE001` (364), `UP045` (343), `UP017` (59), `I001` (59), `S110` (57), and
  roughly 30 further codes.

Either interpreter pointed at the other tree reproduces the same 3-vs-1104
result, so this is a change in Ruff's default rule selection, not a property of
the code. Until the rule set is pinned or declared explicitly, "the linter
passes" only holds for whichever Ruff happens to be installed.

**Resolved (2026-10-16).** `requirements.txt` pins `ruff==0.17.1` and
`ruff.toml` declares the rule set explicitly, so the gate no longer depends on
which Ruff is installed. The declared set is the pyflakes/structural core the
repository already enforced, plus import sorting and a few families with no
behavioural risk; the families left out (`E501`, `UP`, `BLE`, `S`, `B`, `SIM`,
`N`, `PYI`, `SLF`) are named in the file with the reason, so the choice is
explicit rather than accidental. Applying it surfaced 117 findings, all
mechanical — import sorting, a redundant `pass` after a docstring, trailing
whitespace on blank lines, a missing newline at EOF, parenthesised implicit
concatenation, and single-call `startswith`/`endswith`. `ruff check .` now
reports `All checks passed!` under 0.17.1 and under 0.15.0 alike.

### Review and verification on `main` (2026-10-16) — the CI gate

`main` had no CI. Adding one exposed three defects in sequence, all now fixed;
run 38932536707 is `success` in 3m30s with `Modules Passed: 57/57` in 109.5 s.

- The suite could not finish at all — the P0 above. The live test is now opt-in
  and the job is bounded at 30 minutes (`8e3397a`).
- The lint step failed with 1104 findings, because `ruff>=0.5.0` resolved to a
  Ruff whose default rule set is far wider than this repository's. Now pinned and
  declared (`3880bcc`).
- Three prompt-building tests failed on an empty prompt because the runner had no
  `.env`; the job now copies `.env.example` (`f168a42`). The three `F541`
  findings in `run_benchmark.py`, build-breaking once lint gates the exit code,
  were landed (`a28d207`).
- A fourth defect appeared only under load: `test_sqlite_cache_ttl` was racy by
  construction — a 0.1 s TTL on its pre-expiry read — which would have reddened
  the gate at random. Fixed by giving that read a comfortable TTL and exercising
  expiry on a second key.

Two regression tests were added so these failures cannot return silently: the
unit suite must complete with stdin closed (`test_run_tests.py` — it would have
caught the wedge in seconds rather than a 29-minute job), and every `KEY=`
binding in `.env.example` must parse (`config.py`).

### Test-safety audit (P0) — evidence gathered

Auditing tests for writes to persistent state (the first P0 item) produced three
confirmed, reproducible problems, all observed through the repository `.venv`
interpreter on `main`.

| Priority | Finding | Evidence | Status |
| --- | --- | --- | --- |
| P0 | A unit-test module launches a real browser and blocks on stdin. The `linkedin_profile_lookup` suite reaches a real LinkedIn login prompt and waits for input, so `run_tests.py` cannot finish; there is no timeout and no gate on the integration flag. | Full `run_tests.py` run on a complete `.venv` wedged for >600 s at test 12 of `linkedin_profile_lookup`. | Fixed (2026-10-16): the live test is registered only when `SKIP_LIVE_API_TESTS=false` (`--integration`); the CI job is bounded at 30 minutes; and a regression test runs the unit suite with stdin closed, which fails in seconds instead of wedging. |
| P1 | Tests create a real application database file. The `linkedin_engagement`/`linkedin_networking` suites construct their classes with no `db_path`, and both default to `Path(__file__).parent / "content_engine.db"`; a worktree run left a fresh 24 KB `content_engine.db` holding `connection_requests`, `engagement_actions` and `engagement_targets`. This is the defect class already fixed in `analytics_engine.py`, and the hard-coded path also ignores `Config.DB_NAME`. | `linkedin_engagement.py:184`, `linkedin_networking.py:187`; file created during a `run_tests.py` run. | Fixed (2026-10-16): the two singleton tests now run against a temporary database — verified by `content_engine.db` being byte-for-byte untouched across a run — and both classes default to `Path(__file__).parent / Config.DB_NAME`, so the configured name is honoured while the path stays independent of the working directory. |
| P1 | The same two modules opened SQLite connections with `with sqlite3.connect(...) as conn:`, which commits but does not close. The retained handle keeps the file open on Windows and prevents a temporary database from being removed — the isolation fix above exposed it. The same defect was already fixed in `analytics_engine.py`. | Both suites, 2026-10-16: `PermissionError: [WinError 32] ... tmp...\singleton.db` raised by `TemporaryDirectory` cleanup, in both modules. | Fixed: each module gained a `_connection()` context manager that commits on success and always closes; all 11 call sites converted. |
| P2 | `Config`'s metaclass routes attribute assignment onto the private settings object, so the name never lands in `Config.__dict__`. `unittest.mock.patch.object(Config, "DB_NAME", ...)` therefore fails on exit with `AttributeError: type object 'Config' has no attribute 'DB_NAME'`, because `patch` calls `delattr` to undo an attribute it can only reach through `__getattr__`. | Reproduced while isolating the two singleton tests. | Worked around in the tests by patching the backing settings object instead; the metaclass is unchanged. |
| P2 | Heavy ML imports run at module import time. Re-measured 2026-10-16: `import ner_engine` takes **36.4 s** (`sentence_transformers` 40.2 s cold, `spacy` 11.3 s, `browser` 2.0 s) — roughly a third of a full suite run, which is ~110 s. The cost is not in one module: three application modules import the stack at module level — `ner_engine.py` (spacy, sentence_transformers, numpy), `intent_classifier.py` (transformers pipeline) and `rag_engine.py` (sentence_transformers). Whichever imports first pays it, so all three must change together for the run to get faster. `generate_image.py`, `image_providers/z_image.py` and `image_providers/factory.py` already import torch lazily inside functions, which is the pattern to follow. The original 170.0 s figure no longer reproduces. | `python -c "import time; t=time.time(); import ner_engine; print(time.time()-t)"`, plus each library measured on its own. | Open |

Observed but not yet reproduced: the loaded `.env` reports empty LLM prompt
templates (`SEARCH_PROMPT`, `SEARCH_INSTRUCTION_PROMPT`, `VERIFICATION_PROMPT`,
`IMAGE_REFINEMENT_PROMPT`, `IMAGE_FALLBACK_PROMPT`, `IMAGE_STYLE`), so the text
and image LLM paths appear misconfigured in this checkout.

**Explained and fixed (2026-10-16).** The templates were not empty, they were
absent: the local `.env` defines none of the seven prompt keys, so configuration
reported them empty and the three prompt-building tests failed. A clean checkout
has no `.env` at all (it is ignored), which is exactly why CI failed the same
way. Fixed on both sides — the CI job copies the committed `.env.example`
(`f168a42`), and the local `.env` was reconciled for prompt templates: 21 of the
29 `*_PROMPT`/`*_STYLE` keys in the example were absent, and all are now
present (every one round-tripped exactly, and no credential key was touched).
Reconciling them surfaced the `.env.example` parsing defect recorded in the
confirmed findings above.
