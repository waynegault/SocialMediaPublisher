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

## Verification and disposition

The initial baseline was not valid: imports failed because declared packages
were absent from the newly created repository `.venv`. Installing the complete
requirements pulled in a large optional ML stack and was stopped after more
than 16 minutes; lightweight dependencies were installed for targeted tests.
Every validation command uses `.venv\\Scripts\\python.exe`.
