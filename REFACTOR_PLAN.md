# Social Media Publisher - Comprehensive Refactoring Plan

## Executive Summary

This refactoring plan addresses code duplication, redundant API calls, inconsistent caching strategies, and process flow consolidation identified during a thorough codebase review. The goal is to minimize duplicate API/search calls, streamline processes, and ensure all `.env` settings are properly honored.

---

## Phase 1: Domain Credibility Consolidation (HIGH PRIORITY) ✅ COMPLETED
**Goal:** Eliminate duplicate domain lists and create a single source of truth for domain credibility.

### Changes Completed:
- [x] 1.1 Create new `domain_credibility.py` module with unified domain tiers
- [x] 1.2 Export `TIER_1_SOURCES`, `TIER_2_SOURCES`, `TIER_3_SOURCES`, and `ALL_REPUTABLE_DOMAINS`
- [x] 1.3 Update `source_verifier.py` to import from `domain_credibility.py`
- [x] 1.4 Update `searcher.py` to import from `domain_credibility.py`
- [x] 1.5 Add `.gov` and `.edu` pattern recognition to credibility checks

### Impact Achieved:
- Removed ~100 lines of duplicate domain definitions
- Single place to update domain lists
- Consistent credibility scoring across modules

---

## Phase 2: LinkedIn Cache Consolidation ✅ COMPLETED
**Goal:** Unify all LinkedIn caching under the `LinkedInCache` class from `cache.py`.

### Completed Changes:
- [x] 2.1 Added new methods to `LinkedInCache` in `cache.py`:
  - `get_person_by_name()` / `set_person_by_name()` for cross-company person lookups
  - `get_company_canonical_name()` / `set_company_canonical_name()` for reverse URL→name mapping
  - `get_department()` / `set_department()` for department page caching
- [x] 2.2 Updated `linkedin_profile_lookup.py` to integrate with `LinkedInCache`:
  - Added `_linkedin_cache` instance using `get_linkedin_cache()`
  - Added helper methods: `_cache_person_result()`, `_get_cached_person()`, `_cache_company_result()`, `_get_cached_company()`, `_cache_department_result()`, `_get_cached_department()`
  - Marked class-level dict caches as DEPRECATED (backward compatible)
  - Updated `get_cache_stats()` to include unified cache stats
- [x] 2.3 `linkedin_voyager_client.py` already uses `LinkedInCache` with dict fallback

### Remaining Future Work (optional cleanup):
- Remove deprecated dict caches from `linkedin_profile_lookup.py` after full migration
- Update `linkedin_rapidapi_client.py` to use `LinkedInCache` directly
- Migrate `ValidationCache` to use `LinkedInCache` for person lookups

### Impact Achieved:
- LinkedIn profile lookups now use persistent SQLite-backed caching
- Consistent TTL handling via LinkedInCache
- Backward compatible with existing dict-based code

---

## Phase 3: HTTP Session Consolidation ✅ COMPLETED
**Goal:** Ensure all HTTP requests use the centralized `api_client` for rate limiting and retry logic.

### Completed Changes:
- [x] 3.1 Added `http_head()` and `http_put()` convenience methods to `RateLimitedAPIClient`
- [x] 3.2 Updated `searcher.py` `_archive_url()` to use `api_client.http_head()`
- [x] 3.3 Updated `linkedin_publisher.py` binary upload to use `api_client.http_put()`
- [x] 3.4 Updated `image_generator.py` to use `api_client.http_get()` (3 locations)
- [x] 3.5 Updated `linkedin_profile_lookup.py` to use `api_client.linkedin_request()`
- [x] 3.6 Updated `notifications.py` Slack webhook to use `api_client.http_post()`
- [x] 3.7 Removed unused `httpx` imports from `image_generator.py` and `linkedin_profile_lookup.py`
- [x] 3.8 Removed unused `requests` import from `linkedin_publisher.py`
- [x] 3.9 Removed unused `urllib` imports from `notifications.py`

### Impact Achieved:
- All HTTP calls now rate-limited via centralized client
- Consistent retry logic across all modules
- Better quota management with per-endpoint tracking
- Reduced direct dependencies on httpx/urllib

---

## Phase 4: URL Utilities Consolidation ✅ COMPLETED
**Goal:** Create shared URL validation/parsing utilities.

### Issues Resolved:
- `searcher.py` had `validate_url()` with URL parsing
- `searcher.py` had `extract_url_keywords()` with duplicate logic
- `image_generator.py` had inline relative URL resolution
- `company_mention_enricher.py` had duplicate LinkedIn URL validation

### Changes Completed:
- [x] 4.1 Created `url_utils.py` module with shared functions
- [x] 4.2 Implemented `extract_path_keywords(url)` - extract keywords from URL path
- [x] 4.3 Implemented `validate_url_format(url)` - basic URL validation
- [x] 4.4 Implemented `validate_url_accessible(url, timeout)` - HTTP check
- [x] 4.5 Implemented `validate_url(url)` - combined validation
- [x] 4.6 Implemented `resolve_relative_url(relative, base)` - relative URL resolution
- [x] 4.7 Implemented `get_base_url(url)` - extract scheme://netloc
- [x] 4.8 Implemented `validate_linkedin_url(url, type)` - LinkedIn URL format check
- [x] 4.9 Updated `searcher.py` to use `url_utils`
- [x] 4.10 Updated `image_generator.py` to use `url_utils.resolve_relative_url`
- [x] 4.11 Updated `company_mention_enricher.py` to use `url_utils.validate_linkedin_url`
- [x] 4.12 Added url_utils tests to run_tests.py (17 tests)

### Impact Achieved:
- Removed ~80 lines of duplicate code
- Consistent URL handling across codebase
- 17 new unit tests for URL utilities

---

## Phase 5: LinkedIn Data Models Unification ✅ COMPLETED
**Goal:** Create unified data classes for LinkedIn profile/company data.

### Completed Changes:
- [x] 5.1 Unified models already existed in `models.py`:
  - `LinkedInProfile` with superset of all fields
  - `LinkedInOrganization` with superset of all fields
  - Factory methods: `from_voyager_person()`, `from_rapidapi_result()`, `from_voyager_org()`
- [x] 5.2 Updated `linkedin_rapidapi_client.py`:
  - Removed local `LinkedInProfileResult` dataclass (50+ lines)
  - Import `LinkedInProfile` from `models.py`
  - Added alias `LinkedInProfileResult = LinkedInProfile` for compatibility
- [x] 5.3 Updated `linkedin_voyager_client.py`:
  - Removed local `LinkedInPerson` and `LinkedInOrganization` dataclasses
  - Import from `models.py`
  - Added alias `LinkedInPerson = LinkedInProfile` for compatibility
  - Updated usages to use `full_name` instead of `name`
  - Added legacy cache key migration for `name` → `full_name`

### Impact Achieved:
- Single data model for profiles (`LinkedInProfile` in `models.py`)
- Single data model for organizations (`LinkedInOrganization` in `models.py`)
- Backward compatibility via aliases
- ~100 lines of duplicate code removed

---

## Phase 6: Publishing Service Consolidation ⏭️ SKIPPED
**Goal:** Create a single unified publishing service for all entry points.

### Analysis Result:
After reviewing the code paths, consolidation is **not recommended** because:
- `main.py` already delegates to `publisher.publish_due_stories()` (simple one-liner)
- `publish_daemon.py` adds intentional safety checks (human approval, rate limiting delays)
- Both paths properly delegate to `linkedin_publisher.py` for actual publishing
- Only ~30 lines of duplication across 3 files - not worth the risk of refactoring

### Current Architecture (Already Good):
- `main.py` → `LinkedInPublisher.publish_due_stories()` ✅
- `publish_daemon.py` → `LinkedInPublisher.publish_immediately()` ✅
- All actual publishing logic in `linkedin_publisher.py` ✅

---

## Phase 7: Match Scoring Consolidation ✅ COMPLETED
**Goal:** Centralize all LinkedIn profile match scoring in `profile_matcher.py`.

### Completed Changes:
- [x] 7.1 Added `score_person_candidate()` function to `profile_matcher.py`
- [x] 7.2 Added `ScoredCandidate` dataclass for detailed scoring results
- [x] 7.3 Added `COMPANY_SKIP_WORDS` constant for reuse
- [x] 7.4 Updated `linkedin_voyager_client.py` to use `score_person_candidate()`
- [x] 7.5 `linkedin_rapidapi_client.py` already uses `calculate_match_score()`

### Impact Achieved:
- Consistent multi-signal scoring across all LinkedIn lookup methods
- Single place to tune matching algorithm weights
- ~50 lines of duplicate scoring logic removed from voyager_client

---

## Phase 8: Rate Limiting Consolidation ✅ COMPLETED
**Goal:** Ensure all rate limiting uses `api_client` centrally.

### Completed Changes:
- [x] 8.1 Added `browser_limiter` to `api_client` for browser-based searches
- [x] 8.2 Added `browser_wait()`, `browser_on_success()`, `browser_on_captcha()` methods
- [x] 8.3 Updated `linkedin_profile_lookup.py` to use centralized browser rate limiting
- [x] 8.4 Kept specialized CAPTCHA cooldown and progressive slowdown logic

### Impact Achieved:
- Browser search timing now uses centralized `api_client.browser_limiter`
- Adaptive rate adjustment on CAPTCHA detection
- Better visibility into browser search metrics

---

## Phase 9: Dependency Injection Integration ✅ COMPLETED
**Goal:** Register all singletons with the DI container for better lifecycle management.

### Completed Changes:
- [x] 9.1 Created `CacheServicesProvider` in `di.py`
- [x] 9.2 Registered `LinkedInCache` with DI container (`linkedin_cache`)
- [x] 9.3 Registered `api_client` with DI container (`api_client`)
- [x] 9.4 Created `DatabaseServicesProvider` for database registration
- [x] 9.5 Added TYPE_CHECKING imports for type hints

### Impact Achieved:
- All major singletons now registerable via DI container
- Better testing through dependency injection
- Cleaner lifecycle management for future use

---

## Phase 10: Database Abstraction Cleanup ✅ COMPLETED
**Goal:** Remove direct SQL queries from `main.py` and add proper Database methods.

### Completed Changes:
- [x] 10.1 Added `Database.get_stories_needing_promotion(require_image)` method
- [x] 10.2 Added `Database.get_stories_needing_profile_lookup()` method
- [x] 10.3 Added `Database.get_stories_needing_urn_extraction()` method
- [x] 10.4 Added `Database.count_people_needing_urns()` method
- [x] 10.5 Added `Database.get_stories_needing_indirect_people()` method
- [x] 10.6 Added `Database.update_story_indirect_people()` method
- [x] 10.7 Added `Database.update_story_promotion()` method
- [x] 10.8 Added `Database.mark_stories_enriched()` method
- [x] 10.9 Added `Database.get_recent_stories()` method
- [x] 10.10 Updated `main.py` to use new Database methods
- [x] 10.11 Removed all direct cursor access from `main.py` helper functions

### Impact Achieved:
- Clean separation of concerns - database logic in Database class
- 8 new methods for common query patterns
- Removed ~100 lines of inline SQL from main.py
- Easier to optimize and test database queries

---

## Phase 11: .env Settings Validation (COMPLETED)
**Goal:** Ensure all .env settings are properly documented and honored.

### Changes Completed:
- [x] 11.1 Added `SEARCH_ENGINE` to `.env.example` with documentation
- [x] 11.2 Added `MAX_PEOPLE_PER_STORY` to `.env.example` with documentation
- [x] 11.3 Verified all Config class settings have corresponding Pydantic validation

### Settings Verified:
- `SEARCH_ENGINE` - Used in `linkedin_profile_lookup.py` for browser search engine selection
- `MAX_PEOPLE_PER_STORY` - Used in `company_mention_enricher.py` to limit API calls
- `RAPIDAPI_KEY` - Used across all LinkedIn lookup modules
- `SKIP_LINKEDIN_DIRECT_SEARCH` - Properly honored in config

---

## Phase 12: Config Synchronization & Missing Settings ✅ COMPLETED
**Goal:** Add missing .env settings to Config class and fix direct os.getenv() calls.

### Changes Completed:
- [x] 12.1 Added `OPENAI_API_KEY` to Pydantic SettingsModel and Config class
- [x] 12.2 Replaced direct os.getenv() calls with Config class:
  - `main.py`: LINKEDIN_LI_AT, LINKEDIN_JSESSIONID, RAPIDAPI_KEY now use Config
  - `linkedin_voyager_client.py`: LINKEDIN_LI_AT, LINKEDIN_JSESSIONID now use Config
- [x] 12.3 Added LinkedIn engagement limits to Config:
  - `LINKEDIN_DAILY_COMMENT_LIMIT` (default: 25)
  - `LINKEDIN_HOURLY_COMMENT_LIMIT` (default: 5)
  - `LINKEDIN_MIN_COMMENT_INTERVAL` (default: 300 seconds)
- [x] 12.4 Added LinkedIn networking limits to Config:
  - `LINKEDIN_WEEKLY_CONNECTION_LIMIT` (default: 100)
  - `LINKEDIN_DAILY_CONNECTION_LIMIT` (default: 20)
- [x] 12.5 Updated `linkedin_engagement.py` to use Config for limits
- [x] 12.6 Updated `linkedin_networking.py` to use Config for limits

### Remaining (deferred):
- API_TIMEOUT_DEFAULT usage - requires more investigation of all timeout locations
- STORIES_PER_CYCLE and PUBLISH_WINDOW_HOURS - may be deprecated settings

---

## Phase 13: Remove Deprecated Dict Caches (MEDIUM PRIORITY)
**Goal:** Complete migration to LinkedInCache by removing legacy dict-based caches.

### Issues Found:
1. `linkedin_profile_lookup.py` has deprecated class-level dict caches (lines 727-760)
2. `linkedin_voyager_client.py` has dict-based fallback caches (lines 314-316)
3. `company_mention_enricher.py` has ValidationCache duplicating person caching logic

### Changes Required:
- [ ] 13.1 Remove deprecated caches from `linkedin_profile_lookup.py`:
  - `_shared_person_cache`
  - `_shared_found_profiles_by_name`
  - `_shared_company_url_to_name`
  - `_shared_failed_lookups`
  - `_shared_company_cache`
- [ ] 13.2 Remove dict fallback caches from `linkedin_voyager_client.py`:
  - `_person_cache`
  - `_org_cache`
  - `_search_cache`
- [ ] 13.3 Migrate `ValidationCache` in `company_mention_enricher.py` to use `LinkedInCache`
- [ ] 13.4 Update all code paths to use only `LinkedInCache`

---

## Phase 14: Consolidate LinkedIn Match Scoring (MEDIUM PRIORITY) ✅ COMPLETED
**Goal:** Remove duplicate match scoring implementations.

### Analysis:
- **`_score_linkedin_candidate()`** in linkedin_profile_lookup.py was DEAD CODE (never called)
- **`_calculate_contradiction_penalty()`** is correctly used as shared helper
- **`profile_matcher.score_person_candidate()`** is the main scoring function
- **Inline scoring** in `_search_person_playwright()` uses `_calculate_contradiction_penalty()` correctly

### Changes Applied:
- [x] 14.1 Removed dead `_score_linkedin_candidate()` function (~135 lines)
- [x] 14.2 Updated spec.md to reference correct pattern
- [x] 14.3 Future improvement noted: `_search_person_playwright()` inline scoring could
      be refactored to use `profile_matcher.score_person_candidate()` (not blocking)

---

## Phase 15: Consolidate DuckDuckGo Search (LOW PRIORITY)
**Goal:** Unify duplicate DuckDuckGo search implementations.

### Issues Found:
1. `searcher.py` line 840: inline DDGS usage in `_search_local()`
2. `searcher.py` line 1564: reusable `_fetch_duckduckgo_results()` wrapper

### Changes Required:
- [ ] 15.1 Refactor `_search_local()` to use `_fetch_duckduckgo_results()`
- [ ] 15.2 Remove duplicate DDGS initialization code

---

## Phase 16: Extract URL Redirect Resolution (LOW PRIORITY) ⏭️ SKIPPED
**Goal:** Move redirect resolution logic to url_utils.py.

### Analysis:
- Redirect resolution in `searcher.py` is specific to Vertex AI Search URLs
- Browser fallback already exists in shared `browser.py` (`resolve_redirect_with_browser`)
- The logic is tightly coupled to searcher's needs and not reusable elsewhere

### Decision: Not worth extracting - too context-specific

---

## Phase 17: Create Unified AI Response Helper (LOW PRIORITY) - DEFERRED
**Goal:** Consolidate AI response pattern (local LLM → Gemini fallback).

### Analysis:
- Many call sites use `local_client.chat.completions.create()` directly
- Modules affected: verifier.py, searcher.py, originality_checker.py
- `api_client.gemini_generate()` already exists for Gemini rate limiting
- Creating a unified helper would require updating many critical paths

### Decision: Deferred - high effort, low priority, extensive testing needed

---

## Phase 18: Standardize LinkedIn Rate Limiting (LOW PRIORITY)
**Goal:** Ensure all LinkedIn modules use centralized api_client rate limiting.

### Issues Found:
1. `linkedin_engagement.py` creates standalone AdaptiveRateLimiter
2. `linkedin_networking.py` creates standalone AdaptiveRateLimiter
3. Should use `api_client.linkedin_limiter` instead

### Changes Required:
- [ ] 18.1 Remove custom rate limiter from `linkedin_engagement.py`
- [ ] 18.2 Remove custom rate limiter from `linkedin_networking.py`
- [ ] 18.3 Update both modules to use `api_client.linkedin_limiter`

---

## Phase 19: Remove Model Aliases (LOW PRIORITY)
**Goal:** Use consistent model names across all LinkedIn modules.

### Issues Found:
1. `linkedin_voyager_client.py`: `LinkedInPerson = LinkedInProfile` alias
2. `linkedin_rapidapi_client.py`: `LinkedInProfileResult = LinkedInProfile` alias

### Changes Required:
- [ ] 19.1 Replace all `LinkedInPerson` usage with `LinkedInProfile`
- [ ] 19.2 Replace all `LinkedInProfileResult` usage with `LinkedInProfile`
- [ ] 19.3 Remove alias definitions

---

## Phase 20: Centralize LinkedIn Public ID Extraction (LOW PRIORITY)
**Goal:** Move public_id extraction to a shared utility.

### Issues Found:
1. Pattern `/in/` public ID extraction duplicated in 4+ files
2. `linkedin_profile_lookup.py` has `_extract_public_id()` function

### Changes Required:
- [x] 20.1 Move `_extract_public_id()` to `url_utils.py` as `extract_linkedin_public_id()`
- [x] 20.2 Update all files to use the shared utility

---

## Implementation Order

### Quick Wins (Do First) - COMPLETED:
1. ✅ Phase 1: Domain Credibility Consolidation
2. ✅ Phase 4: URL Utilities Consolidation
3. ✅ Phase 11: .env Settings Validation

### High Impact (Do Next) - COMPLETED:
4. ✅ Phase 2: LinkedIn Cache Consolidation
5. ✅ Phase 3: HTTP Session Consolidation
6. ✅ Phase 5: LinkedIn Data Models Unification

### Medium Priority - COMPLETED:
7. ✅ Phase 12: Config Synchronization & Missing Settings
8. Phase 13: Remove Deprecated Dict Caches (complex - deferred)
9. ✅ Phase 14: Consolidate LinkedIn Match Scoring (dead code removed)

### Lower Priority:
10. ✅ Phase 15: Consolidate DuckDuckGo Search
11. ⏭️ Phase 16: Extract URL Redirect Resolution (SKIPPED - too context-specific)
12. ⏸️ Phase 17: Create Unified AI Response Helper (DEFERRED - high effort)
13. ⏭️ Phase 18: Standardize LinkedIn Rate Limiting (SKIPPED - custom limits are intentional)
14. ✅ Phase 19: Remove Model Aliases
15. ✅ Phase 20: Centralize LinkedIn Public ID Extraction

### Skipped/Deferred:
- ⏭️ Phase 6: Publishing Service Consolidation (not recommended)
- ⏸️ Phase 13: Remove Deprecated Dict Caches (complex - deferred)
- ⏭️ Phase 16: URL Redirect Resolution (too context-specific to searcher)
- ⏸️ Phase 17: Unified AI Response Helper (high effort, low priority)
- ⏭️ Phase 18: LinkedIn Rate Limiting (engagement/networking need different rates)

---

## Testing Strategy

After each phase:
1. Run `python run_tests.py` to verify no regressions
2. Fix any test failures before proceeding
3. Commit changes with descriptive message
4. Update this document to mark completed items

---

## Metrics for Success

| Metric | Before | After Phases 1-11 | Target |
|--------|--------|-------------------|--------|
| Duplicate domain lists | 2 | 1 ✅ | 1 |
| Dict-based caches | 11 | 6 | 0 |
| HTTP session implementations | 3 | 1 ✅ | 1 |
| Profile data classes | 2 | 1 ✅ | 1 |
| Match scoring implementations | 3 | 2 | 1 |
| Direct os.getenv() bypasses | 6 | 6 | 0 |
| Hardcoded timeouts | 10+ | 10+ | 0 |
| Custom rate limiters | 3 | 3 | 1 |

---

## Notes

- All changes should maintain backward compatibility where possible
- Use deprecation warnings before removing old interfaces
- Document API changes in module docstrings
- Update type hints throughout

