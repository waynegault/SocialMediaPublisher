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

## Phase 3: HTTP Session Consolidation (MEDIUM PRIORITY)
**Goal:** Ensure all HTTP requests use the centralized `api_client` for rate limiting and retry logic.

### Current Issues:
- `linkedin_profile_lookup.py` uses `httpx.Client` directly (bypasses rate limiting)
- `searcher.py` uses raw `requests.head()` for Wayback archiving (bypasses rate limiting)
- Inconsistent session management across modules

### Changes:
- [ ] 3.1 Update `searcher.py` `_archive_url()` to use `api_client.http_request()`
- [ ] 3.2 Update `linkedin_profile_lookup.py` HTTP client to use `api_client`
- [ ] 3.3 Add `http_head()` method to `RateLimitedAPIClient` for HEAD requests
- [ ] 3.4 Audit all `import requests` and ensure centralized client usage

### Estimated Impact:
- All HTTP calls rate-limited
- Consistent retry logic
- Better quota management

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

## Phase 5: LinkedIn Data Models Unification (MEDIUM PRIORITY)
**Goal:** Create unified data classes for LinkedIn profile/company data.

### Current Issues:
- `linkedin_rapidapi_client.py`: `LinkedInProfileResult` dataclass
- `linkedin_voyager_client.py`: `LinkedInPerson`, `LinkedInOrganization` dataclasses
- Different field sets and naming conventions

### Changes:
- [ ] 5.1 Create `linkedin_models.py` with unified dataclasses
- [ ] 5.2 Define `LinkedInProfile` with superset of all fields
- [ ] 5.3 Define `LinkedInCompany` with superset of all fields
- [ ] 5.4 Add `from_rapidapi()` and `from_voyager()` factory methods
- [ ] 5.5 Update `linkedin_rapidapi_client.py` to use unified models
- [ ] 5.6 Update `linkedin_voyager_client.py` to use unified models
- [ ] 5.7 Update `HybridLinkedInLookup` to return unified models

### Estimated Impact:
- Single data model for profiles
- Easier to compare/merge results from different sources
- Type safety improvements

---

## Phase 6: Publishing Service Consolidation (MEDIUM PRIORITY)
**Goal:** Create a single unified publishing service for all entry points.

### Current Issues:
- `main.py` has `_run_publish()` calling `publish_due_stories()`
- `publish_daemon.py` has its own publishing loop
- `linkedin_publisher.py` has `publish_due_stories()` method
- Three code paths for the same operation

### Changes:
- [ ] 6.1 Create `publishing_service.py` with `PublishingService` class
- [ ] 6.2 Implement unified `publish_due_stories()` with validation options
- [ ] 6.3 Update `main.py` to use `PublishingService`
- [ ] 6.4 Update `publish_daemon.py` to use `PublishingService`
- [ ] 6.5 Deprecate direct `LinkedInPublisher.publish_due_stories()`

### Estimated Impact:
- Single source of truth for publishing logic
- Consistent validation across all entry points
- Easier to maintain and test

---

## Phase 7: Match Scoring Consolidation (LOW PRIORITY)
**Goal:** Centralize all LinkedIn profile match scoring in `profile_matcher.py`.

### Current Issues:
- `linkedin_profile_lookup.py`: `_score_linkedin_result()`, `_verify_linkedin_match()`
- `linkedin_rapidapi_client.py`: `_calculate_match_score()`
- `linkedin_voyager_client.py`: delegates to `profile_matcher`
- Inconsistent scoring algorithms

### Changes:
- [ ] 7.1 Extend `profile_matcher.py` with comprehensive scoring functions
- [ ] 7.2 Add `score_profile_match(candidate, target_name, target_company, target_title)`
- [ ] 7.3 Update `linkedin_profile_lookup.py` to use `profile_matcher`
- [ ] 7.4 Update `linkedin_rapidapi_client.py` to use `profile_matcher`

### Estimated Impact:
- Consistent scoring across all lookup methods
- Single place to tune matching algorithm
- Better match quality

---

## Phase 8: Rate Limiting Consolidation (LOW PRIORITY)
**Goal:** Ensure all rate limiting uses `api_client` centrally.

### Current Issues:
- `linkedin_profile_lookup.py` has custom `_min_search_interval`, `_consecutive_searches` logic
- Separate rate limiting for browser-based searches
- Inconsistent cooldown handling

### Changes:
- [ ] 8.1 Add `browser` endpoint to `api_client.linkedin_limiter`
- [ ] 8.2 Remove custom rate limiting from `linkedin_profile_lookup.py`
- [ ] 8.3 Use centralized limiter for all browser-based searches
- [ ] 8.4 Consolidate CAPTCHA cooldown handling

### Estimated Impact:
- Unified rate limiting
- Easier to tune and monitor
- Consistent backoff behavior

---

## Phase 9: Dependency Injection Integration (LOW PRIORITY)
**Goal:** Register all singletons with the DI container for better lifecycle management.

### Current Issues:
- `LinkedInCache` uses class-level singleton
- `ValidationCache` uses separate singleton pattern
- `api_client` uses module-level global
- `RAGEngine` creates its own instance

### Changes:
- [ ] 9.1 Create `CacheServicesProvider` module
- [ ] 9.2 Register `LinkedInCache` with DI container
- [ ] 9.3 Register `api_client` with DI container
- [ ] 9.4 Register `ValidationCache` with DI container
- [ ] 9.5 Update modules to resolve dependencies from DI container

### Estimated Impact:
- Better testing through dependency injection
- Cleaner lifecycle management
- Easier to mock for unit tests

---

## Phase 10: Database Abstraction Cleanup (LOW PRIORITY)
**Goal:** Remove direct SQL queries from `main.py` and add proper Database methods.

### Current Issues:
- `main.py` has direct cursor queries (lines 283-296, 798-801, 844-854)
- Bypasses Database abstraction layer
- Harder to maintain

### Changes:
- [ ] 10.1 Add `Database.get_stories_needing_promotion()` method
- [ ] 10.2 Add `Database.get_stories_needing_urn_extraction()` method
- [ ] 10.3 Update `main.py` to use new Database methods
- [ ] 10.4 Remove direct cursor access from `main.py`

### Estimated Impact:
- Cleaner separation of concerns
- Database queries in one place
- Easier to optimize queries

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

## Implementation Order

### Quick Wins (Do First):
1. Phase 1: Domain Credibility Consolidation
2. Phase 4: URL Utilities Consolidation
3. Phase 11: .env Settings Validation ✓

### High Impact (Do Next):
4. Phase 2: LinkedIn Cache Consolidation
5. Phase 3: HTTP Session Consolidation
6. Phase 5: LinkedIn Data Models Unification

### Medium Priority (Do Later):
7. Phase 6: Publishing Service Consolidation
8. Phase 7: Match Scoring Consolidation

### Low Priority (Do When Time Permits):
9. Phase 8: Rate Limiting Consolidation
10. Phase 9: Dependency Injection Integration
11. Phase 10: Database Abstraction Cleanup

---

## Testing Strategy

After each phase:
1. Run `python run_tests.py` to verify no regressions
2. Fix any test failures before proceeding
3. Commit changes with descriptive message
4. Update this document to mark completed items

---

## Metrics for Success

| Metric | Before | Target |
|--------|--------|--------|
| Duplicate domain lists | 2 | 1 |
| Dict-based caches | 11 | 3 |
| HTTP session implementations | 3 | 1 |
| Profile data classes | 2 | 1 |
| Publishing code paths | 3 | 1 |
| Match scoring implementations | 3 | 1 |

---

## Notes

- All changes should maintain backward compatibility where possible
- Use deprecation warnings before removing old interfaces
- Document API changes in module docstrings
- Update type hints throughout

