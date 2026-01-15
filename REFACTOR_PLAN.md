# Refactor Plan: Code Consolidation & Optimization

**Created:** January 15, 2026
**Status:** Sprint 2 In Progress (Sprint 1 Complete)
**Estimated LOC Reduction:** ~500 lines (60%)

---

## Overview

This document outlines consolidation opportunities to reduce code duplication, merge process flows, streamline code, and minimize duplicate API/search calls.

---

## Phase 1: High Priority (Estimated ~200 LOC reduction)

### 1.1 Unified Rate Limiting

**Status:** [x] COMPLETED (Sprint 2)
**Effort:** Medium
**Files Affected:**
- `rate_limiter.py` (keep as central implementation)
- `linkedin_voyager_client.py` (remove `_rate_limit()` method, lines 236-255)
- `linkedin_rapidapi_client.py` (remove `_rate_limit()` method, lines 160-165)
- `api_client.py` (ensure all clients use this)

**Current Problem:**
- 3 separate rate limiting implementations
- Each LinkedIn client has its own `_rate_limit()` method
- Duplicated timing variables (`_last_request_time`, `_min_request_interval`)

**Solution:**
1. Remove `_rate_limit()` methods from individual clients
2. Inject `AdaptiveRateLimiter` from `api_client.py` through constructor
3. Use endpoint-specific rate limiting via `api_client.linkedin_request()`
4. Remove duplicate `_last_request_time` tracking variables

**Code to Remove:**
```python
# linkedin_voyager_client.py:L236-255
def _rate_limit(self) -> None:
    now = time.time()
    elapsed = now - self._last_request_time
    if elapsed < self._min_request_interval:
        sleep_time = self._min_request_interval - elapsed
        sleep_time += random.uniform(0.5, 1.5)
        time.sleep(sleep_time)
    self._last_request_time = time.time()

# linkedin_rapidapi_client.py:L160-165
def _rate_limit(self):
    elapsed = time.time() - self._last_request_time
    if elapsed < self.MIN_REQUEST_INTERVAL:
        time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
    self._last_request_time = time.time()
```

---

### 1.2 Unified Caching System

**Status:** [ ] DEFERRED (High complexity, risk of regressions)
**Effort:** High
**Files Affected:**
- `cache.py` (keep as central implementation)
- `linkedin_rapidapi_client.py` (use injected cache)
- `linkedin_voyager_client.py` (remove `_person_cache`, `_org_cache` dicts)
- `linkedin_profile_lookup.py` (remove 6 class-level cache dicts)

**Current Problem:**
- 4+ separate caching implementations
- `linkedin_profile_lookup.py` has 6 class-level caches:
  - `_shared_person_cache`
  - `_shared_found_profiles_by_name`
  - `_shared_company_url_to_name`
  - `_shared_failed_lookups`
  - `_shared_company_cache`
  - `_shared_department_cache`
- `linkedin_voyager_client.py` has `_person_cache` and `_org_cache`
- `linkedin_rapidapi_client.py` has its own cache key generation

**Solution:**
1. Use `HybridCache` from `cache.py` as the single caching layer
2. Define namespace prefixes: `"linkedin:person:"`, `"linkedin:org:"`, `"linkedin:company:"`
3. Inject cache through dependency injection (constructor parameter)
4. Remove all class-level cache dictionaries
5. Migrate JSON persistence to SQLite-based `DiskCache`

**Cache Key Standardization:**
```python
# Standardized key format
def get_cache_key(namespace: str, *parts: str) -> str:
    normalized = "|".join(p.lower().strip() for p in parts)
    return f"{namespace}:{hashlib.md5(normalized.encode()).hexdigest()}"
```

---

### 1.3 Unified LinkedIn Data Models

**Status:** [x] COMPLETED (Sprint 1)
**Effort:** Medium
**Files Affected:**
- Create new `models.py`
- `linkedin_voyager_client.py` (remove `LinkedInPerson`, `LinkedInOrganization`)
- `linkedin_rapidapi_client.py` (remove `LinkedInProfileResult`)
- `ner_engine.py` (update `PersonEntity`, `OrganizationEntity` to use base)

**Current Problem:**
- `LinkedInPerson` (linkedin_voyager_client.py:L39-62) - 11 fields
- `LinkedInProfileResult` (linkedin_rapidapi_client.py:L40-101) - 17 fields
- Significant overlap: name, headline, location, profile_url, match_score

**Solution:**
Create unified models in new `models.py`:

```python
# models.py
@dataclass
class LinkedInProfile:
    """Unified LinkedIn profile representation."""
    urn_id: str = ""
    public_id: str = ""
    linkedin_url: str = ""

    first_name: str = ""
    last_name: str = ""
    full_name: str = ""

    headline: str = ""
    job_title: str = ""
    company: str = ""
    company_linkedin_url: str = ""

    location: str = ""
    about: str = ""
    profile_image_url: str = ""

    # Matching metadata
    match_score: float = 0.0
    match_signals: list[str] = field(default_factory=list)
    confidence: str = ""  # "high", "medium", "low"

    # Additional data
    connection_count: int = 0
    company_domain: str = ""
    company_industry: str = ""

@dataclass
class LinkedInOrganization:
    """Unified LinkedIn organization representation."""
    urn_id: str = ""
    public_id: str = ""
    linkedin_url: str = ""

    name: str = ""
    page_type: str = "company"  # "company" or "school"
    industry: str = ""
    description: str = ""
    location: str = ""
    employee_count: str = ""
```

---

## Phase 2: Medium Priority (Estimated ~150 LOC reduction)

### 2.1 Entity Validation Constants Module

**Status:** [x] COMPLETED (Sprint 1)
**Effort:** Low
**Files Affected:**
- Create new `entity_constants.py` OR add to `text_utils.py`
- `ner_engine.py` (move constants out, keep as optional spaCy module)
- `linkedin_profile_lookup.py` (update imports)
- `company_mention_enricher.py` (update imports)

**Current Problem:**
- Constants defined in `ner_engine.py` (requires optional spaCy)
- Both `linkedin_profile_lookup.py` and `company_mention_enricher.py` have identical fallback patterns:

```python
try:
    from ner_engine import (
        INVALID_ORG_NAMES,
        INVALID_ORG_PATTERNS,
        INVALID_PERSON_NAMES,
        VALID_SINGLE_WORD_ORGS,
    )
except ImportError:
    INVALID_ORG_NAMES = set()
    INVALID_ORG_PATTERNS = []
    INVALID_PERSON_NAMES = set()
    VALID_SINGLE_WORD_ORGS = set()
```

**Solution:**
1. Move constants to `entity_constants.py` (no dependencies)
2. Have `ner_engine.py` import from `entity_constants.py`
3. All other files import directly from `entity_constants.py`
4. Remove fallback patterns from consumer files

---

### 2.2 Spam Pattern Consolidation

**Status:** [x] COMPLETED (Sprint 2)
**Effort:** Low
**Files Affected:**
- Create new `content_validation.py`
- `linkedin_optimizer.py` (remove `SPAM_PATTERNS`, import from new module)
- `linkedin_engagement.py` (remove `SPAM_PATTERNS`, import from new module)

**Current Problem:**
- `linkedin_optimizer.py:L80-89` has 9 spam patterns for post content
- `linkedin_engagement.py:L161-182` has 22 spam patterns for engagement
- Some patterns overlap, some are unique

**Solution:**
Create unified `content_validation.py`:

```python
# content_validation.py

# Patterns for post content optimization
POST_SPAM_PATTERNS = [
    r"(?i)like if you agree",
    r"(?i)share this post",
    r"(?i)comment \d+ for",
    r"(?i)drop an emoji",
    r"(?i)follow me for",
    r"(?i)link in (bio|comments)",
    r"(?i)dm me for",
    r"(?i)🔥{3,}",
    r"(?i)👇{3,}",
]

# Patterns for engagement/comment filtering
COMMENT_SPAM_PATTERNS = [
    r"check out my",
    r"visit my profile",
    r"click (the )?link",
    r"buy now",
    r"limited offer",
    r"free.*download",
    r"guaranteed results",
    r"make money",
    r"dm me",
    r"inbox me",
    r"check inbox",
    r"promo code",
    r"discount",
    r"subscribe to",
    r"follow me",
    r"like my",
]

# Combined patterns for general spam detection
ALL_SPAM_PATTERNS = list(set(POST_SPAM_PATTERNS + COMMENT_SPAM_PATTERNS))
```

---

### 2.3 Organization Aliases Extraction

**Status:** [x] COMPLETED (Sprint 2)
**Effort:** Low
**Files Affected:**
- Create new `organization_aliases.py`
- `linkedin_profile_lookup.py` (remove `_ORG_ALIASES`, import from new module)
- `ner_engine.py` (import shared aliases)

**Current Problem:**
- `linkedin_profile_lookup.py:L449-536` has 80+ organization aliases
- Similar mappings exist in `ner_engine.py` for entity resolution
- No single source of truth

**Solution:**
Create `organization_aliases.py` with:
- University abbreviations (MIT, UCLA, CMU, etc.)
- Company aliases (AWS, MSFT, Meta/Facebook, etc.)
- Research institution mappings

---

### 2.4 LinkedIn Search Consolidation

**Status:** [ ] DEFERRED (HybridLinkedInLookup already serves as facade)
**Effort:** High
**Files Affected:**
- `linkedin_voyager_client.py` (update `HybridLinkedInLookup`)
- `linkedin_rapidapi_client.py` (integrate into unified service)
- `linkedin_profile_lookup.py` (integrate into unified service)

**Current Problem:**
- 4+ ways to search for LinkedIn profiles:
  - `FreshLinkedInAPIClient.search_person()` (RapidAPI)
  - `LinkedInVoyagerClient.find_person()` (Voyager API)
  - `HybridLinkedInLookup.find_person()` (Multi-source)
  - `LinkedInCompanyLookup.search_person()` (Browser-based)
- 3 ways to search for companies:
  - `LinkedInVoyagerClient.find_company()`
  - `HybridLinkedInLookup.find_company()`
  - `LinkedInCompanyLookup.search_company()`

**Solution:**
Enhance `HybridLinkedInLookup` as the single facade:
1. Tries RapidAPI first (fastest, most reliable)
2. Falls back to Voyager API (if not blocked)
3. Falls back to browser-based search (slowest but always works)
4. Returns consistent `LinkedInProfile` objects
5. All other code uses this single entry point

---

## Phase 3: Architecture Improvements (Estimated ~100 LOC reduction)

### 3.1 HTTP Session Factory

**Status:** [x] COMPLETED (Sprint 3)
**Effort:** Medium
**Files Affected:**
- `api_client.py` (add session factory)
- `linkedin_voyager_client.py` (use shared session)
- `linkedin_rapidapi_client.py` (use shared session)
- `linkedin_profile_lookup.py` (use shared session)

**Current Problem:**
- 4 different HTTP session patterns:
  - `linkedin_voyager_client.py:L201` - `requests.Session()`
  - `linkedin_rapidapi_client.py:L146` - `requests.Session()`
  - `linkedin_profile_lookup.py:L1980` - `httpx.Client()` (different library!)
  - `api_client.py` - uses raw `requests` module

**Solution:**
1. Standardize on `requests.Session()` with retry adapters
2. Add session factory to `api_client.py`:

```python
def get_session(self, name: str = "default") -> requests.Session:
    """Get or create a named session with retry logic."""
    if name not in self._sessions:
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        self._sessions[name] = session
    return self._sessions[name]
```

---

### 3.2 Browser Module (UC Chrome)

**Status:** [x] COMPLETED (Sprint 4)
**Effort:** Medium
**Files Affected:**
- Create new `browser.py`
- `linkedin_profile_lookup.py` (use shared browser module)
- `find_indirect_people.py` (use shared browser module)
- `searcher.py` (use shared browser module)

**Current Problem:**
- Duplicate UC Chrome setup in 3 files:
  - `linkedin_profile_lookup.py:L43-75` - UC import + Windows error patch
  - `find_indirect_people.py:L13-28` - Duplicate UC import pattern
  - `searcher.py:L814-825` - Inline driver creation

**Solution:**
Create `browser.py`:

```python
# browser.py
"""Unified browser automation module using undetected-chromedriver."""

UC_AVAILABLE = False
uc = None

try:
    import undetected_chromedriver as _uc
    uc = _uc
    UC_AVAILABLE = True
except ImportError:
    pass

class BrowserPool:
    """Manages shared browser instances."""
    _instance = None
    _driver = None
    _search_count = 0
    _max_searches_per_driver = 8

    @classmethod
    def get_driver(cls, headless: bool = True) -> Any:
        """Get or create a UC Chrome driver."""
        ...

    @classmethod
    def release_driver(cls) -> None:
        """Release and close the current driver."""
        ...
```

---

### 3.3 Match Scoring Consolidation

**Status:** [x] PARTIALLY COMPLETED (Sprint 4)
**Effort:** Medium
**Files Affected:**
- `profile_matcher.py` (added `calculate_match_score()` helper function)
- `linkedin_rapidapi_client.py` (now uses centralized scoring)
- `linkedin_voyager_client.py` (kept custom scoring - works with LinkedInPerson objects)

**Notes:**
- Added `calculate_match_score()` to profile_matcher.py for simple name/company matching
- linkedin_rapidapi_client.py now imports and uses centralized function
- linkedin_voyager_client.py kept its own `_find_best_person_match()` because it:
  - Works with LinkedInPerson dataclass objects (not dicts)
  - Uses a different scoring scale (0-10+ vs 0-1)
  - Attaches signals directly to candidate objects
  - Handles title/location matching more granularly

**Current Problem:**
- 3 match scoring implementations:
  - `linkedin_rapidapi_client.py:L222` - `_calculate_match_score()`
  - `linkedin_voyager_client.py:L720-780` - `_score_person_match()`
  - `profile_matcher.py` - Full matching engine

**Solution:**
1. Keep `profile_matcher.py` as the single scoring implementation
2. Remove scoring methods from individual clients
3. Import and use `ProfileMatcher` in all LinkedIn clients

---

### 3.4 Similarity Function Consolidation

**Status:** [x] COMPLETED (Sprint 3)
**Effort:** Low
**Files Affected:**
- `text_utils.py` (add shared implementation)
- `searcher.py` (remove `calculate_similarity()`)
- `property_tests.py` (remove `calculate_similarity()`)

**Current Problem:**
- Duplicate `calculate_similarity()` functions:
  - `searcher.py:L47` - Jaccard similarity with stopwords
  - `property_tests.py:L69` - Simpler word-based similarity

**Solution:**
Add to `text_utils.py`:

```python
def calculate_similarity(text1: str, text2: str, remove_stopwords: bool = True) -> float:
    """Calculate Jaccard similarity between two texts."""
    ...
```

---

## Phase 4: New Features

### 4.1 HUMAN_IN_IMAGE Configuration Option

**Status:** [x] COMPLETED (Sprint 1)
**Effort:** Low
**Files Affected:**
- `.env.example` (add new option)
- `config.py` (add new config variable)
- `image_generator.py` (implement conditional logic)

**Description:**
Add a `.env` option `HUMAN_IN_IMAGE` to control whether generated images include a central human character.

**Configuration:**
```env
# .env
# Control human presence in generated images
# YES = Include central human character (current behavior)
# NO = No central character; random humans only if incidental/peripheral
HUMAN_IN_IMAGE=YES
```

**Implementation:**

1. **Add to `config.py`:**
```python
HUMAN_IN_IMAGE: bool = os.getenv("HUMAN_IN_IMAGE", "YES").upper() == "YES"
```

2. **Update `image_generator.py`:**
```python
def _build_image_prompt(self, story: Story, style: str) -> str:
    """Build image generation prompt based on story and config."""

    base_prompt = self._extract_visual_elements(story)

    if Config.HUMAN_IN_IMAGE:
        # Current behavior: include central human character
        prompt = f"{base_prompt} with a professional person as the central figure"
    else:
        # No central character - focus on concepts, objects, environments
        prompt = (
            f"{base_prompt}. "
            "Focus on abstract concepts, technology, environments, or objects. "
            "Do not include a central human figure. "
            "If people appear, they should be incidental, peripheral, "
            "or part of the background only."
        )

    return prompt
```

3. **Update image prompt construction** to conditionally exclude human-centric language:
   - Remove "professional person", "expert", "scientist", etc. from prompts
   - Add negative prompts for portraits/close-ups when HUMAN_IN_IMAGE=NO
   - Focus on conceptual/abstract representations of the story topic

---

## Implementation Order

### Sprint 1 (Week 1-2) ✅ COMPLETED

- [x] 1.3 Unified LinkedIn Data Models (`models.py`)
- [x] 4.1 HUMAN_IN_IMAGE Configuration Option
- [x] 2.1 Entity Validation Constants Module

### Sprint 2 (Week 3-4) ✅ COMPLETED

- [x] 1.1 Unified Rate Limiting
- [x] 2.2 Spam Pattern Consolidation
- [x] 2.3 Organization Aliases Extraction

### Sprint 3 (Week 5-6) ✅ COMPLETED
- [x] 1.2 Unified Caching System (DEFERRED - High complexity)
- [x] 3.1 HTTP Session Factory
- [x] 3.4 Similarity Function Consolidation

### Sprint 4 (Week 7-8) 🔄 IN PROGRESS
- [ ] 2.4 LinkedIn Search Consolidation (DEFERRED - already has HybridLinkedInLookup facade)
- [x] 3.2 Browser Module (UC Chrome)
- [ ] 3.3 Match Scoring Consolidation

---

## Testing Strategy

For each refactoring task:
1. Write tests for current behavior before changes
2. Make changes incrementally
3. Verify tests still pass after each change
4. Update integration tests as needed

---

## Rollback Plan

Each phase should be implemented as a separate branch:
- `refactor/phase1-rate-limiting`
- `refactor/phase1-caching`
- `refactor/phase1-models`
- etc.

Merge to main only after full testing. Keep old code commented (not deleted) for first week in production.

---

## Notes

- Priority order based on: frequency of duplication × impact × ease of implementation
- Some changes may require updating tests in `test_phase3.py`, `test_phase4.py`
- Consider adding deprecation warnings before removing public APIs
- Document all changes in README.md changelog section
