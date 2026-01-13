# Named Entity & People Enrichment Workflow (v2 – Clean-Slate)

## Core Principles

1. **Extraction ≠ Validation ≠ Enrichment** — Each step has distinct inputs, outputs, and failure modes
2. **People are records, not guesses** — Every person record requires authoritative evidence
3. **Direct ≠ Indirect is a first-class distinction** — Explicitly named people vs. discovered leadership
4. **Everything is explainable and auditable** — Full provenance trail for all data
5. **Iterate incrementally** — Implement MVP first, validate, then add advanced features
6. **Measure before changing** — Baseline metrics are a HARD PREREQUISITE before any code changes

---

## ⛔ CRITICAL: Phase Gates

> **DO NOT proceed to the next phase until the current phase is stable in production.**

| Gate | Prerequisite | Success Criteria | STOP If... |
|------|--------------|------------------|------------|
| **GATE 0** | Baseline metrics collected | 1 week of data, metrics dashboard working | Cannot measure current state |
| **GATE 1** | Phase 1 MVP complete | Match rate improved OR maintained, error rate <5% | Match rate drops >10% |
| **GATE 2** | Phase 2 stable | 2 weeks in production, no regressions | False positive rate increases |
| **GATE 3** | Phase 3 stable | Full cutover complete | Any critical failures |

**Rollback Triggers (any of these = immediate rollback):**
- LinkedIn match rate drops >10% from baseline
- Error rate exceeds 5% of stories
- False positive rate exceeds 2x baseline
- Processing time >10x baseline

---

## Current State & Baseline Metrics

> **⛔ HARD PREREQUISITE:** Baseline metrics MUST be collected for at least 1 week before
> ANY code changes are made. This is non-negotiable. You cannot improve what you cannot measure.

### ✅ Pre-Implementation Checklist

- [ ] Baseline logging code deployed (see instrumentation code below)
- [ ] 1 week of metrics collected
- [ ] Dashboard or log analysis showing current state
- [ ] Baseline numbers filled in the table below
- [ ] Database reset decision confirmed with stakeholder

### Metrics to Instrument (Before Implementation)

Add logging to current code to capture these baseline metrics:

| Metric | How to Measure | Target Improvement |
|--------|----------------|--------------------|
| LinkedIn match rate | % of story_people with valid linkedin_profile | Current: ~TBD% → Target: 70%+ |
| High-confidence match rate | % with match_confidence = "high" | Current: ~TBD% → Target: 50%+ |
| Zero-match stories | % of stories with no LinkedIn profiles found | Current: ~TBD% → Target: <20% |
| Paywall block rate | % of source URLs that fail to fetch | Current: ~TBD% → Target: <10% |
| False positive rate | Manual sample: wrong person matched | Current: ~TBD% → Target: <5% |

### Known Failure Patterns to Address

| Pattern | Current Behavior | This Spec's Fix |
|---------|------------------|----------------|
| Paywall blocking | Pipeline fails, no enrichment | Article fallback chain (Step 1.1) |
| Common names | Wrong "John Smith" matched | Enhanced exclusion criteria (Step 3.4) |
| No LinkedIn profile | Person dropped from mentions | Org fallback (Step 3.7) |
| Forced bad matches | Low-confidence profiles used | Strict thresholds + fallback |
| No audit trail | Can't debug why match failed | enrichment_log field |

### Instrumentation Code (Add to current pipeline)

```python
# Add to company_mention_enricher.py or main.py:
import logging
logger = logging.getLogger(__name__)

def log_enrichment_metrics(story: Story, results: dict) -> None:
    """Log metrics for baseline measurement."""
    total_people = len(story.story_people) + len(story.org_leaders)
    with_linkedin = sum(
        1 for p in story.story_people + story.org_leaders
        if p.get("linkedin_profile")
    )

    logger.info(f"METRIC:enrichment_complete story_id={story.id} "
                f"total_people={total_people} "
                f"with_linkedin={with_linkedin} "
                f"match_rate={with_linkedin/max(total_people,1):.2%}")
```

---

## High-Level Pipeline

```text
Story Search → Phase 1: Entity Extraction (during search) → Phase 2: Validation & Discovery
           → Phase 3: LinkedIn Matching → Phase 4: Persistence
```

> **Critical Design Principle:** Entity extraction happens during story search/production,
> using the same article content read. Source material is NEVER read twice.

**Phase Summary:**

| Phase | Name | Input | Output |
|-------|------|-------|--------|
| 0 | Story Search | Search query | Story + extracted entities (single read) |
| 1 | Entity Extraction | Article content (from search) | Direct people, organizations |
| 2 | Validation & Discovery | Direct people, orgs | Validated direct + indirect |
| 3 | LinkedIn Matching | All validated people | LinkedIn URLs + confidence |
| 4 | Persistence | Matched people | Updated Story record |

---

## Overview

This specification defines the workflow for extracting people and organizations from story sources, validating their credentials, matching them to LinkedIn profiles, and storing structured data in the Social Media Publisher database.

### Integration with Existing Codebase

This workflow integrates with the existing architecture:

| Component | File | Role |
|-----------|------|------|
| `ContentEngine` | [main.py](main.py) | Main orchestrator, calls enrichment pipeline |
| `CompanyMentionEnricher` | [company_mention_enricher.py](company_mention_enricher.py) | Current enrichment coordinator (to be refactored) |
| `NEREngine` | [ner_engine.py](ner_engine.py) | spaCy-based NER with domain patterns |
| `LinkedInCompanyLookup` | [linkedin_profile_lookup.py](linkedin_profile_lookup.py) | LinkedIn search via UC Chrome + Gemini |
| `ProfileMatcher` | [profile_matcher.py](profile_matcher.py) | Multi-signal scoring for profile matching |
| `Database` | [database.py](database.py) | SQLite storage with `Story` dataclass |
| `api_client` | [api_client.py](api_client.py) | Rate-limited API wrapper |
| `text_utils` | [text_utils.py](text_utils.py) | Name normalization, context keywords |

### Database Fields (from `Story` dataclass)

```python
# Current fields in database.py Story class:
story_people: list[dict]     # Direct people mentioned in story
org_leaders: list[dict]      # Indirect people (leadership of orgs)
organizations: list[str]     # Organization names
enrichment_status: str       # pending, enriched, skipped, error
```

### New Fields to Add

```python
# Proposed changes to Story dataclass:
# NOTE: Database will be reset. Use clean, descriptive field names.

# Rename existing fields for clarity:
direct_people: list[dict]    # People explicitly named in story (was: story_people)
indirect_people: list[dict]  # Leadership discovered from orgs (was: org_leaders)

# New metadata fields:
enrichment_log: dict         # Processing metrics, errors, and audit trail
enrichment_quality: str      # "high", "medium", "low", "failed"
```

---

## STEP 0: ORCHESTRATION ENTRY POINT

> **Goal:** Define where and how the enrichment pipeline is invoked.

### Current Entry Points (to be refactored)

The enrichment logic is currently fragmented across multiple functions in [main.py](main.py):

```python
# Current flow in _test_search():
_run_org_leaders_enrichment_silent(engine)  # Step 2 - Find org leaders
_run_profile_lookup_silent(engine)          # Step 3 - LinkedIn profiles
_run_urn_extraction_silent(engine)          # Step 4 - Extract URNs
_mark_stories_enriched(engine)              # Step 6 - Update status
```

### Proposed: Unified Pipeline Orchestrator

Create a new `EntityEnrichmentPipeline` class or refactor `CompanyMentionEnricher`:

```python
# New orchestrator in company_mention_enricher.py:
class EntityEnrichmentPipeline:
    """Unified pipeline for entity extraction and LinkedIn matching."""

    def __init__(self, db: Database, linkedin_lookup: LinkedInCompanyLookup):
        self.db = db
        self.linkedin_lookup = linkedin_lookup
        self.profile_matcher = ProfileMatcher()

    def enrich_story(self, story: Story) -> EnrichmentResult:
        """
        Run enrichment pipeline for stories with entities already extracted.

        NOTE: Entity extraction happens during story search (Phase 0).
        This method handles Phase 2-4 only (validation, matching, persistence).
        Story.direct_people and Story.organizations are already populated.
        """
        # Phase 2: Validate and discover (entities already extracted)
        validated_direct = self._validate_people(story.direct_people)
        indirect_people = self._discover_org_leaders(story.organizations)

        # Phase 3: LinkedIn matching
        all_people = validated_direct + indirect_people
        matched_people = self._match_linkedin_profiles(all_people)

        # Phase 4: Persist
        return self._persist_results(story, matched_people)

    def enrich_pending_stories(self) -> tuple[int, int]:
        """Batch process stories that have entities but need LinkedIn matching."""
        stories = self.db.get_stories_by_status("pending")
        enriched, skipped = 0, 0
        for story in stories:
            try:
                self.enrich_story(story)
                enriched += 1
            except EnrichmentError as e:
                logger.warning(f"Skipped {story.id}: {e}")
                skipped += 1
        return enriched, skipped
```

### CLI Integration

Update `_test_search()` in [main.py](main.py) to use the unified pipeline:

```python
# Replace fragmented calls with:
pipeline = EntityEnrichmentPipeline(engine.db, engine.linkedin_lookup)
enriched, skipped = pipeline.enrich_pending_stories()
print(f"Enriched {enriched} stories, skipped {skipped}")
```

### Incremental Rollout Strategy

> **Critical:** Don't do a big-bang replacement. Run new and old pipelines in parallel.

**Phase A: Shadow Mode (1-2 weeks)**
```python
# In main.py - run both pipelines, compare results
USE_NEW_PIPELINE = os.environ.get("USE_NEW_ENRICHMENT", "shadow")

if USE_NEW_PIPELINE == "shadow":
    # Run old pipeline (writes to DB)
    _run_org_leaders_enrichment_silent(engine)
    _run_profile_lookup_silent(engine)

    # Run new pipeline (logs only, doesn't write)
    new_pipeline = EntityEnrichmentPipeline(engine.db, engine.linkedin_lookup)
    new_results = new_pipeline.enrich_pending_stories(dry_run=True)

    # Compare and log differences
    compare_enrichment_results(old_results, new_results)

elif USE_NEW_PIPELINE == "true":
    # New pipeline only
    pipeline = EntityEnrichmentPipeline(engine.db, engine.linkedin_lookup)
    enriched, skipped = pipeline.enrich_pending_stories()
```

**Phase B: A/B Testing (2-4 weeks)**
- Route 20% of stories through new pipeline
- Compare LinkedIn match rates between cohorts
- Monitor for regressions (false positives, errors)

**Phase C: Full Cutover**
- Switch to new pipeline as default
- Remove old fragmented functions
- Keep feature flag for emergency rollback

**Rollback Criteria:**
- LinkedIn match rate drops >10% from baseline
- Error rate exceeds 5%
- False positive rate exceeds baseline by >2x

---

## PHASE 1: ENTITY EXTRACTION (INTEGRATED WITH STORY SEARCH)

> **Output:** StoryContext with extracted entities from the SAME article read

> **Critical:** Entity extraction happens during `StorySearcher.search_and_process()`,
> NOT as a separate step. The article content fetched for story discovery is immediately
> used for entity extraction. **Source material is read exactly once.**

### Step 1.1: Integrated Story + Entity Extraction
**Goal:** Extract entities during story search using the same article content

**Implementation:** Extend `StorySearcher._process_stories_data()` in [searcher.py](searcher.py)

The existing story search already fetches article content (via Gemini grounding or UC Chrome).
Entity extraction is added to this same process:

```python
# In StorySearcher._process_stories_data():
def _process_stories_data(self, stories_data: list[dict], since_date: datetime) -> int:
    """Process stories AND extract entities from the same content."""
    for story_data in stories_data:
        # Story content is already available from search
        article_content = story_data.get("full_text") or story_data.get("summary", "")
        title = story_data.get("title", "")

        # Extract entities from the SAME content used for story production
        entities = self._extract_entities_inline(title, article_content)

        # Include entities in the story record
        story_data["direct_people"] = entities["people"]
        story_data["organizations"] = entities["organizations"]

        # Save story with entities already populated
        self._save_story(story_data)

def _extract_entities_inline(self, title: str, content: str) -> dict:
    """Extract entities during story processing (no additional fetch)."""
    full_text = f"{title}\n\n{content}"

    # Use NEREngine with the already-fetched content
    entities = self.ner_engine.extract_entities(full_text)

    return {
        "people": [p for p in entities if isinstance(p, PersonEntity)],
        "organizations": [o for o in entities if isinstance(o, OrganizationEntity)],
    }
```

**Why This Matters:**
- Article content is fetched ONCE during story search
- Entity extraction uses the SAME content (no redundant HTTP requests)
- Reduces latency, API calls, and risk of content changing between reads
- Story is saved with entities already populated

**Fallback:** If story was created without inline extraction (legacy path),
use `story.summary + story.title` for extraction (already in database).

---

### Step 1.2: Extract Named Entities with Context
**Goal:** Identify all people mentioned with their attributes

**Implementation:** Enhance `NEREngine.extract_entities()` in [ner_engine.py](ner_engine.py)

> **Note:** This extraction uses article content already available from story search.
> NO additional HTTP requests or article fetching occurs.

**AI Extraction Prompt Structure:**
Use `api_client.gemini_generate()` for entity extraction (content already in memory):

```text
Extract ALL people mentioned in this article. For each person provide:
- Full name (including titles: Dr., Prof., etc.)
- Job title/position (exact as written)
- Employer organization (company, university, institution)
- Location (city, state, country if mentioned)
- Specialty/field (research area, department)
- Role in story (primary subject, quoted source, mentioned, author)
- Direct mention: YES if explicitly named, NO if inferred
```

**Entity Normalization:**
Use existing utilities from [text_utils.py](text_utils.py):
- `strip_titles()` - Remove title prefixes (Dr., Prof.)
- `extract_first_last_name()` - Parse name components
- `get_name_variants()` - Generate name variations
- `is_nickname_of()` - Check nickname mappings via `NICKNAME_MAP`
- `is_common_name()` - Flag names needing extra validation

**Classification:**
- **DIRECT**: Person explicitly named in article text → stored in `story.direct_people`
- **INDIRECT**: To be discovered (leadership of orgs) → stored in `story.indirect_people`

**Output:** Uses existing `PersonEntity` dataclass from [ner_engine.py](ner_engine.py):

```python
@dataclass
class PersonEntity(Entity):
    """A person entity with additional fields."""
    text: str           # Original name as found
    title: str = ""     # Job title if found
    affiliation: str = ""  # Company/organization
    linkedin_profile: str = ""
    linkedin_urn: str = ""
    # New fields to add:
    location: str = ""
    specialty: str = ""
    role_in_story: str = ""  # primary_subject, quoted, mentioned, author
    is_direct: bool = True
    confidence: float = 0.95
    validation_source: str = ""
```

---

### Step 1.3: Extract and Normalize Organizations
**Goal:** Build canonical list of all organizations mentioned

**Implementation:** Enhance `NEREngine` with `OrganizationEntity` from [ner_engine.py](ner_engine.py)

**Extraction:**
Uses existing dictionaries in `ner_engine.py`:
- `KNOWN_COMPANIES` - Maps abbreviations to full names (BASF, Shell, etc.)
- `KNOWN_UNIVERSITIES` - Academic institution mappings (MIT, Stanford, etc.)
- `COMPANY_SUFFIXES` - Company type indicators (Ltd, Inc, Corp, etc.)
- `ACADEMIC_PATTERNS` - University keywords (university, college, institute)

**Normalization Process:**
Leverage existing `NEREngine._normalize_organization()` and `_classify_org_type()`:

```python
# Existing methods to enhance:
def _normalize_organization(self, text: str) -> str:
    """Normalize org name to canonical form using KNOWN_COMPANIES/UNIVERSITIES."""

def _classify_org_type(self, text: str) -> str:
    """Classify as: university, company, research_institute, government, ngo."""
```

**External Validation:**
Use UC Chrome with Google or DuckDuckGo to verify organization details.

> **Note:** UC Chrome avoids bot detection when searching Google directly.

**Output:** Uses existing `OrganizationEntity` dataclass:

```python
@dataclass
class OrganizationEntity(Entity):
    """An organization/company entity."""
    text: str              # As mentioned in source
    normalized: str        # Canonical name
    org_type: str = ""     # company, university, research_institute, government, ngo
    industry: str = ""
    headquarters: str = ""
    linkedin_page: str = ""
    linkedin_urn: str = ""
```

---

### Step 1.4: Identify Indirect People to Discover
**Goal:** Define which leadership roles to find for each organization

**Implementation:** Enhance `_run_org_leaders_enrichment_silent()` in [main.py](main.py) and use existing role lists from [find_leadership.py](find_leadership.py):

```python
# Existing role lists in find_leadership.py:
COMPANY_ROLES = [
    "CEO", "President", "Managing Director",
    "Chief Technology Officer", "Chief Engineer", "Chief Scientist",
    "Head of HR", "Head of Recruitment",
]

UNIVERSITY_ROLES = [
    "Vice Chancellor", "Chancellor", "President", "Provost", "Dean", "Principal",
]
```

**Role Mapping by Organization Type:**

| Org Type | Roles to Find |
|----------|---------------|
| Company | CEO, CTO, Head of R&D, CMO (for healthcare), VP of relevant division |
| University | President/Chancellor, Provost, Dean of relevant school, Department Chair |
| Research Institute | Director, Principal Investigator, Scientific Director |
| Government | Secretary/Administrator, Deputy Director |

**Discovery Rules:**
1. For each direct person's employer → find top 3 leadership
2. For each organization mentioned → find CEO/Director + relevant department head
3. Prioritize roles most relevant to story category (from `story.category`)
4. Limit to max 5 indirect people per organization

**Output:** Stored in `story.indirect_people` (currently `story.org_leaders`)

---

## PHASE 2: VALIDATION & DISCOVERY

> **Principle:** Authoritative source required for every person record.

This phase validates direct people extracted in Phase 1 and discovers indirect people
(leadership) for each organization.

### Step 2.1: Validate Direct People
**Goal:** Confirm extracted information is accurate

**Implementation:** New `CredentialValidator` class or extend `CompanyMentionEnricher`

> **MVP Simplification:** Start with a single validation query. Add multi-source
> validation incrementally after the basic pipeline is working.

**MVP Validation Prompt (Single Query):**

```text
Verify this person works at the stated organization using web search.

Person: {person.name}
Organization: {person.affiliation}
Title (if known): {person.title}

Search the web and return JSON:
{
  "verified": true/false,
  "confidence": 0.0-1.0,
  "source_url": "URL where you found confirmation",
  "current_title": "their current title if found",
  "notes": "brief explanation"
}

Return verified=false if you cannot confirm they work there.
```

**⚠️ Cost Mitigation Strategies (REQUIRED):**

Validation prompts use Gemini API calls. With 10+ people per story, costs can escalate quickly.
Implement ALL of these mitigations:

| Strategy | Implementation | Expected Savings |
|----------|----------------|------------------|
| **Validation Cache** | Cache by `(normalized_name, org)` tuple for 30 days | ~60% reduction |
| **Skip Previously Validated** | Check if person was validated in another story | ~20% reduction |
| **Priority Sampling** | Validate only primary subjects + high-value people | ~40% reduction |
| **Batch Requests** | Combine 3-5 people per Gemini call when possible | ~50% reduction |

```python
# Validation cache implementation:
from functools import lru_cache
from text_utils import normalize_name

@lru_cache(maxsize=1000)
def get_cached_validation(name: str, org: str) -> dict | None:
    """Return cached validation result if exists."""
    key = f"{normalize_name(name)}|{normalize_name(org)}"
    return _validation_cache.get(key)

def should_validate(person: dict, story: Story) -> bool:
    """Determine if person needs validation (cost control)."""
    # Always validate primary subjects
    if person.get("role_in_story") == "primary_subject":
        return True
    # Skip if already validated
    if get_cached_validation(person["name"], person["employer"]):
        return False
    # Sample 50% of remaining people
    return hash(person["name"]) % 2 == 0
```

**Future Enhancement (Post-MVP):** Multi-source validation across official websites,
Google Scholar, and news articles. Only implement after MVP validation is stable.

**Validation Confidence Levels:**

| Level | Score | Criteria |
|-------|-------|----------|
| HIGH | 0.9+ | Found on official org website |
| MEDIUM | 0.7-0.89 | Found in news or academic profile |
| LOW | 0.5-0.69 | Partial match or outdated |
| FAILED | <0.5 | Cannot verify |

**Rate Limiting Consideration:**
With 10+ people per story, validation can trigger many Gemini calls. Mitigations:
- Cache validation results by (name, org) tuple
- Skip validation for people already validated in previous stories
- Use `AdaptiveRateLimiter` from [rate_limiter.py](rate_limiter.py)

**Output:** Update `PersonEntity` with validation data:

```python
# Fields to add to PersonEntity:
validation_status: str = "pending"  # pending, verified, failed
validation_confidence: float = 0.0
validation_source: str = ""
validation_date: str = ""
```

---

### Step 2.2: Discover and Validate Indirect People
**Goal:** Find current leadership for each organization

**Implementation:** Enhance `_run_org_leaders_enrichment_silent()` in [main.py](main.py) and `search_leadership_uc()` in [find_leadership.py](find_leadership.py)

**Discovery Process:**

**A. Web Search Strategy (UC Chrome Only):**

Uses UC Chrome (undetected_chromedriver) to avoid bot detection:

```python
# Search query format (from find_leadership.py):
def search_leadership_uc(org: str, role: str) -> dict | None:
    """Search for organization leadership using UC Chrome."""
    driver = _get_uc_driver()
    search_query = f"{role} {org} LinkedIn"
    # Try Google first, fallback to DuckDuckGo
    url = f"https://www.google.com/search?q={quote_plus(search_query)}"
    driver.get(url)
    # Parse results
```

> **Note:** UC Chrome bypasses Google bot detection. Use DuckDuckGo as fallback.
```

**B. Source Priority:**
1. Official organization website (leadership page)
2. Recent press releases/news (within 6 months)
3. LinkedIn company page (leadership section)
4. Crunchbase/Wikipedia (for basic info)

**C. Multi-Source Verification:**
- Require 2+ independent sources confirming same person
- Check appointment date (ensure current, not former)
- Use existing `is_university()` function to determine org type

**D. Attribute Extraction:**
Matches existing `org_leaders` structure in `Story` dataclass:

```python
# Current structure in database.py:
org_leaders: list[dict] = [
    {
        "name": "Jonathan Levin",
        "title": "President",
        "organization": "Stanford University",
        "linkedin_profile": "",
        "linkedin_urn": ""
    }
]
```

**Special Cases:**
- **Interim/Acting roles**: Flag in title field
- **Multiple people in role**: Check effective date, use most recent
- **Role doesn't exist**: Skip, don't force a match

**Output:**
```json
{
  "organization": "Stanford University",
  "role_searched": "President",
  "person_found": {
    "name": "Jonathan Levin",
    "title": "President",
    "start_date": "2024-08-01",
    "location": "Stanford, CA, USA",
    "specialty": "Economics",
    "validation_sources": [
      "https://president.stanford.edu",
      "https://news.stanford.edu/president-levin"
    ],
    "confidence": 0.98,
    "relationship_to_story": "employer_leadership"
  }
}
```

---

### Step 2.3: Build Complete People Registry
**Goal:** Create unified, validated dataset

**Deduplication:**
1. Check for duplicate names across direct and indirect
2. Use fuzzy matching for name variations
3. Compare employers and titles to confirm same person
4. Merge duplicates, keeping highest confidence data

**Registry Structure:**
```json
{
  "people_registry": {
    "direct_people": [...],
    "indirect_people": [...],
    "deduplication_log": [
      {
        "duplicate_found": "John Smith",
        "merged_from": ["direct", "indirect"],
        "kept_as": "direct"
      }
    ]
  },
  "validation_summary": {
    "total_direct": 5,
    "total_indirect": 12,
    "verified_count": 15,
    "failed_validation": 2,
    "average_confidence": 0.87
  }
}
```

---

## PHASE 3: LINKEDIN PROFILE MATCHING

> **Principle:** Identity-confirmation only. Never force a match.

This phase finds and validates LinkedIn profiles for all validated people.

### Step 3.1: Build Search Context
**Goal:** Compile all known attributes for effective LinkedIn search

**Implementation:** Uses existing `PersonSearchContext` from [linkedin_profile_lookup.py](linkedin_profile_lookup.py) and `PersonContext` from [profile_matcher.py](profile_matcher.py):

```python
# Existing dataclass in linkedin_profile_lookup.py:
@dataclass
class PersonSearchContext:
    first_name: str
    last_name: str
    company: Optional[str] = None
    department: Optional[str] = None
    position: Optional[str] = None
    location: Optional[str] = None
    role_type: Optional[str] = None  # "academic", "executive", etc.
    research_area: Optional[str] = None
    require_org_match: bool = True
    is_common_name: bool = False
    context_keywords: set[str] = field(default_factory=set)

# Existing dataclass in profile_matcher.py:
@dataclass
class PersonContext:
    name: str
    organization: str
    position: str = ""
    department: str = ""
    role_type: RoleType = RoleType.OTHER  # Enum: ACADEMIC, EXECUTIVE, etc.
    location: str = ""
    research_area: str = ""
    industry: str = ""
    story_title: str = ""
    story_category: str = ""
```

**Context Assembly:**
Use existing `build_context_keywords()` from [text_utils.py](text_utils.py):

```python
from text_utils import build_context_keywords, is_common_name, extract_first_last_name

def build_search_context(person: PersonEntity, story: Story) -> PersonSearchContext:
    first, last = extract_first_last_name(person.text)
    return PersonSearchContext(
        first_name=first,
        last_name=last,
        company=person.affiliation,
        position=person.title,
        location=person.location,
        role_type=_infer_role_type(person, story.category),
        is_common_name=is_common_name(first, last),
        context_keywords=build_context_keywords(
            company=person.affiliation,
            department=person.specialty,
            position=person.title,
        ),
    )
```

---

### Step 3.2: Multi-Stage LinkedIn Search
**Goal:** Find LinkedIn profile candidates using progressive search strategies

**Implementation:** Enhance `LinkedInCompanyLookup` in [linkedin_profile_lookup.py](linkedin_profile_lookup.py)

Uses existing UC Chrome infrastructure with Google/DuckDuckGo:

```python
# Existing search pattern in linkedin_profile_lookup.py:
def _search_linkedin_google(self, query: str) -> list[dict]:
    """Search Google for LinkedIn profiles using UC Chrome."""
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    # Parse results for linkedin.com/in/ URLs
```

**Search Tiers:**

1. **High Precision:** `site:linkedin.com/in/ "[Name]" "[Employer]"`
2. **Medium Precision:** `[Name] [Position] [Employer] LinkedIn`
3. **Broad Search:** `[Name] [Field] LinkedIn`

**Search Engine (UC Chrome Only):**

All searches use UC Chrome (undetected_chromedriver) with Google or DuckDuckGo:

> **⚠️ UC Chrome Resilience Warning:**
> UC Chrome can break with Chrome browser updates. Implement these safeguards:
> 1. **Version pinning**: Pin Chrome/chromedriver versions in CI/CD
> 2. **Health checks**: Test UC Chrome connectivity before batch processing
> 3. **Graceful degradation**: If UC Chrome fails, use org fallback (don't block pipeline)
> 4. **Monitoring**: Alert on UC Chrome failure rate >10%

```python
# Primary: Google via UC Chrome (UC Chrome bypasses bot detection)
def _search_linkedin_uc(self, query: str) -> list[dict]:
    """Search for LinkedIn profiles using UC Chrome browser."""
    driver = self._get_uc_driver()
    # Try Google first, fallback to DuckDuckGo if blocked
    url = f"https://www.google.com/search?q={quote_plus(query)}"
    driver.get(url)
    # Parse results for linkedin.com/in/ URLs
    return self._extract_linkedin_urls(driver.page_source)

def _search_linkedin_duckduckgo(self, query: str) -> list[dict]:
    """Fallback: Search DuckDuckGo for LinkedIn profiles."""
    driver = self._get_uc_driver()
    url = f"https://duckduckgo.com/?q={quote_plus(query)}"
    driver.get(url)
    return self._extract_linkedin_urls(driver.page_source)

def _search_with_fallback(self, query: str) -> list[dict]:
    """Search with full fallback chain."""
    try:
        results = self._search_linkedin_uc(query)  # Google via UC
        if results:
            return results
    except Exception as e:
        logger.warning(f"Google search failed: {e}")

    try:
        return self._search_linkedin_duckduckgo(query)  # DuckDuckGo fallback
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []  # Graceful failure - will trigger org fallback
```
```

> **Important:** Do NOT use Gemini with Google Search grounding for LinkedIn
> searches. The grounding API is detected as bot activity.
> UC Chrome with direct Google search bypasses detection.

**Candidate Extraction:**
Uses existing `_score_linkedin_candidate()` function pattern:
- Parse search results for LinkedIn URLs
- Extract vanity name (e.g., `/in/sarah-chen-phd`)
- Capture preview text (headline, location)
- Limit to top 10 candidates per tier

---

### Step 3.3: Inclusion Criteria Evaluation
**Goal:** Filter candidates that could plausibly be the right person

**REQUIRED (ALL must pass):**

1. **Name Match (Required Strength: Medium)**
   - First name appears in profile URL or title: +PASS
   - Last name appears in profile URL or title: +PASS
   - Both present: REQUIRED MINIMUM
   - Handle edge cases:
     - Middle initials OK: "Sarah Y Chen" matches "Sarah Chen"
     - Name order: "Chen Sarah" acceptable in some cultures
     - Nicknames: Check common variations (e.g., "Bob" for "Robert")

2. **Organization Connection (Required Strength: Any)**
   - Current employer matches: +STRONG PASS
   - Previous employer matches: +PASS (may have moved)
   - Related organization (e.g., hospital affiliated with university): +PASS
   - No organization info visible: +NEUTRAL (don't reject yet)
   - Unrelated organization: Check career path before rejecting

3. **Professional Field Alignment (Required Strength: Medium)**
   - Title contains role keywords (professor, researcher, scientist): +PASS
   - Industry matches expected field: +PASS
   - Seniority level appropriate: +PASS
   - Completely different field (e.g., sales, finance): -REJECT

**Output:**
```json
{
  "candidate_url": "https://linkedin.com/in/sarah-chen-phd-stanford",
  "inclusion_evaluation": {
    "name_match": {"passed": true, "strength": "strong", "notes": "Full name in URL"},
    "org_connection": {"passed": true, "strength": "strong", "notes": "Current employer matches"},
    "field_alignment": {"passed": true, "strength": "strong", "notes": "Professor title present"},
    "overall": "PASS_TO_SCORING"
  }
}
```

---

### Step 3.4: Exclusion Criteria Evaluation
**Goal:** Identify definite mismatches

**AUTOMATIC REJECTION (ANY triggers immediate rejection):**

1. **Wrong Profession/Field**
   - Profile shows unrelated profession:
     - Real estate agent when expecting researcher
     - Teacher when expecting executive
     - Student when expecting senior leader
   - Industry completely different (retail vs academia)
   - **Action:** REJECT, confidence = 0

2. **Geographic Mismatch**
   - Different country (unless expected international work)
   - Different continent
   - Profile shows "Relocated to [far location]"
   - **Exception:** Allow if person recently moved and timing uncertain
   - **Action:** REJECT if strong mismatch, confidence = 0

3. **Employer Contradiction**
   - Profile shows direct competitor organization
   - Profile shows "Former [expected employer]" with recent departure
   - Profile shows completely unrelated employer with no transition noted
   - **Action:** REJECT, confidence = 0

4. **Career Stage Mismatch**
   - Junior role when expecting senior (e.g., "Research Assistant" vs "Professor")
   - Entry-level when expecting executive
   - Student/intern when expecting professional
   - **Exception:** Allow if graduation/promotion timing could explain gap
   - **Action:** REJECT, confidence = 0

5. **Profile Quality Issues**
   - No profile photo (weak signal, don't auto-reject)
   - Fewer than 50 connections (suspicious for professionals)
   - No activity in 3+ years (likely outdated)
   - Minimal information (only name, no details)
   - Profile in "hiring mode" with generic info
   - **Action:** REJECT if multiple quality issues, confidence = 0.2

6. **Name Disambiguation**
   - Common name + no distinguishing info = reject
   - Different middle initial + conflicting info = reject
   - **Action:** REJECT, flag for manual review

**Common Name Handling Thresholds:**

Use `is_common_name()` from [text_utils.py](text_utils.py) with these rules:

> **⚠️ IMPLEMENTATION GAP:** The current `ProfileMatcher` class only has a penalty
> (`WEIGHT_COMMON_NAME_NO_SIGNALS = -0.5`) but does NOT implement hard thresholds.
> The `get_required_signals()` function below MUST be added to [profile_matcher.py](profile_matcher.py)
> as part of Phase 2 implementation.

| Name Commonality | Example | Required Signals for Match |
|------------------|---------|---------------------------|
| Very Common | John Smith, Mary Johnson | Org + Title + Location (all 3) |
| Common | Sarah Chen, Michael Lee | Org + (Title OR Location) |
| Uncommon | Xiaoying Zhang, Sanjay Gupta | Org match sufficient |
| Unique | Satya Nadella, Sundar Pichai | Name match sufficient |

```python
# NEW FUNCTION - Add to profile_matcher.py in Phase 2:
def get_required_signals(first_name: str, last_name: str) -> int:
    """Return minimum positive signals needed for common names.

    This is a HARD THRESHOLD, not a scoring penalty.
    If a candidate doesn't meet this threshold, reject regardless of score.
    """
    from text_utils import is_common_name, COMMON_FIRST_NAMES

    # Check if both first and last are common
    if first_name.lower() in COMMON_FIRST_NAMES:
        if is_common_name(first_name, last_name):
            return 3  # Very common: need org + title + location
        return 2  # Common first name: need org + one more
    return 1  # Uncommon: org match is enough

def meets_common_name_threshold(
    candidate: ProfileCandidate,
    person: PersonContext
) -> bool:
    """Check if candidate meets minimum signal threshold for common names."""
    first, last = person.name.split()[0], person.name.split()[-1]
    required = get_required_signals(first, last)

    # Count positive signals present
    signals_present = sum([
        1 for s in candidate.positive_signals
        if s.weight >= 1.0  # Only count significant signals
    ])

    if signals_present < required:
        logger.debug(f"Common name '{person.name}' needs {required} signals, "
                     f"only has {signals_present}")
        return False
    return True
```

**Output:**
```json
{
  "candidate_url": "https://linkedin.com/in/sarah-chen-realtor",
  "exclusion_evaluation": {
    "wrong_field": {"triggered": true, "reason": "Profile shows real estate agent, expected researcher"},
    "geographic_mismatch": {"triggered": false},
    "employer_contradiction": {"triggered": true, "reason": "Works at realty company, not university"},
    "career_stage_mismatch": {"triggered": false},
    "quality_issues": {"triggered": false},
    "overall": "REJECT",
    "confidence": 0.0
  }
}
```

---

### Step 3.5: Confidence Scoring System
**Goal:** Rank remaining candidates by match likelihood

**Implementation:** Uses existing `ProfileMatcher.score_candidate()` in [profile_matcher.py](profile_matcher.py)

**Existing Weight Constants (from ProfileMatcher class):**

```python
# Positive weights (from profile_matcher.py):
WEIGHT_NAME_MATCH = 3.0
WEIGHT_ORG_MATCH = 3.0
WEIGHT_ROLE_MATCH = 2.0
WEIGHT_DEPT_MATCH = 1.5
WEIGHT_LOCATION_MATCH = 1.0
WEIGHT_ROLE_TYPE_INDICATOR = 0.5

# Negative weights (from profile_matcher.py):
WEIGHT_WRONG_FIELD = -5.0
WEIGHT_WRONG_COUNTRY = -4.0
WEIGHT_INCOMPATIBLE_ROLE = -3.0
WEIGHT_CONFLICTING_ORG = -2.0
WEIGHT_COMMON_NAME_NO_SIGNALS = -0.5
```

**Confidence Thresholds (existing in ProfileMatcher):**

```python
HIGH_CONFIDENCE_THRESHOLD = 4.0   # Name + Org match = 6.0
MEDIUM_CONFIDENCE_THRESHOLD = 2.0 # Accept with just org match
```

**Match Confidence Levels (existing enum):**

```python
class MatchConfidence(str, Enum):
    HIGH = "high"        # >= 6.0 - Use without verification
    MEDIUM = "medium"    # 4.0-5.9 - Needs verification
    LOW = "low"          # < 4.0 - Reject, use fallback
    VERIFIED = "verified"      # Confirmed via profile page visit
    REJECTED = "rejected"      # Explicitly rejected
    ORG_FALLBACK = "org_fallback"  # Using org profile as fallback
```

**Scoring Output:** Uses existing `ProfileCandidate` dataclass:

```python
@dataclass
class ProfileCandidate:
    linkedin_url: str
    vanity_name: str
    result_title: str = ""
    result_snippet: str = ""
    confidence_score: float = 0.0
    positive_signals: list[MatchSignal] = field(default_factory=list)
    negative_signals: list[MatchSignal] = field(default_factory=list)
    verified: bool = False
```

---

### Step 3.6: Profile Verification for Medium-Confidence Matches
**Goal:** Make final determination for borderline cases (5.0-6.9 score)

**When to Verify:**
- Score is 5.0-6.9 (medium confidence)
- Multiple candidates with similar scores
- Any conflicting information detected
- High-value person (e.g., CEO, primary story subject)

**Verification Process:**

**A. Fetch Full LinkedIn Profile:**
- Visit actual LinkedIn profile page
- Extract all visible information:
  - Current headline
  - Current company and position
  - Location
  - Education (schools and degrees)
  - About/summary section
  - Top skills
  - Featured posts/articles
  - Recent activity

**B. Deep Comparison:**
```json
{
  "expected": {
    "employer": "Stanford University",
    "title": "Associate Professor",
    "location": "Stanford, CA"
  },
  "actual": {
    "headline": "Associate Professor of Neuroscience at Stanford University",
    "current_position": "Stanford University · Associate Professor",
    "location": "San Francisco Bay Area",
    "education": ["PhD Neuroscience - MIT", "BS Biology - Harvard"]
  },
  "comparison": {
    "employer": "MATCH",
    "title": "MATCH",
    "location": "MATCH (metro area)",
    "education_level": "MATCH (PhD confirmed)"
  },
  "decision": "ACCEPT",
  "final_confidence": 0.92
}
```

**C. Red Flags to Check:**
- Timeline inconsistencies (started role after story publication?)
- Multiple concurrent positions that don't make sense
- Degree doesn't match field (e.g., JD but working as scientist)
- Location conflict that can't be explained

**D. Decision Matrix:**
- All key attributes match → ACCEPT (confidence 0.85-0.95)
- 1 minor mismatch → ACCEPT with note (confidence 0.75-0.84)
- 1 major mismatch → REJECT or flag for manual review
- 2+ mismatches → REJECT

**Output:**
```json
{
  "verification_result": {
    "candidate_url": "https://linkedin.com/in/sarah-chen-stanford",
    "verification_method": "full_profile_fetch",
    "match_decision": "ACCEPT",
    "final_confidence": 0.88,
    "match_details": {
      "headline_match": true,
      "employer_confirmed": true,
      "title_confirmed": true,
      "location_confirmed": true,
      "education_appropriate": true
    },
    "notes": "Profile headline exactly matches expected position. Recent activity confirms current employment."
  }
}
```

---

### Step 3.7: Fallback to Organization Profile
**Goal:** Provide partial data when personal profile cannot be found

**When to Use Fallback:**
- No candidates pass inclusion criteria (score > 3.0)
- All candidates rejected by exclusion criteria
- Verification fails for all medium-confidence candidates
- Person explicitly has no LinkedIn presence (verified via search)

**Fallback Process:**

**A. Find Organization LinkedIn Page:**
```
Search: "[Organization name] LinkedIn"
Example: "Stanford University LinkedIn"
```

**B. Extract Organization Data:**
- Official LinkedIn company page URL
- Organization name and tagline
- Employee count range
- Location(s)
- Industry classification
- Company description

**C. Link Person to Organization Page:**
```json
{
  "name": "Dr. Sarah Chen",
  "linkedin_match_type": "organization_fallback",
  "linkedin_profile": null,
  "linkedin_organization": "https://linkedin.com/company/stanford-university",
  "match_confidence": "organization_only",
  "fallback_reason": "Could not confidently identify personal profile",
  "attempted_searches": 3,
  "candidates_evaluated": 12,
  "highest_candidate_score": 4.2,
  "notes": "Person works at this organization but personal LinkedIn profile not found or not identifiable with sufficient confidence"
}
```

**D. Manual Review Flag:**
- Flag record for potential manual LinkedIn search
- Provide search summary to help human reviewer
- Store all candidate URLs for reference

**Benefits of Fallback:**
- Maintains connection to organization
- Enables "@mention" of org page in LinkedIn outreach
- Preserves data integrity (doesn't force bad matches)
- Transparent about confidence level

---

## PHASE 4: DATABASE PERSISTENCE

> **Principle:** Atomic storage with full audit trail.

This phase stores all validated, LinkedIn-matched people into the Story record.

### Step 4.1: Structure for Direct People
**Goal:** Store people mentioned directly in the story

**Implementation:** Update `Story.story_people` field in [database.py](database.py)

**Current Schema (from Story dataclass):**

```python
# Current structure in database.py:
story_people: list[dict] = field(default_factory=list)
# Each dict contains:
# {
#     "name": "Dr. Jane Smith",
#     "title": "Lead Researcher",       # job_title field
#     "affiliation": "MIT",              # employer/company
#     "linkedin_profile": "",
#     "linkedin_urn": ""
# }
```

**Enhanced Schema (proposed changes):**

```python
# Enhanced structure for direct_people:
direct_people: list[dict] = [
    {
        # Core identification
        "name": "Dr. Sarah Chen",
        "normalized_name": "Sarah Chen",

        # Professional details
        "job_title": "Associate Professor of Neuroscience",
        "employer": "Stanford University",
        "location": "Stanford, CA, USA",
        "specialty": "Computational Neuroscience",

        # LinkedIn matching
        "linkedin_profile": "https://linkedin.com/in/sarah-chen-stanford",
        "linkedin_urn": "urn:li:person:ACoAAABcD1234",
        "match_confidence": "verified",  # MatchConfidence enum value
        "match_score": 8.5,

        # Validation
        "validation_status": "verified",
        "validation_source": "https://profiles.stanford.edu/sarah-chen",

        # Story context
        "role_in_story": "primary_subject",  # quoted, mentioned, author
        "is_direct": True,
    }
]
```

**Migration Strategy:**
- Rename `story_people` → `direct_people` (database reset, clean schema)
- Add new fields during schema creation

---

### Step 4.2: Structure for Indirect People
**Goal:** Store discovered leadership of mentioned organizations

**Implementation:** Update `Story.org_leaders` field in [database.py](database.py)

**Current Schema (from Story dataclass):**

```python
# Current structure in database.py:
org_leaders: list[dict] = field(default_factory=list)
# Each dict contains:
# {
#     "name": "John Doe",
#     "title": "CEO",
#     "organization": "BASF",
#     "linkedin_profile": "",
#     "linkedin_urn": ""
# }
```

**Enhanced Schema (proposed changes):**

```python
# Enhanced structure for indirect_people:
indirect_people: list[dict] = [
    {
        # Core identification
        "name": "Jonathan Levin",
        "normalized_name": "Jonathan Levin",

        # Professional details
        "job_title": "President",
        "employer": "Stanford University",
        "location": "Stanford, CA, USA",
        "specialty": "Economics",

        # LinkedIn matching
        "linkedin_profile": "https://linkedin.com/in/jonathanlevin",
        "linkedin_urn": "urn:li:person:ACoAAAXyZ9876",
        "match_confidence": "high",
        "match_score": 7.5,

        # Relationship to story
        "relationship_type": "employer_leadership",  # org_leadership, employer_of_subject
        "connected_to_org": "Stanford University",
        "discovery_reason": "President of organization where story subject works",

        # Validation
        "validation_status": "verified",
        "validation_source": "https://president.stanford.edu",
    }
]
```

**Migration Strategy:**
- Rename `org_leaders` → `indirect_people` (database reset, clean schema)
- Add relationship fields during schema creation

---

### Step 4.3: Story-Level Aggregation
**Goal:** Update Story record with complete enrichment data

**Implementation:** Extend `Story` dataclass in [database.py](database.py)

**Current Story Fields (from database.py):**

```python
@dataclass
class Story:
    # Existing enrichment fields:
    enrichment_status: str = "pending"  # pending, enriched, skipped, error
    organizations: list[str] = field(default_factory=list)  # ["BASF", "MIT"]
    story_people: list[dict] = field(default_factory=list)  # RENAME to direct_people
    org_leaders: list[dict] = field(default_factory=list)   # RENAME to indirect_people
```

**New Story Schema (clean slate):**

```python
@dataclass
class Story:
    # ... other fields ...

    # Enrichment fields (renamed for clarity):
    enrichment_status: str = "pending"
    organizations: list[str] = field(default_factory=list)
    direct_people: list[dict] = field(default_factory=list)    # was: story_people
    indirect_people: list[dict] = field(default_factory=list)  # was: org_leaders

    # New metadata fields:
    enrichment_log: dict = field(default_factory=dict)
    enrichment_quality: str = ""  # "high", "medium", "low", "failed"
```

**Schema Creation (fresh database):**

```python
# Add to Story dataclass in database.py:

# Enhanced enrichment tracking
enrichment_log: dict = field(default_factory=dict)
# Structure:
# {
#     "enriched_at": "2025-01-12T11:00:00Z",
#     "processing_time_seconds": 180,
#     "direct_people_count": 3,
#     "indirect_people_count": 5,
#     "linkedin_match_rate": 0.875,
#     "average_confidence": 0.89,
#     "errors": [],
#     "warnings": []
# }

# Quality metrics
enrichment_quality: str = "high"  # high, medium, low, failed
needs_manual_review: bool = False
```

**Database Migration:**
Use existing `Database._migrate_add_column()` pattern:

```python
self._migrate_add_column(
    cursor, "enrichment_log", "TEXT DEFAULT '{}'", existing_columns
)
self._migrate_add_column(
    cursor, "enrichment_quality", "TEXT DEFAULT 'pending'", existing_columns
)
```

---

### Step 4.4: Error Handling and Logging
**Goal:** Track failures and edge cases

**Implementation:** Use existing logging patterns and extend error handling

**Logging Pattern (from existing codebase):**

```python
import logging
logger = logging.getLogger(__name__)

# Use existing patterns from main.py:
logger.info(f"[{i}/{total}] Enriching: {story.title}")
logger.info(f"  ✓ Found {len(profiles)} LinkedIn profiles")
logger.info(f"  ⏭ No organizations or people found")
logger.error(f"  ! Error: {e}")
```

**Error Categories:**

| Error Type | Handling Strategy | Example |
|------------|-------------------|---------|
| Network timeout | Retry with backoff via `api_client` | Leadership search fails |
| API rate limit | Use `AdaptiveRateLimiter` | Gemini 429 error |
| Validation failed | Log and continue | Cannot verify credentials |
| LinkedIn no match | Use org fallback | No profile found |
| Parse error | Log and skip entity | Malformed JSON response |

**Error Logging Structure:**

```python
# Store in story.enrichment_log["errors"]:
{
    "error_type": "validation_failed",
    "entity_name": "J. Smith",
    "reason": "Could not verify employment",
    "attempted_sources": ["org website", "Google Scholar"],
    "resolution": "flagged_for_review",
    "timestamp": "2025-01-12T10:20:00Z"
}
```

---

### Step 4.5: Retry and Recovery Strategy
**Goal:** Handle transient failures and optimize processing

**Implementation:** Use existing `AdaptiveRateLimiter` from [rate_limiter.py](rate_limiter.py) and `api_client` patterns

**Existing Retry Infrastructure:**

```python
# From api_client.py - already has retry built in:
class RateLimitedAPIClient:
    def __init__(self):
        # Gemini API - generous limits (60 RPM)
        self.gemini_limiter = AdaptiveRateLimiter(
            initial_fill_rate=1.0,
            min_fill_rate=0.1,
            max_fill_rate=2.0,
            success_threshold=5,
            rate_limiter_429_backoff=0.5,
        )

        # HTTP requests (web scraping) - moderate
        self.http_limiter = AdaptiveRateLimiter(
            initial_fill_rate=2.0,
            min_fill_rate=0.2,
            max_fill_rate=5.0,
        )
```

**Validation Failures:**
- If validation fails, try alternate sources (org website, news)
- If all sources fail, mark for manual review (don't block pipeline)
- Store attempted sources in enrichment_log

**LinkedIn Search Failures (UC Chrome):**
- If Google search returns no results → try DuckDuckGo
- If UC Chrome fails → retry with fresh browser instance
- If all searches fail → use organization fallback
- Never block story processing due to LinkedIn match failure

> **Note:** All web searches use UC Chrome with Google (primary) or DuckDuckGo (fallback).
> Do not use Gemini grounding for LinkedIn searches.

**Resume Capability:**
Use existing `enrichment_status` field for checkpointing:

```python
# Story.enrichment_status values:
# "pending" - Not started
# "in_progress" - Currently processing (for resume)
# "enriched" - Complete
# "error" - Failed, needs retry
# "skipped" - Skipped intentionally
```

---

### Step 4.6: Data Quality Validation
**Goal:** Ensure data integrity before final storage

**Pre-Storage Validation Checks:**

**A. Required Fields Check:**
```javascript
// Pseudo-code for validation
function validatePersonRecord(person) {
  const required = ['name', 'employer'];
  const missing = required.filter(field => !person[field]);

  if (missing.length > 0) {
    return {
      valid: false,
      error: `Missing required fields: ${missing.join(', ')}`
    };
  }

  return { valid: true };
}
```

**B. Data Type Validation:**
- URLs must be valid HTTP(S) URLs
- Dates must be valid ISO 8601 format
- Confidence scores must be 0.0-1.0
- Person IDs must be unique within story

**C. Relationship Integrity:**
- Every indirect person must reference valid organization
- Organization IDs referenced must exist in organizations array
- Person IDs in relationships must exist

**D. Confidence Thresholds:**
```json
{
  "quality_thresholds": {
    "min_validation_confidence": 0.5,
    "min_linkedin_confidence": 3.0,
    "warn_if_below": {
      "validation_confidence": 0.7,
      "linkedin_confidence": 5.0
    },
    "block_if_below": {
      "validation_confidence": 0.3,
      "linkedin_confidence": 0.0
    }
  }
}
```

**E. Consistency Checks:**
- Person name in direct_people should match extracted name
- LinkedIn profile employer should align with stated employer
- Location should be consistent across validation sources

**Validation Report:**
```json
{
  "validation_report": {
    "story_id": "story_12345",
    "validated_at": "2025-01-12T11:00:00Z",
    "checks_passed": 23,
    "checks_failed": 2,
    "warnings": 1,
    "errors": [
      {
        "severity": "error",
        "check": "required_fields",
        "entity": "dp_002",
        "message": "Missing job_title field",
        "resolution": "Field populated with 'Unknown' placeholder"
      }
    ],
    "warnings": [
      {
        "severity": "warning",
        "check": "confidence_threshold",
        "entity": "ip_003",
        "message": "LinkedIn match confidence below recommended threshold (4.2 < 5.0)",
        "resolution": "Accepted but flagged for review"
      }
    ],
    "overall_status": "passed_with_warnings"
  }
}
```

---

### Step 4.7: Database Transaction Strategy
**Goal:** Atomic storage to prevent partial updates

**Implementation:** Use existing `Database._get_connection()` context manager from [database.py](database.py)

**Existing Transaction Pattern:**

```python
# From database.py - already handles commit/rollback:
@contextmanager
def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(self.db_name)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
```

**Enrichment Transaction Flow:**

```python
def enrich_story_atomic(self, story: Story) -> bool:
    """Atomically enrich a story with all entity data."""
    with self.db._get_connection() as conn:
        cursor = conn.cursor()
        try:
            # 1. Mark in progress
            cursor.execute(
                "UPDATE stories SET enrichment_status = ? WHERE id = ?",
                ("in_progress", story.id)
            )

            # 2. Update all fields
            cursor.execute("""
                UPDATE stories SET
                    organizations = ?,
                    story_people = ?,
                    org_leaders = ?,
                    enrichment_status = ?,
                    enrichment_log = ?
                WHERE id = ?
            """, (
                json.dumps(story.organizations),
                json.dumps(story.story_people),
                json.dumps(story.org_leaders),
                "enriched",
                json.dumps(enrichment_log),
                story.id
            ))
            # Auto-commit on context exit
            return True
        except Exception as e:
            logger.error(f"Enrichment failed: {e}")
            # Auto-rollback on exception
            raise
```

**Upsert Logic:**
- Use person identifiers (name + employer) to detect duplicates
- Update existing records with higher confidence data
- Preserve historical LinkedIn match attempts

---

## MONITORING & OPTIMIZATION

### Step 5.1: Performance Metrics
**Goal:** Track pipeline health and efficiency

**Implementation:** Extend existing stats methods in [linkedin_profile_lookup.py](linkedin_profile_lookup.py)

**Existing Stats Methods:**

```python
# Already available in LinkedInCompanyLookup:
def get_cache_stats(self) -> dict[str, int]:
    """Cache statistics for monitoring."""
    return {
        "person_cache_size": len(self._person_cache),
        "company_cache_size": len(self._company_cache),
        "department_cache_size": len(self._department_cache),
    }

def get_timing_stats(self) -> dict[str, Any]:
    """Timing statistics for monitoring."""
    return {
        "total_requests": self._request_count,
        "avg_response_time": self._avg_response_time,
        "last_request_time": self._last_request_time,
    }

def get_gemini_stats(self) -> dict[str, Any]:
    """Gemini API usage statistics."""
    return {
        "gemini_requests": self._gemini_request_count,
        "gemini_successes": self._gemini_success_count,
        "gemini_avg_time": self._gemini_avg_time,
    }
```

**Extended Enrichment Metrics:**

```python
@dataclass
class EnrichmentMetrics:
    """Comprehensive enrichment pipeline metrics."""
    # Throughput
    stories_processed: int = 0
    direct_people_found: int = 0
    indirect_people_found: int = 0
    linkedin_matches: int = 0

    # Quality
    high_confidence_matches: int = 0
    medium_confidence_matches: int = 0
    low_confidence_matches: int = 0
    rejected_matches: int = 0

    # Timing (seconds)
    avg_story_enrichment_time: float = 0.0
    avg_validation_time: float = 0.0
    avg_linkedin_search_time: float = 0.0

    # API Usage
    gemini_calls: int = 0        # LLM processing (not web search)
    google_searches: int = 0     # Via UC Chrome (LinkedIn, leadership)
    duckduckgo_searches: int = 0 # Fallback searches
    # Note: Article content comes from story search, no separate fetch counter
```

**Key Metrics:**

- Stories processed per hour
- People extracted per story (direct/indirect)
- LinkedIn match rate by confidence level
- API calls by service (Gemini, Google, DuckDuckGo)
- Cache hit rates from `get_cache_stats()`
- Average enrichment time per story

---

### Step 5.2: Quality Assurance Checks
**Goal:** Continuous validation of system accuracy

**Implementation:** Leverage existing `MatchConfidence` enum and validation patterns

**Automated QA Using Existing Structures:**

```python
from profile_matcher import MatchConfidence

@dataclass
class QACheckResult:
    """Results of quality assurance checks."""
    story_id: str
    checks_passed: int = 0
    checks_failed: int = 0
    warnings: list[str] = field(default_factory=list)

    # Per-check results
    linkedin_url_validity: bool = True
    employer_consistency: bool = True
    duplicate_detection: bool = True
    confidence_distribution_ok: bool = True

def validate_enrichment_quality(story: Story) -> QACheckResult:
    """Run QA checks on enriched story."""
    result = QACheckResult(story_id=str(story.id))

    # Check 1: LinkedIn URL validity (must start with linkedin.com)
    for person in story.story_people + story.org_leaders:
        if linkedin := person.get("linkedin_url"):
            if "linkedin.com/in/" not in linkedin:
                result.linkedin_url_validity = False
                result.checks_failed += 1
            else:
                result.checks_passed += 1

    # Check 2: No duplicates across direct/indirect
    direct_names = {p.get("name") for p in story.story_people}
    indirect_names = {p.get("name") for p in story.org_leaders}
    if duplicates := direct_names & indirect_names:
        result.duplicate_detection = False
        result.warnings.append(f"Duplicate names: {duplicates}")

    # Check 3: Confidence distribution reasonable
    confidence_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for person in story.story_people + story.org_leaders:
        conf = person.get("match_confidence", "LOW")
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

    total = sum(confidence_counts.values())
    if total > 0 and confidence_counts.get("LOW", 0) / total > 0.4:
        result.confidence_distribution_ok = False
        result.warnings.append("High proportion of low-confidence matches")

    return result
```

**Sampling for Manual Review:**

- Randomly sample 5% of enriched stories
- Prioritize low-confidence matches for review
- Flag unusual patterns (all failures, all low confidence)

---

### Step 5.3: Learning and Improvement
**Goal:** Continuously improve matching accuracy

**Implementation:** Track and tune `ProfileMatcher` weights from [profile_matcher.py](profile_matcher.py)

**Current Weight Constants:**

```python
# From ProfileMatcher - candidates for tuning:
WEIGHT_NAME_MATCH = 3.0      # Exact name match
WEIGHT_ORG_MATCH = 3.0       # Organization match
WEIGHT_ROLE_MATCH = 2.0      # Role/title match
WEIGHT_LOCATION_MATCH = 1.5  # Location match
WEIGHT_CONTEXT_MATCH = 1.0   # Context keyword match
WEIGHT_WRONG_FIELD = -5.0    # Wrong field penalty

# Thresholds
HIGH_CONFIDENCE_THRESHOLD = 4.0
MEDIUM_CONFIDENCE_THRESHOLD = 2.0
```

**Tuning Insights Collection:**

```python
@dataclass
class MatchOutcome:
    """Track match outcomes for weight tuning."""
    story_id: str
    person_name: str
    search_strategy: str  # "google_search" or "duckduckgo_search" via UC Chrome
    factors_present: list[str]  # ["name_match", "org_match", ...]
    score: float
    confidence: MatchConfidence
    human_verified: bool = False
    was_correct: bool | None = None

def analyze_weight_effectiveness(outcomes: list[MatchOutcome]) -> dict:
    """Analyze which factors predict accurate matches."""
    factor_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})

    for outcome in outcomes:
        if outcome.human_verified:
            for factor in outcome.factors_present:
                factor_accuracy[factor]["total"] += 1
                if outcome.was_correct:
                    factor_accuracy[factor]["correct"] += 1

    return {
        factor: stats["correct"] / stats["total"]
        for factor, stats in factor_accuracy.items()
        if stats["total"] >= 10  # Minimum sample size
    }
```

**Common Failure Patterns to Track:**

| Pattern | Example | Resolution |
|---------|---------|------------|
| Common name disambiguation | "John Smith at Microsoft" | Require location + title |
| Recent job change | LinkedIn shows old employer | Accept recent changes |
| No LinkedIn profile | Senior exec with no presence | Verify via org page |

---

### Step 5.4: Alerts and Notifications
**Goal:** Proactive monitoring using existing logging patterns

**Implementation:** Extend existing `logger` configuration from main.py

**Alert Integration:**

```python
import logging
from notifications import send_notification  # If available

# Severity levels using existing logger
def alert_critical(message: str, context: dict):
    """Critical alert requiring immediate action."""
    logger.critical(f"ALERT: {message}", extra=context)
    # Could integrate with notifications.py if implemented

def alert_warning(message: str, context: dict):
    """Warning requiring review within 24h."""
    logger.warning(f"ALERT: {message}", extra=context)

def alert_info(message: str, context: dict):
    """Informational for weekly review."""
    logger.info(f"ALERT: {message}", extra=context)
```

**Alert Triggers:**

**Critical (immediate):**
- Enrichment success rate drops below 50%
- Database transaction failures
- API authentication failures

**Warning (24h review):**
- Success rate 50-70%
- High rate of low-confidence matches (>40%)
- Processing time >5 minutes per story

**Info (weekly):**
- Success rate trends
- New failure patterns detected

---

### Step 5.5: Reporting Dashboard
**Goal:** Visibility into system health

**Implementation:** Aggregate stats from existing methods

**Stats Aggregation:**

```python
def get_enrichment_dashboard_stats() -> dict:
    """Aggregate stats for dashboard display."""
    lookup = LinkedInCompanyLookup.get_instance()

    return {
        "cache": lookup.get_cache_stats(),
        "timing": lookup.get_timing_stats(),
        "gemini": lookup.get_gemini_stats(),
        "enrichment": {
            "stories_today": db.count_stories_enriched_today(),
            "success_rate": db.calculate_enrichment_success_rate(),
            "avg_processing_time": metrics.avg_story_enrichment_time,
        },
        "quality": {
            "high_confidence_pct": metrics.high_confidence_matches / max(metrics.linkedin_matches, 1),
            "rejection_rate": metrics.rejected_matches / max(metrics.linkedin_matches, 1),
        }
    }
```

**Dashboard Metrics:**

**Real-time:**
- Stories processed today
- Current success rate
- Active jobs
- Error rate (last hour)

**Historical:**
- Daily volume (7/30/90 days)
- Success rate trends
- Top failure reasons

**Entity Statistics:**
- Direct/indirect people counts
- Unique organizations
- LinkedIn match rates by confidence
- Top organizations mentioned

---

## ADVANCED FEATURES & OPTIMIZATIONS

### Step 6.1: Batch Processing Optimization
**Goal:** Efficiently process multiple stories

**Implementation:** Extend existing `async_utils.py` patterns and `AdaptiveRateLimiter`

**Batch Processing Using Existing async_utils:**

```python
from async_utils import run_async, gather_with_concurrency
from rate_limiter import AdaptiveRateLimiter

class BatchEnrichmentProcessor:
    """Process multiple stories with controlled concurrency."""

    def __init__(self):
        self.rate_limiter = AdaptiveRateLimiter(
            initial_rate=1.0,
            min_rate=0.1,
            max_rate=5.0
        )
        self.max_concurrent_stories = 3
        self.max_concurrent_validations = 5

    async def process_stories_batch(
        self,
        stories: list[Story]
    ) -> list[EnrichmentResult]:
        """Process multiple stories with rate limiting."""
        semaphore = asyncio.Semaphore(self.max_concurrent_stories)

        async def process_one(story: Story) -> EnrichmentResult:
            async with semaphore:
                await self.rate_limiter.acquire()
                return await self.enrich_story(story)

        results = await asyncio.gather(
            *[process_one(s) for s in stories],
            return_exceptions=True
        )
        return results
```

**Optimization Strategies:**

- Deduplicate validation requests across stories
- Cache organization leadership across stories
- Use existing `_person_cache`, `_company_cache` from `LinkedInCompanyLookup`
- Process high-priority stories first

---

### Step 6.2: Incremental Updates
**Goal:** Refresh data without full reprocessing

**Implementation:** Track changes using `enrichment_log` field in Story

**Update Triggers:**

- Story source updated (new articles added)
- Manual correction made (human feedback)
- Scheduled refresh (quarterly for org_leaders)
- Profile 404 detected (LinkedIn URL broken)

**Incremental Enrichment:**

```python
def needs_refresh(story: Story, refresh_policy: dict) -> bool:
    """Determine if story needs re-enrichment."""
    if not story.enrichment_status:
        return True  # Never enriched

    log = story.enrichment_log or {}
    last_enriched = log.get("completed_at")

    if not last_enriched:
        return True

    # Check age against policy
    days_old = (datetime.now() - datetime.fromisoformat(last_enriched)).days

    if story.enrichment_status == "partial":
        return True  # Incomplete enrichment

    # Direct people: only refresh if source changed
    if story.content_hash != log.get("source_hash"):
        return True

    # Indirect people: quarterly refresh
    if days_old > refresh_policy.get("indirect_refresh_days", 90):
        return True

    return False

def incremental_enrich(story: Story) -> Story:
    """Only re-process changed portions."""
    changes = detect_changes(story)

    if changes.source_updated:
        # Re-extract entities from new content
        story = extract_entities_incremental(story, changes.new_content)

    if changes.org_leaders_stale:
        # Refresh leadership only
        story = refresh_org_leaders(story)

    return story
```

---

### Step 6.3: Entity Resolution Across Stories
**Goal:** Identify same person appearing in multiple stories

**Implementation:** Use LinkedIn URN as global identifier

**Person Identity Using Existing Fields:**

```python
def generate_person_key(person: dict) -> str:
    """Generate consistent identity key for cross-story matching."""
    # Prefer LinkedIn URN if available (most reliable)
    if urn := person.get("linkedin_urn"):
        return f"urn:{urn}"

    # Fallback to name + employer (from text_utils normalization)
    name = normalize_name(person.get("name", ""))
    employer = normalize_name(person.get("employer", ""))
    return f"name:{name}|employer:{employer}"

def find_existing_person(db: Database, person: dict) -> dict | None:
    """Check if person already exists across stories."""
    key = generate_person_key(person)

    # Search existing stories for matching person
    if key.startswith("urn:"):
        urn = key.split(":")[1]
        return db.find_person_by_linkedin_urn(urn)

    # Fuzzy match on name + employer
    return db.find_person_by_attributes(
        name=person.get("name"),
        employer=person.get("employer")
    )
```

**Cross-Story Benefits:**

- Avoid re-validating same person across stories
- Build comprehensive person profiles over time
- Track person's presence across multiple stories
- Share LinkedIn match results

---

### Step 6.4: External Data Integration
**Goal:** Enrich with additional data sources

**Implementation:** Extend existing patterns from `api_client.py`

**Integration via RateLimitedAPIClient:**

```python
# Using existing api_client.py patterns:
api_client = RateLimitedAPIClient.get_instance()

async def enrich_from_external_sources(person: dict) -> dict:
    """Add external data to person record."""
    enhanced = person.copy()

    # Example: Google Scholar (if specialty is academic)
    if person.get("specialty") in ["researcher", "professor", "scientist"]:
        try:
            scholar_data = await api_client.get_with_retry(
                f"https://scholar.google.com/citations?user={person.get('name')}",
                endpoint_type="http"
            )
            if scholar_data:
                enhanced["google_scholar"] = extract_scholar_profile(scholar_data)
        except Exception as e:
            logger.debug(f"Scholar lookup failed: {e}")

    return enhanced
```

**Potential Integrations:**

- CRM systems (check existing contacts)
- Google Scholar (academic profiles)
- Company websites (leadership pages)
- News APIs (recent mentions)

---

### Step 6.5: Export Options
**Goal:** Make enriched data accessible

**Implementation:** Add export methods to `Database` class

**Export Functionality:**

```python
def export_story_people(self, story_id: int, format: str = "json") -> str:
    """Export people data in various formats."""
    story = self.get_story(story_id)
    if not story:
        return ""

    people = {
        "direct": story.story_people or [],
        "indirect": story.org_leaders or [],
    }

    if format == "json":
        return json.dumps(people, indent=2, default=str)

    elif format == "csv":
        rows = []
        for category, persons in people.items():
            for p in persons:
                rows.append({
                    "category": category,
                    "name": p.get("name"),
                    "title": p.get("title"),
                    "employer": p.get("employer"),
                    "linkedin_url": p.get("linkedin_url"),
                    "confidence": p.get("match_confidence"),
                })
        return pandas_to_csv(rows)  # If pandas available

    return ""
```

**Available Formats:**

- JSON (full structured data)
- CSV (flattened for spreadsheets)
- Markdown (human-readable summary)

---

## TESTING STRATEGY

> **Critical:** Test each phase independently before integration.

### Unit Tests

**Uses existing `TestSuite` from [test_framework.py](test_framework.py):**

```python
# enrichment_tests.py
"""Unit tests for enrichment pipeline using TestSuite framework."""

from test_framework import TestSuite, suppress_logging
from profile_matcher import ProfileMatcher, PersonContext, MatchConfidence, RoleType


# =============================================================================
# Test Fixtures
# =============================================================================

def mock_candidate(name_match: bool = False, org_match: bool = False,
                   headline: str = "") -> dict:
    """Create a mock LinkedIn candidate for testing."""
    return {
        "url": "https://linkedin.com/in/test-user",
        "name_match": name_match,
        "org_match": org_match,
        "headline": headline,
    }

MOCK_CANDIDATES = {
    "exact_match": {
        "url": "https://linkedin.com/in/sarah-chen-stanford",
        "headline": "Associate Professor at Stanford University",
        "location": "San Francisco Bay Area",
    },
    "wrong_field": {
        "url": "https://linkedin.com/in/sarah-chen-realtor",
        "headline": "Luxury Real Estate Specialist",
        "location": "Los Angeles, CA",
    },
    "common_name_ambiguous": {
        "url": "https://linkedin.com/in/john-smith-12345",
        "headline": "Software Engineer",
        "location": "Seattle, WA",
    },
}


# =============================================================================
# Common Name Threshold Tests
# =============================================================================

def build_common_name_tests() -> TestSuite:
    """Build test suite for common name handling."""
    suite = TestSuite("Common Name Threshold Tests")
    matcher = ProfileMatcher()

    def test_very_common_name_requires_three_signals() -> None:
        """John Smith needs org + title + location."""
        context = PersonContext(name="John Smith", organization="Microsoft")
        candidate = mock_candidate(name_match=True, org_match=True)
        result = matcher.score_candidate(candidate, context)
        assert result.confidence == MatchConfidence.LOW, \
            f"Expected LOW confidence for common name, got {result.confidence}"

    def test_uncommon_name_needs_only_org() -> None:
        """Xiaoying Zhang needs only org match."""
        context = PersonContext(name="Xiaoying Zhang", organization="Stanford")
        candidate = mock_candidate(name_match=True, org_match=True)
        result = matcher.score_candidate(candidate, context)
        assert result.confidence in [MatchConfidence.MEDIUM, MatchConfidence.HIGH], \
            f"Expected MEDIUM+ confidence for uncommon name, got {result.confidence}"

    suite.add_test("Very common name requires 3 signals", test_very_common_name_requires_three_signals)
    suite.add_test("Uncommon name needs only org match", test_uncommon_name_needs_only_org)
    return suite


# =============================================================================
# Exclusion Criteria Tests
# =============================================================================

def build_exclusion_tests() -> TestSuite:
    """Build test suite for exclusion criteria."""
    suite = TestSuite("Exclusion Criteria Tests")
    matcher = ProfileMatcher()

    def test_wrong_field_rejection() -> None:
        """Real estate agent should not match researcher."""
        context = PersonContext(
            name="Sarah Chen",
            organization="MIT",
            role_type=RoleType.ACADEMIC
        )
        candidate = mock_candidate(headline="Real Estate Agent at Remax")
        result = matcher.evaluate_exclusions(candidate, context)
        assert result.rejected, "Should reject wrong field candidate"
        assert "wrong_field" in result.reasons, f"Expected wrong_field reason, got {result.reasons}"

    def test_geographic_mismatch_rejection() -> None:
        """Different continent should trigger rejection."""
        context = PersonContext(
            name="John Doe",
            organization="Google USA",
            location="Mountain View, CA"
        )
        candidate = mock_candidate(headline="Engineer at Google India")
        result = matcher.evaluate_exclusions(candidate, context)
        assert result.rejected or result.score < 0, "Should penalize geographic mismatch"

    suite.add_test("Wrong field triggers rejection", test_wrong_field_rejection)
    suite.add_test("Geographic mismatch rejection", test_geographic_mismatch_rejection)
    return suite


# =============================================================================
# Run All Tests
# =============================================================================

def run_enrichment_tests() -> None:
    """Run all enrichment-related unit tests."""
    with suppress_logging():
        common_name_suite = build_common_name_tests()
        common_name_suite.run(verbose=True)

        exclusion_suite = build_exclusion_tests()
        exclusion_suite.run(verbose=True)


if __name__ == "__main__":
    run_enrichment_tests()
```

### Integration Tests

**Follows [integration_tests.py](integration_tests.py) patterns:**

```python
# enrichment_integration_tests.py
"""Integration tests for enrichment pipeline."""

from dataclasses import dataclass, field
from typing import Any

from test_framework import TestSuite, suppress_logging
from database import Database, Story


# =============================================================================
# Test Fixtures
# =============================================================================

@dataclass
class MockLinkedInLookup:
    """Mock LinkedIn lookup for integration testing."""
    should_fail: bool = False
    candidates: list[dict] = field(default_factory=list)
    call_count: int = 0

    def search_person(self, name: str, org: str) -> list[dict]:
        self.call_count += 1
        if self.should_fail:
            raise ConnectionError("Simulated API failure")
        return self.candidates

    def reset(self) -> None:
        self.should_fail = False
        self.candidates = []
        self.call_count = 0


class EnrichmentIntegrationSuite:
    """Integration test suite for enrichment pipeline."""

    def __init__(self) -> None:
        self.db: Database | None = None
        self.lookup: MockLinkedInLookup | None = None

    def setup(self) -> None:
        """Set up test fixtures."""
        self.db = Database(":memory:")  # In-memory for tests
        self.lookup = MockLinkedInLookup()

    def teardown(self) -> None:
        """Clean up test fixtures."""
        self.db = None
        self.lookup = None

    def build_suite(self) -> TestSuite:
        """Build the integration test suite."""
        suite = TestSuite("Enrichment Pipeline Integration Tests")
        suite.set_setup(self.setup)
        suite.set_teardown(self.teardown)

        def test_entity_extraction_uses_existing_content() -> None:
            """Entity extraction uses content already in story (no re-fetch)."""
            # Story already has content from search phase
            story = Story(
                id=1,
                title="Test Story",
                summary="Summary with Dr. Jane Smith from MIT.",
                source_links=["https://example.com/article"],
                # Entities extracted during story search:
                direct_people=[{"name": "Dr. Jane Smith", "employer": "MIT"}],
                organizations=["MIT"]
            )
            # Enrichment uses already-extracted entities
            from company_mention_enricher import EntityEnrichmentPipeline
            pipeline = EntityEnrichmentPipeline(self.db, self.lookup)
            # No article fetch occurs - entities already in story
            assert len(story.direct_people) == 1, \
                "Entities should be populated during story search"

        def test_org_fallback_when_no_personal_match() -> None:
            """Use org profile when personal profile not found."""
            self.lookup.candidates = []  # No matches
            person = {"name": "Unknown Person", "employer": "Google"}
            from company_mention_enricher import EntityEnrichmentPipeline
            pipeline = EntityEnrichmentPipeline(self.db, self.lookup)
            result = pipeline._match_linkedin_profiles([person])
            assert result[0].get("match_confidence") == "org_fallback", \
                "Should use org fallback when no personal match"

        def test_enrichment_log_populated() -> None:
            """Enrichment log captures processing details."""
            story = Story(id=1, title="Test", summary="Test summary")
            from company_mention_enricher import EntityEnrichmentPipeline
            pipeline = EntityEnrichmentPipeline(self.db, self.lookup)
            result = pipeline.enrich_story(story)
            assert "completed_at" in result.enrichment_log, \
                "enrichment_log should have completion timestamp"

        suite.add_test("Article fallback to summary", test_article_fallback_to_summary)
        suite.add_test("Org fallback when no personal match", test_org_fallback_when_no_personal_match)
        suite.add_test("Enrichment log populated", test_enrichment_log_populated)
        return suite


# =============================================================================
# Regression Tests (Known Failures)
# =============================================================================

def build_regression_tests() -> TestSuite:
    """Tests derived from actual production failures."""
    suite = TestSuite("Regression Tests - Known Failures")

    def test_paywall_does_not_block_pipeline() -> None:
        """Issue: WSJ articles blocked entire enrichment."""
        story = Story(
            id=1,
            title="WSJ Article",
            summary="Content about tech company",
            source_links=["https://wsj.com/paywalled"]
        )
        from company_mention_enricher import EntityEnrichmentPipeline
        db = Database(":memory:")
        pipeline = EntityEnrichmentPipeline(db, MockLinkedInLookup())
        result = pipeline.enrich_story(story)
        assert result.status != "error", "Paywall should not block enrichment"

    def test_common_name_uses_strict_matching() -> None:
        """Issue: Wrong John Smith matched at Microsoft."""
        person = {
            "name": "John Smith",
            "employer": "Microsoft",
            "title": "Principal Engineer"
        }
        from profile_matcher import ProfileMatcher
        matcher = ProfileMatcher()
        # Without location, should NOT be high confidence
        # Should either match with high confidence or use org fallback
        result = matcher.requires_additional_signals(person)
        assert result is True, "Common name should require additional signals"

    suite.add_test("Paywall does not block pipeline", test_paywall_does_not_block_pipeline)
    suite.add_test("Common name uses strict matching", test_common_name_uses_strict_matching)
    return suite


# =============================================================================
# Run All Tests
# =============================================================================

def run_integration_tests() -> None:
    """Run all enrichment integration tests."""
    with suppress_logging():
        integration_suite = EnrichmentIntegrationSuite()
        integration_suite.build_suite().run(verbose=True)

        regression_suite = build_regression_tests()
        regression_suite.run(verbose=True)


if __name__ == "__main__":
    run_integration_tests()
```

### Manual Testing Checklist

Before deploying changes, manually verify:

- [ ] Run enrichment on 5 stories with known people (verify correct matches)
- [ ] Run enrichment on 5 stories with obscure people (verify org fallback used)
- [ ] Check enrichment_log contains expected audit trail
- [ ] Verify no false positives in sample of 10 matches
- [ ] Confirm rate limiting doesn't cause timeouts

---

## IMPLEMENTATION CHECKLIST

> **⛔ STRICT PHASE GATES:** Complete each phase and validate with metrics before proceeding.
> Violating phase gates leads to untested code in production.

### Phase 0: Baseline Measurement — Target: 1 week (MANDATORY FIRST)

| Priority | Item | File | Status | Success Criteria |
|----------|------|------|--------|------------------|
| P0 | Baseline metrics instrumentation | main.py | ⬜ | Logging deployed |
| P0 | Collect 1 week of baseline data | - | ⬜ | Metrics dashboard populated |
| P0 | Fill in "Current" values in metrics table | spec.md | ⬜ | TBD values replaced |
| P0 | Database reset decision confirmed | - | ⬜ | Stakeholder sign-off |

**⛔ GATE 0 CHECKPOINT:** Do NOT proceed until baseline metrics are collected.

---

### Phase 1: Core Extraction (MVP) — Target: 2 weeks

| Priority | Item | File | Status | Success Criteria |
|----------|------|------|--------|------------------|
| P0 | Exclusion criteria (wrong field) | profile_matcher.py | ⬜ | False positives <5% |
| P0 | Organization fallback | linkedin_profile_lookup.py | ⬜ | Zero-match rate <20% |
| P1 | enrichment_log field | database.py | ⬜ | Can debug failed matches |
| P1 | Unified pipeline class | company_mention_enricher.py | ⬜ | Single entry point |
| P1 | Feature flag for rollout | main.py | ⬜ | Can A/B test pipelines |
| P1 | Shadow mode comparison | main.py | ⬜ | Old vs new results logged |

**⛔ GATE 1 CHECKPOINT:** Match rate must be >= baseline. Error rate <5%. Run 1 week in shadow mode.

---

### Phase 2: Production Hardening — Target: 2 weeks after MVP

| Priority | Item | File | Status | Success Criteria |
|----------|------|------|--------|------------------|
| P0 | Common name thresholds | profile_matcher.py | ⬜ | "John Smith" problem fixed |
| P0 | Validation cache | company_mention_enricher.py | ⬜ | API cost reduced 60% |
| P1 | MVP validation (single query) | company_mention_enricher.py | ⬜ | Validation working |
| P1 | Indirect people discovery | find_leadership.py | ⬜ | Org leaders found |
| P2 | Profile verification for medium conf | profile_matcher.py | ⬜ | Borderline cases resolved |

**⛔ GATE 2 CHECKPOINT:** False positive rate <5%. High-confidence matches >50%. Run 2 weeks stable.

---

### Phase 3: Advanced Features — Only after Phase 2 stable in production for 2+ weeks

| Item | File | Status | Notes |
|------|------|--------|-------|
| Cross-story entity resolution | database.py | ⬜ | Defer until Phase 2 stable |
| Batch async processing | async_utils.py | ⬜ | Defer until Phase 2 stable |
| Dashboard metrics | dashboard.py | ⬜ | Defer until Phase 2 stable |
| Manual review workflow | dashboard.py | ⬜ | Defer until Phase 2 stable |

**⛔ DO NOT START Phase 3 until Phase 2 has been stable for at least 2 weeks.**

---

### Phase 4: Enterprise Scale — Future consideration (not scheduled)

| Item | Status | Notes |
|------|--------|-------|
| External data integration | ⬜ | Only if needed |
| A/B testing framework | ⬜ | Only if needed |
| Webhook notifications | ⬜ | Only if needed |

**Note:** Phase 4 items are documented for completeness but are NOT part of the current roadmap.

---

## APPENDIX: Codebase Integration Reference

### Existing Classes to Reuse

| Class | File | Purpose |
|-------|------|---------|
| `PersonEntity` | [ner_engine.py](ner_engine.py) | Extracted person data |
| `OrganizationEntity` | [ner_engine.py](ner_engine.py) | Extracted org data |
| `PersonSearchContext` | [linkedin_profile_lookup.py](linkedin_profile_lookup.py) | Search context |
| `ProfileMatcher` | [profile_matcher.py](profile_matcher.py) | Multi-signal matching |
| `MatchConfidence` | [profile_matcher.py](profile_matcher.py) | Confidence enum |
| `ProfileCandidate` | [profile_matcher.py](profile_matcher.py) | Candidate profiles |
| `LinkedInCompanyLookup` | [linkedin_profile_lookup.py](linkedin_profile_lookup.py) | LinkedIn searches |
| `RateLimitedAPIClient` | [api_client.py](api_client.py) | Rate-limited requests |
| `AdaptiveRateLimiter` | [rate_limiter.py](rate_limiter.py) | Dynamic rate limiting |
| `Story` | [database.py](database.py) | Story dataclass |
| `Database` | [database.py](database.py) | SQLite operations |

### Existing Constants to Reference

| Constant | File | Value |
|----------|------|-------|
| `WEIGHT_NAME_MATCH` | profile_matcher.py | 3.0 |
| `WEIGHT_ORG_MATCH` | profile_matcher.py | 3.0 |
| `WEIGHT_ROLE_MATCH` | profile_matcher.py | 2.0 |
| `WEIGHT_WRONG_FIELD` | profile_matcher.py | -5.0 |
| `HIGH_CONFIDENCE_THRESHOLD` | profile_matcher.py | 4.0 |
| `MEDIUM_CONFIDENCE_THRESHOLD` | profile_matcher.py | 2.0 |
| `COMPANY_ROLES` | find_leadership.py | CEO, CTO, CFO, etc. |
| `UNIVERSITY_ROLES` | find_leadership.py | President, Provost, etc. |
| `COMMON_FIRST_NAMES` | text_utils.py | Set of common names |
| `NICKNAME_MAP` | text_utils.py | Nickname mappings |

### Database Schema Changes Required

```python
# Updated Story dataclass in database.py (database will be reset):
@dataclass
class Story:
    # ... existing fields ...

    # Renamed for clarity (was story_people/org_leaders):
    direct_people: list[dict] = field(default_factory=list)    # Explicitly named in story
    indirect_people: list[dict] = field(default_factory=list)  # Discovered leadership

    # New enrichment metadata:
    enrichment_log: dict = field(default_factory=dict)
    enrichment_quality: str = ""  # "high", "medium", "low", "failed"
```

---

## APPENDIX: Edge Cases

### Common Names
**Problem:** "John Smith" at "Microsoft" matches hundreds of profiles

**Solution:** Use `is_common_name()` from [text_utils.py](text_utils.py) to require
additional attributes (location + title + specialty) for common names.

### Recent Job Changes
**Problem:** Article says "Professor at MIT" but LinkedIn shows "now at Stanford"

**Solution:** Accept recent job changes, check profile update date vs article date,
store note about potential change in `enrichment_log`.

### No LinkedIn Presence
**Problem:** Senior executive with no LinkedIn profile

**Solution:** Use organization fallback (`MatchConfidence.ORG_FALLBACK`), verify via
org website, store note in person record.

### Name Variations
**Problem:** Nicknames, maiden names, transliterations

**Solution:** Use `NICKNAME_MAP` from [text_utils.py](text_utils.py), store all
variations, search using multiple variants.

### Organizational Complexity
**Problem:** "MIT Media Lab" vs "MIT" matching

**Solution:** Normalize to parent org, use existing org normalization patterns,
accept LinkedIn showing either level.

---

## BENEFITS

This clean-slate workflow provides:

- **Deterministic** — Same input always produces same output
- **Auditable** — Full provenance trail for every data point
- **Safe automation** — No forced matches, clear confidence levels
- **Separation of concerns** — Each phase has distinct inputs/outputs/failure modes

---

## MIGRATION CHECKLIST

> **⛔ CONFIRMATION REQUIRED:** Database will be reset, losing all existing story data.
> This decision must be explicitly confirmed before proceeding.

### Database Reset Confirmation

- [ ] **Stakeholder approval obtained** for database reset
- [ ] **Backup created** of current `content_engine.db` (even if not migrating)
- [ ] **Test environment validated** with fresh database
- [ ] **Rollback plan documented** in case new schema has issues

**Alternative to Reset (if data preservation needed):**
```python
# Migration path if keeping existing data:
# 1. Add new columns with defaults (don't rename yet)
cursor.execute("ALTER TABLE stories ADD COLUMN direct_people TEXT DEFAULT '[]'")
cursor.execute("ALTER TABLE stories ADD COLUMN indirect_people TEXT DEFAULT '[]'")
cursor.execute("ALTER TABLE stories ADD COLUMN enrichment_log TEXT DEFAULT '{}'")

# 2. Copy data from old columns
cursor.execute("UPDATE stories SET direct_people = story_people")
cursor.execute("UPDATE stories SET indirect_people = org_leaders")

# 3. In a future release, drop old columns
```

> Use feature flags for incremental rollout of new pipeline logic.

### Code Changes Required

**1. Add Baseline Metrics (Do First)**

Before any other changes, instrument current code:

```python
# Add to main.py or company_mention_enricher.py
def log_enrichment_baseline(story: Story) -> None:
    """Log metrics to establish baseline before changes."""
    total = len(story.direct_people) + len(story.indirect_people)
    with_linkedin = sum(1 for p in story.direct_people + story.indirect_people
                        if p.get("linkedin_profile"))
    logger.info(f"BASELINE:enrichment story={story.id} "
                f"total={total} matched={with_linkedin}")
```

**2. New Class: `EntityEnrichmentPipeline`**

Create in [company_mention_enricher.py](company_mention_enricher.py):

```python
class EntityEnrichmentPipeline:
    def enrich_story(self, story: Story, dry_run: bool = False) -> EnrichmentResult:
        """dry_run=True for shadow mode comparison."""
        ...
    def enrich_pending_stories(self, dry_run: bool = False) -> tuple[int, int]: ...
```

**3. Extend `PersonEntity` in [ner_engine.py](ner_engine.py)**

Add these fields to the existing dataclass:

```python
@dataclass
class PersonEntity(Entity):
    # Existing fields:
    title: str = ""
    affiliation: str = ""
    linkedin_profile: str = ""
    linkedin_urn: str = ""

    # NEW fields to add:
    location: str = ""                    # City, state, country
    specialty: str = ""                   # Research area, department
    role_in_story: str = ""               # primary_subject, quoted, mentioned, author
    is_direct: bool = True                # True = extracted, False = discovered
    validation_status: str = "pending"    # pending, verified, failed
    validation_confidence: float = 0.0
    validation_source: str = ""           # URL of authoritative source
```

**4. Update `Story` in [database.py](database.py)**

Rename fields and add metadata (database reset allows clean schema):

```python
# Updated Story dataclass with clean field names:
direct_people: list[dict] = field(default_factory=list)    # was: story_people
indirect_people: list[dict] = field(default_factory=list)  # was: org_leaders
enrichment_log: dict = field(default_factory=dict)
enrichment_quality: str = ""
```

**5. Update `main.py` with Feature Flag**

Use environment variable for rollout control:

```python
import os
USE_NEW_PIPELINE = os.environ.get("USE_NEW_ENRICHMENT", "false")

if USE_NEW_PIPELINE == "shadow":
    # Run both, compare results
    _run_org_leaders_enrichment_silent(engine)
    _run_profile_lookup_silent(engine)
    new_pipeline = EntityEnrichmentPipeline(engine.db, engine.linkedin_lookup)
    new_pipeline.enrich_pending_stories(dry_run=True)
elif USE_NEW_PIPELINE == "true":
    # New pipeline only
    pipeline = EntityEnrichmentPipeline(engine.db, engine.linkedin_lookup)
    enriched, skipped = pipeline.enrich_pending_stories()
else:
    # Old behavior (default)
    _run_org_leaders_enrichment_silent(engine)
    _run_profile_lookup_silent(engine)
    _run_urn_extraction_silent(engine)
    _mark_stories_enriched(engine)
```

**6. Database Schema Creation**

Fresh schema with new field names (no migration needed):

```python
# In Database.__init__() or _create_tables():
cursor.execute("""
    CREATE TABLE IF NOT EXISTS stories (
        -- ... other columns ...
        direct_people TEXT DEFAULT '[]',
        indirect_people TEXT DEFAULT '[]',
        enrichment_log TEXT DEFAULT '{}',
        enrichment_quality TEXT DEFAULT ''
    )
""")
```

### Files to Modify

| File | Changes | Risk Level |
|------|---------|------------|
| main.py | Add baseline logging + feature flag | Low |
| database.py | Add 2 new fields to `Story` | Low |
| ner_engine.py | Add 7 new fields to `PersonEntity` | Low |
| profile_matcher.py | Add exclusion criteria + common name thresholds | Medium |
| company_mention_enricher.py | Add `EntityEnrichmentPipeline` class | Medium |
| linkedin_profile_lookup.py | Add org fallback logic | Medium |

### Rollout Sequence

| Week | Action | Validation |
|------|--------|------------|
| 1 | Deploy baseline metrics | Collect 1 week of data |
| 2 | Deploy new pipeline in shadow mode | Compare old vs new results |
| 3 | Route 20% traffic to new pipeline | Monitor error rates |
| 4 | Route 50% traffic | Monitor match rates |
| 5 | Full cutover if metrics pass | Keep old code for rollback |
| 6+ | Remove old code | After 2 weeks stable |

---

## CONCLUSION

This specification defines a **4-phase** entity extraction and LinkedIn matching pipeline:

| Phase | Name | Key Output |
|-------|------|------------|
| 1 | Canonical Story Context | Immutable text + extracted entities |
| 2 | Validation & Discovery | Verified direct + discovered indirect people |
| 3 | LinkedIn Profile Matching | LinkedIn URLs with confidence scores |
| 4 | Database Persistence | Updated Story record with full audit trail |

**Key integration points:**

1. **Orchestration**: New `EntityEnrichmentPipeline` class replaces fragmented calls
2. **NER**: Extend existing `PersonEntity`/`OrganizationEntity` dataclasses
3. **LinkedIn Search**: Use existing `LinkedInCompanyLookup` + `ProfileMatcher`
4. **Rate Limiting**: Use existing `AdaptiveRateLimiter` patterns
5. **Database**: Store in existing `Story` fields with proposed extensions

**Key improvements over current approach:**

- Entity extraction integrated with story search (single read, no re-fetch)
- Validation prompt provides auditable credential verification
- Unified pipeline replaces 4+ separate enrichment functions
- Clear confidence thresholds determine when to use org fallback


