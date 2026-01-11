# Social Media Publisher - Improvement Tasks

> **Project:** https://github.com/waynegault/SocialMediaPublisher
> **Last Updated:** January 11, 2026
> **Test Status:** 415 tests passing ✅

---

## 📋 Outstanding Tasks (12 remaining)

### Priority 1: Critical Mission Improvements

#### TASK 1.1: RAG for Personalized Content Generation
**Complexity:** High | **Files:** New `rag_engine.py`

Create a Retrieval-Augmented Generation engine to personalize content based on the author's experience.

- [ ] Index author's resume, projects, publications, and target job descriptions
- [ ] Use vector embeddings (ChromaDB) for context retrieval
- [ ] Inject personal context into summary generation prompts
- [ ] Enable prompts like: "Connect this story to my experience at [Company X]"

---

#### TASK 1.3: Intent Recognition for Job-Relevant Story Filtering
**Complexity:** Medium | **Files:** New `intent_classifier.py`

Ensure content aligns with career goals through NLP-based intent classification.

- [ ] Implement DistilBERT-based intent classifier
- [ ] Create "career alignment score" for each story
- [ ] Classify stories: skill_showcase, network_building, thought_leadership, industry_awareness
- [ ] Filter stories that could position author negatively

---

#### TASK 1.4: Entity Extraction & Named Entity Recognition
**Complexity:** Medium | **Files:** New `ner_engine.py`

Improve people/company identification for @mentions using spaCy NER.

- [ ] Implement spaCy NER with custom domain patterns
- [ ] Create entity disambiguation using knowledge graphs
- [ ] Build company/person cache with fuzzy matching
- [ ] Add chemical engineering domain-specific rules

---

### Priority 2: LinkedIn API & Automation

#### TASK 2.1: Robust LinkedIn Profile URN Resolution
**Complexity:** Medium | **Files:** Modify `linkedin_profile_lookup.py`

Make @mentions more reliable with improved URN extraction.

- [ ] Integrate Playwright for browser automation
- [ ] Add browser cookie persistence for sessions
- [ ] Implement rate-limited concurrent lookups
- [ ] Create fallback chain: Cache → API → Playwright → Google Search

---

#### TASK 2.2: LinkedIn Connection Request Automation
**Complexity:** Medium | **Files:** New `linkedin_networking.py`

Build network with story-relevant people automatically.

- [ ] Send personalized connection requests to mentioned people
- [ ] Implement "warm intro" message templates
- [ ] Add rate limiting (100 requests/week)
- [ ] Track connection acceptance rates

---

#### TASK 2.3: Comment Engagement Automation
**Complexity:** Medium | **Files:** New `linkedin_engagement.py`

Increase visibility through strategic engagement.

- [ ] Monitor posts from target companies/people
- [ ] Generate thoughtful AI comments on relevant posts
- [ ] Implement engagement scheduling
- [ ] Add spam detection safeguards

---

#### TASK 2.4: LinkedIn Analytics Integration
**Complexity:** Medium | **Files:** New `analytics_engine.py`

Enable data-driven content optimization.

- [ ] Implement LinkedIn Analytics API integration
- [ ] Track impressions, engagement rate, profile views
- [ ] Correlate content types with performance
- [ ] Auto-adjust content strategy based on analytics

---

### Priority 4: Architecture

#### TASK 4.1: Async/Await Pattern Throughout
**Complexity:** High | **Files:** Most core files

Convert to asyncio for better performance.

- [ ] Convert I/O operations to async
- [ ] Use aiohttp for HTTP requests
- [ ] Implement concurrent story processing
- [ ] Maintain backward compatibility with sync wrappers

---

### Priority 6: Image Generation

#### TASK 6.3: Multi-Model Image Generation
**Complexity:** Medium | **Files:** Modify `image_generator.py`

Improve image quality through model selection.

- [ ] Add DALL-E 3 as alternative generator
- [ ] Implement model selection based on prompt type
- [ ] Add A/B testing for image models
- [ ] Auto-fallback on model failures

---

### Priority 7: Testing

#### TASK 7.3: Load Testing
**Complexity:** Medium | **Files:** New `load_tests.py`

Ensure scalability under load.

- [ ] Test rate limiter under load
- [ ] Test database performance
- [ ] Test concurrent story processing
- [ ] Identify bottlenecks

---

### Priority 8: User Experience

#### TASK 8.1: Enhance Human Validation GUI
**Complexity:** Medium | **Files:** Modify `validation_server.py`

Make review workflow more efficient.

- [ ] Add keyboard shortcuts for approve/reject
- [ ] Implement batch operations
- [ ] Add inline editing
- [ ] Mobile-responsive design

---

#### TASK 8.4: Web Dashboard
**Complexity:** High | **Files:** New `dashboard/`

Provide real-time operational visibility.

- [ ] Real-time pipeline status
- [ ] Analytics visualization
- [ ] Story queue management
- [ ] Historical performance charts

---

## ✅ Completed Tasks (23 total)

### Quick Wins (5)
- **QW.1**: Retry Logic for LinkedIn URN Extraction
- **QW.2**: Post Preview Before Publishing
- **QW.3**: Story Quality Score Calibration
- **QW.4**: Automatic Source URL Archiving
- **QW.5**: Duplicate Detection Across Stories

### Technical Debt (4)
- **TD.2**: Consolidated retry logic
- **TD.3**: Configurable constants
- **TD.4**: Type hints for core modules
- **TD.5**: Improved docstrings

### Architecture (4)
- **TASK 4.2**: Dependency Injection (`di.py`)
- **TASK 4.3**: Event-Driven Architecture (`events.py`)
- **TASK 4.4**: Logging & Monitoring (`monitoring.py`)
- **TASK 4.5**: Pydantic Config Validation (`config.py`)

### Content Quality (4)
- **TASK 1.2**: LinkedIn Algorithm Optimization (`linkedin_optimizer.py`)
- **TASK 3.1**: Multi-Source Verification (`source_verifier.py`)
- **TASK 3.2**: Trend Detection (`trend_detector.py`)
- **TASK 3.3**: A/B Testing (`ab_testing.py`)
- **TASK 3.4**: Originality Checking (`originality_checker.py`)

### Performance (3)
- **TASK 5.1**: Database Migrations (`migrations.py`)
- **TASK 5.2**: Analytics Data Warehouse (`analytics_db.py`)
- **TASK 5.3**: Caching Layer (`cache.py`)

### Image Generation (2)
- **TASK 6.1**: Image Quality Assessment (`image_quality.py`)
- **TASK 6.2**: Image Style Consistency (`image_style.py`)

### Testing (2)
- **TASK 7.1**: Integration Tests (`integration_tests.py`)
- **TASK 7.2**: Property-Based Testing (`property_tests.py`)

### User Experience (2)
- **TASK 8.2**: Notifications (`notifications.py`)
- **TASK 8.3**: CLI Improvements (`cli.py`)

---

## Recommended Next Steps

**High Impact, Medium Complexity:**
1. **TASK 1.4** - Entity Extraction (NER) - Improves @mention reliability
2. **TASK 1.3** - Intent Recognition - Better story filtering
3. **TASK 2.4** - LinkedIn Analytics - Data-driven optimization

**High Impact, High Complexity:**
1. **TASK 1.1** - RAG Engine - Personalized content generation
2. **TASK 4.1** - Async/Await - Performance improvement

---

## Dependencies to Add (for remaining tasks)

```
# NLP & Entity Recognition
spacy>=3.7.0
sentence-transformers>=2.2.0

# RAG
chromadb>=0.4.0

# Async
aiohttp>=3.9.0

# Load Testing
locust>=2.0.0
```
