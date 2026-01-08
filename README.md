# Social Media Publisher

An automated Python application that discovers news stories, generates AI images,
verifies content quality, and publishes to LinkedIn on a configurable schedule.

## Overview

Social Media Publisher automates the entire content curation and publishing workflow:
1. Discovers relevant news stories using AI-powered search
2. Generates professional images for each story
3. Verifies content quality before publication
4. Schedules and publishes to LinkedIn automatically

The system supports both Google Gemini AI (with Search grounding) and local LLMs
via LM Studio for flexible deployment options.

For detailed architecture and component documentation, see the [Overview wiki](https://github.com/waynegault/SocialMediaPublisher/wiki/Overview).

## Features

- **Automated Story Discovery**: Searches the internet for news stories matching
  your criteria using Google's Gemini AI with Search grounding or local LLMs
- **Semantic Deduplication**: Uses Jaccard similarity to detect duplicate stories
  even when titles differ slightly
- **Quality Scoring with Justification**: Rates stories 1-10 with explanations
  for each score
- **Category Classification**: Automatically categorizes stories (Technology,
  Business, Science, AI, Other)
- **AI Image Generation**: Creates professional illustrations using Google's
  Imagen model or Hugging Face Inference API
- **Content Verification**: Uses a separate AI pass to verify professionalism,
  decency, and adherence to your criteria
- **Smart Scheduling**: Publishes stories spread evenly across your preferred
  hours with configurable jitter
- **LinkedIn Integration**: Posts stories with images, source links, hashtags,
  @ mentions, and your signature block
- **LinkedIn Analytics**: Track impressions, likes, comments, shares for posts
- **Smart Hashtags**: AI-generated relevant hashtags (up to 3 per post)
- **LinkedIn Mentions**: Automatic @ mentions for companies/people in stories
- **Opportunity Messages**: Optional professional postscript about job openings
- **Automatic Cleanup**: Removes old unused stories after a configurable
  exclusion period
- **Retry Logic**: Automatic retry with exponential backoff for transient API
  failures
- **URL Validation**: Validates source URLs before saving stories
- **Date Post-Filtering**: Filters stories by actual article publication date
- **Preview Mode**: Preview discovered stories before saving with selective save
  capability

## How It Works

1. **Search Cycle**: Discovers new stories based on your prompt and date range
2. **Image Generation**: Creates images for stories that meet quality threshold
3. **Verification**: AI reviews each story+image for quality before publication
4. **Scheduling**: Approved stories are scheduled within your publishing window
5. **Publishing**: Stories are published to LinkedIn at their scheduled times
6. **Cleanup**: Old unpublished stories are removed after the exclusion period

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/waynegault/SocialMediaPublisher.git
   cd SocialMediaPublisher
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## Configuration

Copy `.env.example` to `.env` and configure the following:

### Required API Keys

| Variable | Description |
| :--- | :--- |
| `GEMINI_API_KEY` | Google AI API key from [AI Studio](https://aistudio.google.com/) |
| `LINKEDIN_ACCESS_TOKEN` | OAuth token from [LinkedIn Developer](https://developer.linkedin.com/) |
| `LINKEDIN_AUTHOR_URN` | Your LinkedIn URN (e.g., `urn:li:person:ABC123`) |

### Optional: Hugging Face Image Generation

If you have a Hugging Face token, the app can generate images via the Inference API. When a token is configured, the app will prefer Hugging Face for image generation.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HUGGINGFACE_API_TOKEN` or `HUGGINGFACE_API_KEY` | - | Your HF access token |
| `HF_TTI_MODEL` | stabilityai/stable-diffusion-xl-base-1.0 | Text-to-image model id |
| `HF_INFERENCE_ENDPOINT` | - | Custom endpoint URL (optional) |
| `HF_PREFER_IF_CONFIGURED` | True | Prefer HF when token present |
| `HF_NEGATIVE_PROMPT` | text, watermark, logo, blurry... | Negative prompt used |

### Image Style Settings

| Variable | Default | Description |
| :--- | :--- | :--- |
| `IMAGE_STYLE` | photorealistic, 1960s Kodachrome... | Style directives for images |

The default style creates photorealistic images with a vintage 1960s cinema aesthetic featuring warm Kodachrome tones, soft golden lighting, subtle film grain, and muted pastel highlights. Customize this to change the visual style of all generated images.

### Image Prompt Engineering

The system uses professional prompt engineering techniques based on Google's
official Imagen prompt guide. When generating images, an LLM refines the story
context into an optimized prompt following these principles:

**Prompt Structure** (Subject → Context → Style → Technical):

- Start with the main subject and action
- Add environment and setting details
- Include lighting and atmosphere
- End with camera/technical specifications

**Photography Modifiers Used**:

- **Camera/Lens**: "professional DSLR", "35mm lens", "wide-angle", "macro"
- **Quality**: "4K", "high detail", "sharp focus", "HDR"
- **Lighting**: "golden hour", "soft natural daylight", "dramatic side lighting"
- **Composition**: "rule of thirds", "leading lines", "depth of field"

**Best Practices Applied**:

- Descriptive, specific language (not vague terms)
- Concrete visual elements from the story
- Avoids faces (uses silhouettes/backs instead)
- No text, logos, or watermarks in prompts
- Maximum 150 words for optimal results

**Customizing Image Style**:

Set `IMAGE_STYLE` to override the default aesthetic. Examples:

- Modern: `"clean modern photography, bright natural lighting, minimalist"`
- Dramatic: `"dramatic noir photography, high contrast, deep shadows"`
- Editorial: `"warm editorial photography, soft golden tones, magazine style"`

### Search Settings

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SEARCH_PROMPT` | - | The prompt describing what stories to search for |
| `SEARCH_PROMPT_TEMPLATE` | - | Full instruction template with placeholders |
| `SEARCH_LOOKBACK_DAYS` | 7 | Days to look back when searching |
| `USE_LAST_CHECKED_DATE` | True | Use last check time instead of lookback |
| `SEARCH_CYCLE_HOURS` | 24 | How often to run search cycles |
| `MAX_STORIES_PER_SEARCH` | 5 | Maximum stories to request per search |

### Content Settings

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SUMMARY_WORD_COUNT` | 250 | Target word count for summaries |
| `MIN_QUALITY_SCORE` | 7 | Minimum score (1-10) for publication |

### Deduplication Settings

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DEDUP_SIMILARITY_THRESHOLD` | 0.6 | Jaccard similarity threshold (0.0-1.0) |

### URL Validation Settings

| Variable | Default | Description |
| :--- | :--- | :--- |
| `VALIDATE_SOURCE_URLS` | False | Enable URL validation before saving |

### API Retry Settings

| Variable | Default | Description |
| :--- | :--- | :--- |
| `API_RETRY_COUNT` | 3 | Number of retry attempts for API calls |
| `API_RETRY_DELAY` | 1.0 | Base delay in seconds (doubles each retry) |

### Publication Settings

| Variable | Default | Description |
| :--- | :--- | :--- |
| `STORIES_PER_CYCLE` | 3 | Maximum stories to publish per cycle |
| `PUBLISH_WINDOW_HOURS` | 24 | Window to spread publications over |
| `PUBLISH_START_HOUR` | 8 | Earliest hour to publish (0-23) |
| `PUBLISH_END_HOUR` | 20 | Latest hour to publish (0-23) |
| `JITTER_MINUTES` | 30 | Random variance in publish time (+/-) |

### Cleanup Settings

| Variable | Default | Description |
| :--- | :--- | :--- |
| `EXCLUSION_PERIOD_DAYS` | 30 | Days before unused stories are deleted |

### Signature Block

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SIGNATURE_BLOCK` | - | Text/hashtags appended to each post |

## Usage

### Interactive Menu (Default)

Run the application with an interactive menu for testing and manual operations:

```bash
python main.py
```

### Continuous Mode

Run the application continuously, checking for due publications every minute:

```bash
python main.py --continuous
```

### Single Cycle

Run one search cycle and exit:

```bash
python main.py --once
```

### Check Status

View current configuration and statistics:

```bash
python main.py --status
```

### Publish Now

Publish any due stories immediately:

```bash
python main.py --publish-now
```

### View Configuration

Display current configuration:

```bash
python main.py --config
```

## Project Structure

```text
SocialMediaPublisher/
├── main.py                  # Main entry point and orchestration
├── config.py                # Configuration management
├── database.py              # SQLite database operations
├── searcher.py              # Story discovery using Gemini
├── image_generator.py       # AI image generation (Imagen/HF)
├── verifier.py              # Content quality verification
├── scheduler.py             # Publication timing logic
├── linkedin_publisher.py    # LinkedIn API integration
├── error_handling.py        # Circuit breaker and retry logic
├── rate_limiter.py          # Adaptive rate limiting
├── test_framework.py        # Custom unit test framework
├── unit_tests.py            # Unit test suite
├── linkedin_test.py         # LinkedIn connection test utility
├── linkedin_diagnostics.py  # LinkedIn token diagnostics
├── linkedin_dryrun_publish.py # Dry-run publish preview
├── publish_story.py         # Manual story publish utility
├── requirements.txt         # Python dependencies
├── .env.example             # Example environment configuration
├── generated_images/        # AI-generated story images
└── README.md                # This file
```

## Database Schema

Stories are stored in SQLite with the following fields:

- `id`: Unique identifier
- `title`: Story title
- `summary`: Generated summary
- `source_links`: JSON array of source URLs
- `acquire_date`: When the story was discovered
- `quality_score`: AI-assigned quality rating (1-10)
- `category`: Story category (Technology, Business, Science, AI, Other)
- `quality_justification`: Explanation for the quality score
- `image_path`: Path to generated image
- `verification_status`: pending/approved/rejected
- `publish_status`: unpublished/scheduled/published
- `scheduled_time`: When to publish
- `published_time`: When actually published
- `linkedin_post_id`: LinkedIn's post identifier

## Workflow Details

### Story Selection Process

1. Gemini AI searches for stories matching your `SEARCH_PROMPT`
2. Stories are grouped by topic (multiple sources = one story)
3. Each story gets a quality score based on:
   - Relevance to search criteria (40%)
   - Significance and newsworthiness (30%)
   - Source credibility (20%)
   - Timeliness (10%)

### Verification Process

Before publication, a separate AI review checks:

- Adherence to original search criteria
- Professional tone
- Appropriateness for all audiences
- Factual accuracy of summary

### Scheduling Logic

Stories are scheduled to:

- Only publish between `PUBLISH_START_HOUR` and `PUBLISH_END_HOUR`
- Spread evenly across the `PUBLISH_WINDOW_HOURS`
- Have random jitter of +/- `JITTER_MINUTES`

### Backfill Behavior

If fewer than `STORIES_PER_CYCLE` new stories are found:

- Previously approved but unpublished stories are used to fill the gap
- Stories are always selected by quality score (highest first)

## Troubleshooting

### No stories found

- Check your `SEARCH_PROMPT` is specific but not too narrow
- Verify `SEARCH_LOOKBACK_DAYS` covers recent news
- Ensure your Gemini API key has search grounding enabled
- The system provides actionable feedback when no results are found

### Images not generating

- Verify your API key has Imagen access
- Check `MIN_QUALITY_SCORE` isn't too high

### LinkedIn posting fails

- Verify `LINKEDIN_ACCESS_TOKEN` is valid and not expired
- Check `LINKEDIN_AUTHOR_URN` format
- Ensure token has `w_member_social` permission
- For identity checks use the OpenID endpoint `/v2/userinfo` (do **not** call `/v2/me` — it's legacy and may not return OpenID-compliant claims).

### Duplicate stories appearing

- Adjust `DEDUP_SIMILARITY_THRESHOLD` (higher = stricter matching)
- Default of 0.6 catches most duplicates while allowing related stories

---

## Developer Instructions

This section provides detailed technical information for developers working on
or extending the Social Media Publisher.

### Architecture Overview

The application follows a modular architecture:

1. **config.py**: Centralized configuration from environment variables
2. **database.py**: SQLite persistence with Story dataclass
3. **searcher.py**: Story discovery with multiple backends (Gemini, Local LLM)
4. **image_generator.py**: Image generation (Imagen, Hugging Face)
5. **verifier.py**: Content quality verification
6. **scheduler.py**: Publication timing logic
7. **linkedin_publisher.py**: LinkedIn API integration
8. **main.py**: Orchestration and CLI

### Key Implementation Details

#### Semantic Deduplication

The `calculate_similarity()` function in `searcher.py` uses Jaccard similarity:

- Normalizes text to lowercase, removes punctuation
- Filters common stopwords
- Calculates intersection/union of word sets
- Threshold configurable via `DEDUP_SIMILARITY_THRESHOLD`

#### Retry Logic

The `retry_with_backoff()` decorator provides:

- Configurable retry count and base delay
- Exponential backoff (delay doubles each attempt)
- Configurable exception types to catch
- Logging of retry attempts

#### URL Validation

When `VALIDATE_SOURCE_URLS=True`:

- Validates URL format (scheme, netloc)
- Optionally performs HEAD request to check accessibility
- Filters invalid URLs before saving stories

#### Date Post-Filtering

The `filter_stories_by_date()` function:

- Extracts dates from multiple common field names
- Supports various date formats (ISO, human-readable)
- Stories without parseable dates are included (benefit of doubt)

#### Preview Mode

The `search_preview()` method:

- Searches without saving to database
- Caches results in `_preview_stories`
- `save_selected_stories(indices)` saves specific stories
- `save_all_preview_stories()` saves all previewed stories

#### JSON Repair

When initial JSON parsing fails:

1. Tries markdown code block extraction
2. Tries outermost brace/bracket extraction
3. Tries regex-based list extraction
4. Attempts to salvage individual story objects
5. Falls back to LLM-based JSON repair

### Environment Variables

All configuration is loaded at import time from environment variables.
The `Config` class uses `@staticmethod` properties for lazy evaluation
where needed.

#### Timeout and Search Settings

These settings control API timeouts and search behavior:

- `API_TIMEOUT_DEFAULT` (30): Default timeout for cloud API calls in seconds
- `LLM_LOCAL_TIMEOUT` (120): Timeout for local LLM calls (LM Studio) in seconds - local models often need longer processing time
- `DUCKDUCKGO_MAX_RESULTS` (10): Maximum number of results to fetch from DuckDuckGo searches
- `LLM_MAX_OUTPUT_TOKENS` (8192): Maximum output tokens for LLM responses
- `IMAGE_ASPECT_RATIO` (16:9): Aspect ratio for generated images (options: 1:1, 16:9, 9:16, 4:3, 3:4)

### Database Migrations

New columns are added with `ALTER TABLE` statements that use
`IF NOT EXISTS` logic (checking sqlite_master). This allows seamless
upgrades without data loss.

### API Endpoints Used

#### Google Gemini API

- **Search**: `models.generate_content()` with `google_search` tool
- **Verification**: `models.generate_content()` for content review
- **JSON Repair**: `models.generate_content()` with JSON response type

#### DuckDuckGo (via ddgs library)

- **News Search**: `ddgs.news()` for recent news articles
- **Text Search**: `ddgs.text()` as fallback

#### LinkedIn API

- **Post Creation**: `POST /ugcPosts` with image and text
- **Image Upload**: Multi-step upload process
- **Analytics**: `GET /organizationalEntityShareStatistics` for org posts,
  `GET /socialActions/{urn}` for personal posts

#### LinkedIn Analytics

The system tracks post performance metrics for published stories:

- **Impressions**: Number of times the post was viewed
- **Clicks**: Number of clicks on the post
- **Likes**: Number of likes received
- **Comments**: Number of comments received
- **Shares**: Number of times the post was shared
- **Engagement Rate**: (likes + comments + shares) / impressions

Use menu options 22 and 23 to view and refresh analytics:
- **Option 22**: View LinkedIn Analytics - displays a table of all published
  stories with their performance metrics
- **Option 23**: Refresh All Analytics - fetches latest metrics from LinkedIn
  API for all published stories

Note: Full analytics (impressions, clicks, shares) require organization admin
permissions. Personal posts only provide likes and comments via the
socialActions endpoint.

### Utility Scripts

The project includes several utility scripts for testing and debugging:

#### linkedin_test.py

Quick LinkedIn connection test:

```bash
python linkedin_test.py              # Test token from .env
python linkedin_test.py --prompt     # Enter token interactively
python linkedin_test.py --save       # Save working token to .env
```

#### linkedin_diagnostics.py

Comprehensive LinkedIn token diagnostics:

```bash
python linkedin_diagnostics.py       # Full diagnostic report
```

#### linkedin_dryrun_publish.py

Preview what would be published without making API calls:

```bash
python linkedin_dryrun_publish.py    # Shows payloads for due stories
```

#### publish_story.py

Manually publish a specific story:

```bash
python publish_story.py <story_id>   # Publish story by ID
```

#### unit_tests.py

Run the unit test suite:

```bash
python unit_tests.py                 # Run all tests
```

### Error Handling and Resilience

The `error_handling.py` module provides production-grade resilience:

#### Circuit Breaker Pattern

Prevents cascading failures by tracking service health:

- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Service unhealthy, requests fail fast
- **HALF-OPEN**: Testing if service has recovered

Configuration via `CircuitBreakerConfig`:

- `failure_threshold`: Failures before opening (default: 5)
- `recovery_timeout`: Seconds before trying half-open (default: 30)
- `success_threshold`: Successes needed to close (default: 2)

#### Enhanced Recovery Decorator

`@with_enhanced_recovery()` provides:

- Exponential backoff with jitter
- Configurable max attempts and delays
- Recovery context tracking
- Automatic retry for transient failures

### Rate Limiting

The `rate_limiter.py` module implements adaptive rate limiting:

- Token bucket algorithm for smooth request pacing
- Automatic rate reduction on 429 errors
- Gradual rate increase after consecutive successes
- Per-endpoint tracking and metrics

### Gotchas and Known Issues

1. **Gemini Rate Limits**: The 429 RESOURCE_EXHAUSTED error requires
   waiting or upgrading quota. The rate limiter handles this automatically.
2. **LinkedIn Token Expiry**: OAuth tokens expire after 60 days.
   Use `linkedin_diagnostics.py` to check token validity.
3. **Image Generation Failures**: If Imagen fails, check your Gemini API
   quota. The system will skip image generation gracefully.
4. **Database Locking**: SQLite may lock briefly during writes. The
   connection pool handles this with timeouts.

### Testing

Run the custom unit test framework:

```bash
python unit_tests.py
```

The test framework (`test_framework.py`) provides:

- Isolated test execution with error catching
- Logging suppression for clean output
- Detailed pass/fail reporting
- MockLogger for testing logged output

You can also run tests from the interactive menu using option **15**.

### Database Backup and Recovery

The database (`content_engine.db`) is tracked in git for version control.
A local backup file (`*.db.backup`) is excluded from git for recovery.

From the interactive menu:

- **16. Backup Database**: Create a backup of the current database
- **17. Verify Database Integrity**: Run SQLite integrity check
- **18. Restore Database from Backup**: Restore from the backup file

---

## Future Developer Ideas

1. **Multi-Platform Publishing**: Add support for Twitter/X, Facebook,
   Instagram
2. **Story Clustering**: Group related stories into threads
3. **Sentiment Analysis**: Add sentiment scoring to stories
4. **A/B Testing**: Test different summary styles for engagement
5. **Analytics Dashboard**: Track post performance metrics
6. **Webhook Notifications**: Alert on publish success/failure
7. **Story Templates**: Customizable post formats per category
8. **Batch Processing**: Process multiple search prompts in parallel
9. **Content Calendar**: Visual scheduling interface
10. **Source Reputation Scoring**: Track and weight source reliability

---

## Appendix A: Chronology of Changes

### Version 2.8 (January 8, 2026)

- **Fixed source_links bug**: Stories no longer saved with empty sources
  - Added early exit check before URL validation
  - Ensures every saved story has at least one valid source URL
- **Enhanced image generation for more attractive women**:
  - Updated IMAGE_STYLE to emphasize glamour and model-quality beauty
  - Updated IMAGE_REFINEMENT_PROMPT with detailed beauty requirements
  - Updated IMAGE_FALLBACK_PROMPT for consistency
  - Women now featured with flawless skin, styled hair, fashion-forward workwear
- **Codebase cleanup**: Removed 10 redundant/unused files
  - Removed: linkedin_diagnostics.py, linkedin_test.py, test_linkedin_connection.py
  - Removed: publish_story.py, linkedin_dryrun_publish.py, linkedin_mentions.py
  - Removed: scripts/ folder, archive/ folder
- Added COMPANY_MENTION_PROMPT to config.py (was missing)
- Fixed unused import in company_mention_enricher.py

### Version 2.7 (January 8, 2026)

- Added LinkedIn Analytics tracking for published posts
- New Story fields for analytics: impressions, clicks, likes, comments, shares
- New menu options: View LinkedIn Analytics (22), Refresh All Analytics (23)
- Moved ALL LLM prompts to environment configuration
- New configurable prompts: SEARCH_DISTILL_PROMPT, LOCAL_LLM_SEARCH_PROMPT
- New configurable prompts: LINKEDIN_MENTION_PROMPT, JSON_REPAIR_PROMPT
- All prompts now use Config class with .env defaults
- Updated .env.example with full prompt documentation

### Version 2.6 (January 8, 2026)

- Added AI-generated hashtags to stories (up to 3 per post)
- Added LinkedIn @ mentions for companies/people in stories
- Created linkedin_mentions.py for URN lookup
- Added opportunity_messages.py with 50 professional postscripts
- New INCLUDE_OPPORTUNITY_MESSAGE config setting
- Updated Story dataclass with hashtags and linkedin_mentions fields
- Enhanced _format_post_text to include new elements

### Version 2.5 (January 8, 2026)

- Improved URL-to-Story Matching with semantic scoring
- New extract_url_keywords function for URL path analysis
- New calculate_url_story_match_score for match confidence
- Removed blind positional URL assignment that caused mix-ups
- Added URL validation for local LLM path
- Unmatched URLs now logged as warnings

### Version 2.4 (January 8, 2026)

- Added configurable timeout and search settings to .env
- New settings: API_TIMEOUT_DEFAULT, LLM_LOCAL_TIMEOUT, DUCKDUCKGO_MAX_RESULTS
- New settings: LLM_MAX_OUTPUT_TOKENS, IMAGE_ASPECT_RATIO
- Moved hardcoded values from searcher.py and image_generator.py to config

### Version 2.3 (January 7, 2026)

- Fixed LLM hallucinating source URLs - now extracts real URLs from grounding metadata
- Improved URL validation to reject 404 errors and connection failures
- Updated prompts to explicitly require using only real URLs from search results

### Version 2.2 (January 7, 2026)

- Added "Run Unit Tests" option to interactive menu (option 15)
- Added database backup/restore/verify menu options (16, 17, 18)
- Database now tracked in git with backup excluded
- Fixed all pylance errors and type warnings
- Improved null-safety in LinkedIn authentication code

### Version 2.1 (January 7, 2026)

- Added production-grade error handling with circuit breaker pattern
- Implemented adaptive rate limiting with token bucket algorithm
- Added custom unit test framework with isolated test execution
- Created comprehensive utility scripts for LinkedIn debugging
- Enhanced configuration validation and environment variable handling
- Sanitized `.env.example` to remove sensitive data
- Updated documentation with developer-focused sections

### Version 2.0 (January 2026)

- Added configurable story count per search (`MAX_STORIES_PER_SEARCH`)
- Implemented semantic deduplication with Jaccard similarity
- Added category classification and persistence
- Added quality score justification field
- Implemented retry logic with exponential backoff
- Added source URL validation
- Added date post-filtering for stories
- Added progress indication callbacks
- Implemented intermediate result caching
- Added LLM-based JSON repair fallback
- Added story preview mode with selective saving
- Improved empty results feedback

### Version 1.0 (Initial Release)

- Core story discovery with Gemini AI
- Image generation with Imagen/Hugging Face
- Content verification
- LinkedIn publishing
- Automatic scheduling
- Database persistence

---

## Appendix B: Technical Specifications

### System Requirements

- Python 3.10+
- SQLite 3.x
- Internet connection for API access

### API Requirements

| Service | Required | Purpose |
| :--- | :--- | :--- |
| Google Gemini API | Yes* | Story search, verification |
| LM Studio | Yes* | Alternative to Gemini |
| Google Imagen | Optional | Image generation |
| Hugging Face | Optional | Alternative image generation |
| LinkedIn API | Yes | Publishing |

*Either Gemini or LM Studio is required

### Database Specifications

- **Engine**: SQLite 3
- **File**: `content_engine.db` (configurable via `DB_NAME`)
- **Tables**: `stories`, `state`
- **Indexes**: Primary key on `id`
- **Connection**: Thread-safe with connection pooling

### Rate Limits

| API | Limit | Notes |
| :--- | :--- | :--- |
| Gemini | Varies by tier | Free tier: 60 RPM |
| DuckDuckGo | Unofficial | May block aggressive use |
| LinkedIn | 100 posts/day | Per application |

### Performance Characteristics

- **Search Latency**: 2-10 seconds (depends on backend)
- **Image Generation**: 5-30 seconds
- **Database Operations**: <100ms typical
- **Memory Usage**: ~50-100MB typical

## License

MIT License
