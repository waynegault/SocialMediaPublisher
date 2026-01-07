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
- **LinkedIn Integration**: Posts stories with images, source links, and your
  signature block
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
├── main.py              # Main entry point and orchestration
├── config.py            # Configuration management
├── database.py          # SQLite database operations
├── searcher.py          # Story discovery using Gemini
├── image_generator.py   # AI image generation using Imagen
├── verifier.py          # Content quality verification
├── scheduler.py         # Publication timing logic
├── linkedin_publisher.py # LinkedIn API integration
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment configuration
└── README.md            # This file
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

## Security Best Practices

### Protecting Your API Keys and Credentials

**IMPORTANT**: Never commit your actual API keys, tokens, or credentials to version control.

1. **Use the .env file**: Copy `.env.example` to `.env` and add your real credentials there. The `.env` file is already in `.gitignore` and will not be committed.

2. **Keep .env.example generic**: The `.env.example` file should only contain placeholder values (e.g., `your_api_key_here`), never real credentials.

3. **Verify before committing**: Before committing changes, run:
   ```bash
   git diff
   ```
   Ensure no real API keys or tokens appear in your changes.

4. **Rotate compromised keys**: If you accidentally commit a real API key:
   - Immediately revoke/regenerate the key in the respective service
   - Remove the key from git history (consider `git filter-branch` or BFG Repo-Cleaner)
   - Update your `.env` file with the new key

5. **Use environment-specific credentials**: For production deployments, use environment variables or secret management services (AWS Secrets Manager, Azure Key Vault, etc.) instead of `.env` files.

### Secured Files

The following files are automatically excluded from git commits via `.gitignore`:
- `.env` - Your actual environment variables with real credentials
- `*.db` - Database files that may contain sensitive data
- `generated_images/` - Generated images directory
- `data/`, `output/` - Local data directories

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

### Gotchas and Known Issues

1. **Gemini Rate Limits**: The 429 RESOURCE_EXHAUSTED error requires
   waiting or upgrading quota
2. **LM Studio Model Loading**: Must have a model loaded before API calls
3. **LinkedIn Token Expiry**: Tokens expire and need manual refresh
4. **DuckDuckGo Rate Limits**: Aggressive searching may trigger blocks

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
- **File**: `stories.db` (configurable)
- **Tables**: `stories`, `state`
- **Indexes**: Primary key on `id`, index on `title`

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
