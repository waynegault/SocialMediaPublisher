# Social Media Publisher

An automated Python application that discovers news stories, generates AI images,
verifies content quality, and publishes to LinkedIn on a configurable schedule.

## Features

- **Automated Story Discovery**: Searches the internet for news stories matching
  your criteria using Google's Gemini AI with Search grounding
- **Smart Deduplication**: Groups multiple sources covering the same story
  together
- **Quality Scoring**: Rates stories 1-10 based on relevance, significance, and
  source credibility
- **AI Image Generation**: Creates professional illustrations for each story
  using Google's Imagen model
- **Content Verification**: Uses a separate AI pass to verify professionalism,
  decency, and adherence to your criteria
- **Smart Scheduling**: Publishes stories spread evenly across your preferred
  hours with configurable jitter
- **LinkedIn Integration**: Posts stories with images, source links, and your
  signature block
- **Automatic Cleanup**: Removes old unused stories after a configurable
  exclusion period

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

| Variable                | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| `GEMINI_API_KEY`        | Google AI API key from [AI Studio](https://aistudio.google.com/) |
| `LINKEDIN_ACCESS_TOKEN` | OAuth token from [LinkedIn Developer](https://developer.linkedin.com/) |
| `LINKEDIN_AUTHOR_URN`   | Your LinkedIn URN (e.g., `urn:li:person:ABC123`)              |

### Search Settings

| Variable                | Default | Description                                   |
| ----------------------- | ------- | --------------------------------------------- |
| `SEARCH_PROMPT`         | -       | The prompt describing what stories to search for |
| `SEARCH_LOOKBACK_DAYS`  | 7       | Days to look back when searching              |
| `USE_LAST_CHECKED_DATE` | True    | Use last check time instead of lookback days  |
| `SEARCH_CYCLE_HOURS`    | 24      | How often to run search cycles                |

### Content Settings

| Variable             | Default | Description                        |
| -------------------- | ------- | ---------------------------------- |
| `SUMMARY_WORD_COUNT` | 250     | Target word count for summaries    |
| `MIN_QUALITY_SCORE`  | 7       | Minimum score (1-10) for publication |

### Publication Settings

| Variable               | Default | Description                        |
| ---------------------- | ------- | ---------------------------------- |
| `STORIES_PER_CYCLE`    | 3       | Maximum stories to publish per cycle |
| `PUBLISH_WINDOW_HOURS` | 24      | Window to spread publications over |
| `PUBLISH_START_HOUR`   | 8       | Earliest hour to publish (0-23)    |
| `PUBLISH_END_HOUR`     | 20      | Latest hour to publish (0-23)      |
| `JITTER_MINUTES`       | 30      | Random variance in publish time (+/-) |

### Cleanup Settings

| Variable                | Default | Description                             |
| ----------------------- | ------- | --------------------------------------- |
| `EXCLUSION_PERIOD_DAYS` | 30      | Days before unused stories are deleted |

### Signature Block

| Variable          | Default | Description                        |
| ----------------- | ------- | ---------------------------------- |
| `SIGNATURE_BLOCK` | -       | Text/hashtags appended to each post |

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

### Images not generating

- Verify your API key has Imagen access
- Check `MIN_QUALITY_SCORE` isn't too high

### LinkedIn posting fails

- Verify `LINKEDIN_ACCESS_TOKEN` is valid and not expired
- Check `LINKEDIN_AUTHOR_URN` format
- Ensure token has `w_member_social` permission

## License

MIT License
