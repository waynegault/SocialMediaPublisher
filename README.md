# Social Media Publisher

A Python application for generating and publishing content to social media platforms using Google's Gemini AI.

## Features

- AI-powered content generation using Google Gemini
- Multi-platform social media publishing
- Scheduled posting
- Image processing support

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
   # Edit .env with your API keys
   ```

## Usage

```bash
python main.py
```

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:

- `GEMINI_API_KEY` - Your Google Gemini API key

## License

MIT License
