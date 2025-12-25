"""Configuration management for Social Media Publisher."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration."""

    # Google Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Social Media API Keys
    TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
    TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
    TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
    TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

    FACEBOOK_ACCESS_TOKEN = os.getenv("FACEBOOK_ACCESS_TOKEN")
    INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
    LINKEDIN_ACCESS_TOKEN = os.getenv("LINKEDIN_ACCESS_TOKEN")

    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.GEMINI_API_KEY:
            print("Warning: GEMINI_API_KEY is not set")
            return False
        return True
