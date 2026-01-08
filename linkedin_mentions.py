"""
LinkedIn profile/company search for story mentions.

This module searches for LinkedIn profiles and company pages associated with
stories to enable @ mentions when publishing.
"""

import logging
import json
import re
from typing import Optional
from dataclasses import dataclass

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class LinkedInMention:
    """Represents a LinkedIn profile or company page for mentions."""

    name: str  # Display name (person or company name)
    urn: str  # LinkedIn URN (urn:li:person:xxx or urn:li:organization:xxx)
    mention_type: str  # "person" or "organization"
    role: str  # Role in story context: "company", "ceo", "researcher", "author", etc.
    linkedin_url: Optional[str] = None  # Public LinkedIn URL if known

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "urn": self.urn,
            "type": self.mention_type,
            "role": self.role,
            "linkedin_url": self.linkedin_url,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LinkedInMention":
        return cls(
            name=data.get("name", ""),
            urn=data.get("urn", ""),
            mention_type=data.get("type", "person"),
            role=data.get("role", ""),
            linkedin_url=data.get("linkedin_url"),
        )


class LinkedInMentionSearcher:
    """
    Search for LinkedIn profiles/companies associated with stories.

    Uses Gemini with Google Search grounding to find LinkedIn profiles
    for companies, executives, and researchers mentioned in stories.
    """

    def __init__(self, gemini_client=None):
        """Initialize the mention searcher."""
        self.client = gemini_client

    def search_mentions_for_story(
        self, title: str, summary: str, sources: list[str]
    ) -> list[dict]:
        """
        Search for LinkedIn profiles/companies related to a story.

        Args:
            title: Story title
            summary: Story summary text
            sources: List of source URLs

        Returns:
            List of LinkedIn mention dicts with name, urn, type, role
        """
        if not self.client:
            logger.warning("No Gemini client available for LinkedIn search")
            return []

        # Build the search prompt
        prompt = self._build_search_prompt(title, summary, sources)

        try:
            response = self.client.models.generate_content(
                model=Config.MODEL_TEXT,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "response_mime_type": "application/json",
                    "max_output_tokens": 2048,
                },
            )

            if response.text:
                return self._parse_mentions_response(response.text)

        except Exception as e:
            logger.error(f"LinkedIn mention search failed: {e}")

        return []

    def _build_search_prompt(self, title: str, summary: str, sources: list[str]) -> str:
        """Build the prompt for finding LinkedIn profiles."""
        sources_text = "\n".join(f"- {s}" for s in sources[:3]) if sources else "None"

        return Config.LINKEDIN_MENTION_PROMPT.format(
            title=title,
            summary=summary,
            sources_text=sources_text,
        )

    def _parse_mentions_response(self, response_text: str) -> list[dict]:
        """Parse the JSON response from the LLM."""
        if not response_text:
            return []

        text = response_text.strip()

        # Try to extract JSON from markdown code blocks
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
        elif "```" in text:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)

        try:
            data = json.loads(text)
            mentions = data.get("mentions", [])

            # Validate and clean mentions
            valid_mentions = []
            for m in mentions:
                if not m.get("name") or not m.get("linkedin_url"):
                    continue

                # Validate LinkedIn URL format
                url = m.get("linkedin_url", "")
                if not self._is_valid_linkedin_url(url):
                    logger.debug(f"Invalid LinkedIn URL: {url}")
                    continue

                # Generate URN from URL (placeholder - real URN requires API lookup)
                urn = self._url_to_placeholder_urn(url, m.get("type", "person"))

                valid_mentions.append(
                    {
                        "name": m.get("name", ""),
                        "urn": urn,
                        "type": m.get("type", "person"),
                        "role": m.get("role", ""),
                        "linkedin_url": url,
                    }
                )

            return valid_mentions

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LinkedIn mentions JSON: {e}")
            return []

    def _is_valid_linkedin_url(self, url: str) -> bool:
        """Check if URL is a valid LinkedIn profile/company URL."""
        if not url:
            return False

        patterns = [
            r"^https?://(www\.)?linkedin\.com/in/[\w\-]+/?$",
            r"^https?://(www\.)?linkedin\.com/company/[\w\-]+/?$",
        ]

        for pattern in patterns:
            if re.match(pattern, url, re.IGNORECASE):
                return True

        return False

    def _url_to_placeholder_urn(self, url: str, mention_type: str) -> str:
        """
        Generate a placeholder URN from LinkedIn URL.

        Note: Real URNs require LinkedIn API lookup. This generates a
        placeholder that can be used for display purposes.
        """
        # Extract the profile/company ID from URL
        if "/in/" in url:
            match = re.search(r"/in/([\w\-]+)", url)
            if match:
                return f"urn:li:person:{match.group(1)}"
        elif "/company/" in url:
            match = re.search(r"/company/([\w\-]+)", url)
            if match:
                return f"urn:li:organization:{match.group(1)}"

        return f"urn:li:{mention_type}:unknown"


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests():
    """Create unit tests for linkedin_mentions module."""
    from test_framework import TestSuite

    suite = TestSuite("linkedin_mentions")

    def test_linkedin_mention_dataclass():
        mention = LinkedInMention(
            name="Test Company",
            urn="urn:li:organization:12345",
            mention_type="organization",
            role="company",
            linkedin_url="https://www.linkedin.com/company/test-company",
        )
        assert mention.name == "Test Company"
        assert mention.mention_type == "organization"

        # Test to_dict
        d = mention.to_dict()
        assert d["name"] == "Test Company"
        assert d["type"] == "organization"

        # Test from_dict
        m2 = LinkedInMention.from_dict(d)
        assert m2.name == mention.name
        assert m2.urn == mention.urn

    def test_valid_linkedin_url():
        searcher = LinkedInMentionSearcher()

        # Valid URLs
        assert searcher._is_valid_linkedin_url("https://www.linkedin.com/in/john-doe")
        assert searcher._is_valid_linkedin_url("https://linkedin.com/company/acme-corp")

        # Invalid URLs
        assert not searcher._is_valid_linkedin_url("")
        assert not searcher._is_valid_linkedin_url("https://google.com")
        assert not searcher._is_valid_linkedin_url("https://linkedin.com/jobs/12345")

    def test_url_to_urn():
        searcher = LinkedInMentionSearcher()

        urn = searcher._url_to_placeholder_urn(
            "https://www.linkedin.com/in/john-doe", "person"
        )
        assert urn == "urn:li:person:john-doe"

        urn = searcher._url_to_placeholder_urn(
            "https://www.linkedin.com/company/acme-corp", "organization"
        )
        assert urn == "urn:li:organization:acme-corp"

    suite.add_test("LinkedInMention dataclass", test_linkedin_mention_dataclass)
    suite.add_test("Valid LinkedIn URL check", test_valid_linkedin_url)
    suite.add_test("URL to URN conversion", test_url_to_urn)

    return suite


if __name__ == "__main__":
    suite = _create_module_tests()
    suite.run()
