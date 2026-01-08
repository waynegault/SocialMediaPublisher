"""LinkedIn publishing functionality."""

import logging
import requests
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from config import Config
from database import Database, Story
from opportunity_messages import get_random_opportunity_message

logger = logging.getLogger(__name__)


class LinkedInPublisher:
    """Publish stories to LinkedIn."""

    BASE_URL = "https://api.linkedin.com/v2"
    UPLOAD_URL = "https://api.linkedin.com/v2/assets?action=registerUpload"

    def __init__(self, database: Database):
        """Initialize the LinkedIn publisher."""
        self.db = database
        self._validate_credentials()

    def _validate_credentials(self) -> None:
        """Validate LinkedIn credentials are configured."""
        if not Config.LINKEDIN_ACCESS_TOKEN:
            logger.warning("LINKEDIN_ACCESS_TOKEN is not configured")
        if not Config.LINKEDIN_AUTHOR_URN:
            logger.warning("LINKEDIN_AUTHOR_URN is not configured")

    def _get_headers(self) -> dict:
        """Get request headers for LinkedIn API."""
        return {
            "Authorization": f"Bearer {Config.LINKEDIN_ACCESS_TOKEN}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        }

    def _get_post_url(self, post_id: str) -> str:
        """
        Convert a LinkedIn post ID (URN) to a viewable URL.

        Args:
            post_id: The post URN (e.g., 'urn:li:share:7415014901590642688')

        Returns:
            The LinkedIn URL to view the post
        """
        # LinkedIn feed URLs use the format: /feed/update/{urn}
        return f"https://www.linkedin.com/feed/update/{post_id}"

    def publish_due_stories(self) -> tuple[int, int]:
        """
        Publish all stories that are due.
        Returns tuple of (success_count, failure_count).
        """
        due_stories = self.db.get_scheduled_stories_due()

        if not due_stories:
            return (0, 0)

        logger.info(f"Publishing {len(due_stories)} due stories...")

        success_count = 0
        failure_count = 0

        for story in due_stories:
            try:
                post_id = self._publish_story(story)
                if post_id:
                    story.publish_status = "published"
                    story.published_time = datetime.now()
                    story.linkedin_post_id = post_id
                    story.linkedin_post_url = self._get_post_url(post_id)
                    self.db.update_story(story)
                    success_count += 1
                    logger.info(
                        f"Successfully published story {story.id}: {story.title}"
                    )
                else:
                    failure_count += 1
                    logger.error(f"Failed to publish story {story.id}")
            except Exception as e:
                failure_count += 1
                logger.error(f"Exception publishing story {story.id}: {e}")

        return (success_count, failure_count)

    def _publish_story(self, story: Story) -> str | None:
        """
        Publish a single story to LinkedIn.
        Returns the post ID if successful, None otherwise.
        """
        if not Config.LINKEDIN_ACCESS_TOKEN or not Config.LINKEDIN_AUTHOR_URN:
            logger.error("LinkedIn credentials not configured")
            return None

        # Try to upload image if available
        image_asset = None
        if story.image_path and Path(story.image_path).exists():
            image_asset = self._upload_image(story.image_path)

        # Build and post content
        return self._create_post(story, image_asset)

    def _upload_image(self, image_path: str, owner: str | None = None) -> str | None:
        """
        Upload an image to LinkedIn.
        Returns the asset URN if successful.
        """
        try:
            upload_owner = owner or Config.LINKEDIN_AUTHOR_URN

            # Step 1: Register the upload
            register_payload = {
                "registerUploadRequest": {
                    "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
                    "owner": upload_owner,
                    "serviceRelationships": [
                        {
                            "relationshipType": "OWNER",
                            "identifier": "urn:li:userGeneratedContent",
                        }
                    ],
                }
            }

            response = requests.post(
                self.UPLOAD_URL,
                headers=self._get_headers(),
                json=register_payload,
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(f"Image registration failed: {response.text}")
                return None

            upload_data = response.json()
            upload_url = upload_data["value"]["uploadMechanism"][
                "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"
            ]["uploadUrl"]
            asset_urn = upload_data["value"]["asset"]

            # Step 2: Upload the binary image
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()

            upload_headers = {
                "Authorization": f"Bearer {Config.LINKEDIN_ACCESS_TOKEN}",
                "Content-Type": "application/octet-stream",
            }

            upload_response = requests.put(
                upload_url,
                headers=upload_headers,
                data=image_data,
                timeout=60,
            )

            if upload_response.status_code in (200, 201):
                logger.debug(f"Image uploaded successfully: {asset_urn}")
                return asset_urn
            else:
                logger.error(f"Image upload failed: {upload_response.text}")
                return None

        except Exception as e:
            logger.error(f"Image upload exception: {e}")
            return None

    def _create_post(self, story: Story, image_asset: str | None = None) -> str | None:
        """
        Create a LinkedIn post with the story content.
        Returns the post ID if successful.
        """
        # Format the post text
        post_text = self._format_post_text(story)

        # Build the payload
        if image_asset:
            payload = self._build_image_post_payload(post_text, story, image_asset)
        else:
            payload = self._build_article_post_payload(post_text, story)

        try:
            response = requests.post(
                f"{self.BASE_URL}/ugcPosts",
                headers=self._get_headers(),
                json=payload,
                timeout=30,
            )

            if response.status_code == 201:
                # Extract post ID from response
                post_id = response.headers.get("X-RestLi-Id", "")
                return post_id
            else:
                logger.error(f"Post creation failed: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Post creation exception: {e}")
            return None

    def _format_post_text(self, story: Story) -> str:
        """Format the post text for LinkedIn."""
        # Title and summary
        text_parts = [story.title, "", story.summary, ""]

        # Source links
        sources = story.source_links[:3]  # Limit to 3 sources
        if sources:
            text_parts.append("Sources:")
            for source in sources:
                text_parts.append(f"• {source}")
            text_parts.append("")

        # Add LinkedIn mentions (companies/people related to the story)
        if story.linkedin_mentions:
            mentions_text = self._format_mentions(story.linkedin_mentions)
            if mentions_text:
                text_parts.append(mentions_text)
                text_parts.append("")

        # Add hashtags (max 3)
        if story.hashtags:
            hashtag_str = " ".join(f"#{tag}" for tag in story.hashtags[:3])
            text_parts.append(hashtag_str)
            text_parts.append("")

        # Signature block
        if Config.SIGNATURE_BLOCK:
            text_parts.append(Config.SIGNATURE_BLOCK.strip())
            text_parts.append("")

        # Job opportunity postscript
        if Config.INCLUDE_OPPORTUNITY_MESSAGE:
            opportunity_msg = get_random_opportunity_message()
            text_parts.append(opportunity_msg)

        return "\n".join(text_parts)

    def _format_mentions(self, mentions: list[dict]) -> str:
        """
        Format LinkedIn mentions for the post.

        Uses LinkedIn's mention syntax: @[Name](urn:li:organization:ID)
        Falls back to including LinkedIn URLs if URN is not available.
        """
        if not mentions:
            return ""

        mention_parts = []
        for mention in mentions[:3]:  # Limit to 3 mentions
            name = mention.get("name", "")
            urn = mention.get("urn", "")
            linkedin_url = mention.get("linkedin_url", "")

            if not name:
                continue

            # If we have a proper URN with numeric ID, use @ mention syntax
            # URN format: urn:li:organization:12345 or urn:li:person:12345
            if urn and self._is_valid_urn(urn):
                mention_parts.append(f"@[{name}]({urn})")
            elif linkedin_url:
                # Fallback: include as a regular link
                mention_parts.append(f"{name}: {linkedin_url}")

        if mention_parts:
            return "Related: " + " | ".join(mention_parts)
        return ""

    def _is_valid_urn(self, urn: str) -> bool:
        """Check if URN has a numeric ID (required for @ mentions)."""
        import re

        # Valid URN must end with numeric ID
        pattern = r"^urn:li:(person|organization):\d+$"
        return bool(re.match(pattern, urn))

    def _build_image_post_payload(
        self, text: str, story: Story, image_asset: str
    ) -> dict:
        """Build payload for a post with an image."""
        return {
            "author": Config.LINKEDIN_AUTHOR_URN,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": text},
                    "shareMediaCategory": "IMAGE",
                    "media": [
                        {
                            "status": "READY",
                            "description": {"text": story.summary[:200]},
                            "media": image_asset,
                            "title": {"text": story.title},
                        }
                    ],
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
        }

    def _build_article_post_payload(self, text: str, story: Story) -> dict:
        """Build payload for a post with article link."""
        primary_link = story.source_links[0] if story.source_links else ""

        return {
            "author": Config.LINKEDIN_AUTHOR_URN,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": text},
                    "shareMediaCategory": "ARTICLE",
                    "media": [
                        {
                            "status": "READY",
                            "description": {"text": story.summary[:200]},
                            "originalUrl": primary_link,
                            "title": {"text": story.title},
                        }
                    ]
                    if primary_link
                    else [],
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
        }

    def publish_one_off(
        self, author_urn: str, text: str, image_path: str | None = None
    ) -> str | None:
        """Publish a one-off UGC post for given author (person or org). Returns the post ID if successful."""
        image_asset = None
        if image_path and Path(image_path).exists():
            image_asset = self._upload_image(image_path, owner=author_urn)

        if image_asset:
            payload = {
                "author": author_urn,
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {"text": text},
                        "shareMediaCategory": "IMAGE",
                        "media": [
                            {
                                "status": "READY",
                                "description": {"text": text[:200]},
                                "media": image_asset,
                                "title": {"text": text[:100]},
                            }
                        ],
                    }
                },
                "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
            }
        else:
            payload = {
                "author": author_urn,
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {"text": text},
                        "shareMediaCategory": "NONE",
                    }
                },
                "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
            }

        try:
            response = requests.post(
                f"{self.BASE_URL}/ugcPosts",
                headers=self._get_headers(),
                json=payload,
                timeout=30,
            )

            if response.status_code in (200, 201):
                post_id = response.headers.get("X-RestLi-Id", "")
                logger.info(f"Published one-off post: {post_id}")
                return post_id
            else:
                logger.error(
                    f"One-off post failed: {response.status_code} {response.text}"
                )
                return None

        except Exception as e:
            logger.error(f"One-off post exception: {e}")
            return None

    def test_connection(self) -> bool:
        """Test the LinkedIn API connection."""
        if not Config.LINKEDIN_ACCESS_TOKEN:
            return False

        try:
            response = requests.get(
                f"{self.BASE_URL}/userinfo",
                headers=self._get_headers(),
                timeout=10,
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"LinkedIn connection test failed: {e}")
            return False

    def get_profile_info(self) -> dict | None:
        """Get the authenticated user's profile info."""
        if not Config.LINKEDIN_ACCESS_TOKEN:
            return None

        try:
            response = requests.get(
                f"{self.BASE_URL}/userinfo",
                headers=self._get_headers(),
                timeout=10,
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    def verify_post_exists(self, post_id: str) -> tuple[bool, dict | None]:
        """
        Verify a post exists on LinkedIn by fetching its details.
        Returns tuple of (exists, post_data).

        Uses the LinkedIn REST API /rest/posts endpoint which requires:
        - URL-encoded URN in the path
        - Linkedin-Version header in YYYYMM format
        - X-Restli-Protocol-Version: 2.0.0
        """
        if not Config.LINKEDIN_ACCESS_TOKEN or not post_id:
            return (False, None)

        try:
            # URL-encode the URN (urn:li:share:123 -> urn%3Ali%3Ashare%3A123)
            encoded_urn = quote(post_id, safe="")

            # Use REST API endpoint with required headers
            rest_url = f"https://api.linkedin.com/rest/posts/{encoded_urn}"

            # Get current month in YYYYMM format for Linkedin-Version header
            version = datetime.now().strftime("%Y%m")

            headers = {
                "Authorization": f"Bearer {Config.LINKEDIN_ACCESS_TOKEN}",
                "X-Restli-Protocol-Version": "2.0.0",
                "Linkedin-Version": version,
            }

            response = requests.get(rest_url, headers=headers, timeout=10)

            if response.status_code == 200:
                post_data = response.json()
                # Check if the post has a valid lifecycle state
                lifecycle = post_data.get("lifecycleState", "")
                if lifecycle == "PUBLISHED":
                    logger.info(f"Post verified: {post_id} is PUBLISHED")
                    return (True, post_data)
                else:
                    logger.info(f"Post found but state is: {lifecycle}")
                    return (True, post_data)
            elif response.status_code == 404:
                logger.warning(f"Post not found: {post_id}")
                return (False, None)
            else:
                logger.warning(
                    f"Post verification failed: {response.status_code} - {response.text}"
                )
                return (False, None)
        except Exception as e:
            logger.error(f"Post verification exception: {e}")
            return (False, None)

    def publish_immediately(self, story: Story) -> str | None:
        """
        Publish a story immediately, bypassing scheduling.
        Updates the story status to 'published' if successful.
        Returns the post ID if successful, None otherwise.
        """
        post_id = self._publish_story(story)
        if post_id:
            story.publish_status = "published"
            story.published_time = datetime.now()
            story.linkedin_post_id = post_id
            story.linkedin_post_url = self._get_post_url(post_id)
            self.db.update_story(story)
            logger.info(
                f"Published story {story.id} immediately with post ID: {post_id}"
            )
            logger.info(f"Post URL: {story.linkedin_post_url}")
        return post_id


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for linkedin_publisher module."""
    import os
    import tempfile

    from test_framework import TestSuite

    suite = TestSuite("LinkedIn Publisher Tests")

    def test_publisher_init():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            publisher = LinkedInPublisher(db)
            assert publisher.db is db
        finally:
            os.unlink(db_path)

    def test_get_headers():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            publisher = LinkedInPublisher(db)
            headers = publisher._get_headers()
            assert "Authorization" in headers
            assert "Content-Type" in headers
            assert headers["Content-Type"] == "application/json"
        finally:
            os.unlink(db_path)

    def test_format_post_text():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            publisher = LinkedInPublisher(db)
            story = Story(
                title="Test Title",
                summary="Test summary content",
                source_links=["https://example.com"],
                quality_score=8,
            )
            text = publisher._format_post_text(story)
            assert "Test Title" in text
            assert "Test summary content" in text
            assert "https://example.com" in text
        finally:
            os.unlink(db_path)

    def test_build_image_post_payload():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            publisher = LinkedInPublisher(db)
            story = Story(
                title="Test", summary="Summary", source_links=[], quality_score=7
            )
            payload = publisher._build_image_post_payload(
                "Test text", story, "urn:li:digitalmediaAsset:123"
            )
            assert "author" in payload
            assert "specificContent" in payload
            assert "visibility" in payload
        finally:
            os.unlink(db_path)

    def test_build_article_post_payload():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            publisher = LinkedInPublisher(db)
            story = Story(
                title="Test",
                summary="Summary",
                source_links=["https://example.com"],
                quality_score=7,
            )
            payload = publisher._build_article_post_payload("Test text", story)
            assert "author" in payload
            assert "specificContent" in payload
        finally:
            os.unlink(db_path)

    suite.add_test("Publisher init", test_publisher_init)
    suite.add_test("Get headers", test_get_headers)
    suite.add_test("Format post text", test_format_post_text)
    suite.add_test("Build image post payload", test_build_image_post_payload)
    suite.add_test("Build article post payload", test_build_article_post_payload)

    return suite
