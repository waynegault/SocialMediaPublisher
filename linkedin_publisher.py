"""LinkedIn publishing functionality."""

import logging
import requests
from datetime import datetime
from pathlib import Path

from config import Config
from database import Database, Story

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

    def _upload_image(self, image_path: str) -> str | None:
        """
        Upload an image to LinkedIn.
        Returns the asset URN if successful.
        """
        try:
            # Step 1: Register the upload
            register_payload = {
                "registerUploadRequest": {
                    "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
                    "owner": Config.LINKEDIN_AUTHOR_URN,
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

        # Signature block
        if Config.SIGNATURE_BLOCK:
            text_parts.append(Config.SIGNATURE_BLOCK.strip())

        return "\n".join(text_parts)

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

    def test_connection(self) -> bool:
        """Test the LinkedIn API connection."""
        if not Config.LINKEDIN_ACCESS_TOKEN:
            return False

        try:
            response = requests.get(
                f"{self.BASE_URL}/me",
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
                f"{self.BASE_URL}/me",
                headers=self._get_headers(),
                timeout=10,
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
