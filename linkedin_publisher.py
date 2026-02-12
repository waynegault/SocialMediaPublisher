"""LinkedIn publishing functionality."""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from config import Config
from database import Database, Story
from api_client import api_client
from url_utils import extract_linkedin_public_id

logger = logging.getLogger(__name__)

# Pre-compiled regex for URN validation (must end with numeric ID)
_URN_PATTERN = re.compile(r"^urn:li:(person|organization):\d+$")


@dataclass
class PublishValidationResult:
    """Result of pre-publish validation checks."""

    is_valid: bool = True
    author_verified: bool = False
    author_name: Optional[str] = None
    author_urn_from_api: Optional[str] = None
    mention_validations: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add an error and mark validation as failed."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning (doesn't fail validation)."""
        self.warnings.append(warning)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "author_verified": self.author_verified,
            "author_name": self.author_name,
            "author_urn_from_api": self.author_urn_from_api,
            "mention_validations": self.mention_validations,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class LinkedInPublisher:
    """Publish stories to LinkedIn."""

    BASE_URL = "https://api.linkedin.com/v2"
    UPLOAD_URL = "https://api.linkedin.com/v2/assets?action=registerUpload"

    def __init__(self, database: Database) -> None:
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

            response = api_client.linkedin_request(
                method="POST",
                url=self.UPLOAD_URL,
                headers=self._get_headers(),
                json=register_payload,
                timeout=30,
                endpoint="upload_register",
            )

            if response.status_code != 200:
                response_text = response.text
                if response_text.strip().startswith(
                    "<!"
                ) or response_text.strip().startswith("<html"):
                    logger.error(
                        f"LinkedIn returned HTML instead of JSON (status {response.status_code}). "
                        "Access token may be expired."
                    )
                    raise ValueError(
                        "LinkedIn access token expired or invalid. Please re-authenticate."
                    )
                logger.error(f"Image registration failed: {response_text[:500]}")
                return None

            # Check for HTML in response before parsing JSON
            response_text = response.text
            if response_text.strip().startswith(
                "<!"
            ) or response_text.strip().startswith("<html"):
                logger.error(
                    "LinkedIn returned HTML instead of JSON. Access token may be expired."
                )
                raise ValueError(
                    "LinkedIn access token expired or invalid. Please re-authenticate."
                )

            upload_data = response.json()
            upload_url = upload_data["value"]["uploadMechanism"][
                "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"
            ]["uploadUrl"]
            asset_urn = upload_data["value"]["asset"]

            # Step 2: Upload the binary image
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()

            upload_headers = {**self._get_headers(), "Content-Type": "application/octet-stream"}

            # Use centralized client for rate limiting and retry logic
            upload_response = api_client.http_put(
                upload_url,
                headers=upload_headers,
                data=image_data,
                timeout=60,
                endpoint="linkedin_upload",
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
            response = api_client.linkedin_request(
                method="POST",
                url=f"{self.BASE_URL}/ugcPosts",
                headers=self._get_headers(),
                json=payload,
                timeout=30,
                endpoint="create_post",
            )

            if response.status_code == 201:
                # Extract post ID from response
                post_id = response.headers.get("X-RestLi-Id", "")
                return post_id
            else:
                # Check if we got HTML instead of JSON (usually means auth error)
                response_text = response.text
                if response_text.strip().startswith(
                    "<!"
                ) or response_text.strip().startswith("<html"):
                    logger.error(
                        f"LinkedIn returned HTML instead of JSON (status {response.status_code}). "
                        "This usually means the access token is expired or invalid. "
                        "Please refresh your LinkedIn token."
                    )
                    raise ValueError(
                        "LinkedIn access token expired or invalid. Please re-authenticate with LinkedIn."
                    )
                else:
                    logger.error(
                        f"Post creation failed (status {response.status_code}): {response_text[:500]}"
                    )
                return None

        except ValueError:
            # Re-raise auth errors with clear message
            raise
        except Exception as e:
            logger.error(f"Post creation exception: {e}")
            return None

    def _format_post_text(self, story: Story) -> str:
        """Format the post text for LinkedIn."""
        # Title and summary
        text_parts = [story.title, "", story.summary, ""]

        # Promotion message (if present)
        if story.promotion:
            text_parts.append(story.promotion)
            text_parts.append("")

        # Source links
        sources = story.source_links[:3]  # Limit to 3 sources
        if sources:
            text_parts.append("Source:")
            for source in sources:
                text_parts.append(source)
            text_parts.append("")

        # Add hashtags (max 3)
        if story.hashtags:
            hashtag_str = " ".join(f"#{tag}" for tag in story.hashtags[:3])
            text_parts.append(hashtag_str)
            text_parts.append("")

        # Add LinkedIn mentions from direct_people (directly mentioned in story)
        direct_people_mentions = self._get_mentions_from_people_list(
            story.direct_people
        )
        if direct_people_mentions:
            mentions_text = self._format_mentions(direct_people_mentions)
            if mentions_text:
                text_parts.append(mentions_text)
                text_parts.append("")

        # Add LinkedIn mentions from indirect_people (institution leaders) with paragraph separation
        indirect_people_mentions = self._get_mentions_from_people_list(
            story.indirect_people
        )
        if indirect_people_mentions:
            mentions_text = self._format_mentions(indirect_people_mentions)
            if mentions_text:
                text_parts.append(mentions_text)
                text_parts.append("")

        return "\n".join(text_parts)

    def preview_post(self, story: Story) -> dict:
        """
        Generate a preview of what will be published to LinkedIn.

        Returns a dictionary with:
        - text: The formatted post text
        - char_count: Total character count
        - has_image: Whether the story has an image
        - image_path: Path to image (if any)
        - warnings: List of potential issues
        - stats: Post statistics (word count, hashtag count, mention count)

        This helps users review content before publishing.
        """
        text = self._format_post_text(story)
        char_count = len(text)

        # Use configurable thresholds for post length warnings
        min_chars = Config.LINKEDIN_POST_MIN_CHARS
        optimal_chars = Config.LINKEDIN_POST_OPTIMAL_CHARS
        max_chars = Config.LINKEDIN_POST_MAX_CHARS
        max_hashtags = Config.LINKEDIN_MAX_HASHTAGS

        warnings: list[str] = []
        if char_count < min_chars:
            warnings.append(
                f"Post is very short ({char_count} chars). Consider adding more content."
            )
        elif char_count > max_chars:
            warnings.append(
                f"Post is very long ({char_count} chars). LinkedIn may truncate it."
            )
        elif char_count > optimal_chars:
            warnings.append(
                f"Post is longer than optimal ({char_count} chars). Ideal: 1200-{optimal_chars}."
            )

        # Check for image
        has_image = bool(story.image_path and Path(story.image_path).exists())
        if not has_image:
            warnings.append(
                "No image attached. Posts with images get 2x more engagement."
            )

        # Check hashtags
        hashtag_count = len(story.hashtags) if story.hashtags else 0
        if hashtag_count == 0:
            warnings.append("No hashtags. Consider adding 1-3 relevant hashtags.")
        elif hashtag_count > max_hashtags:
            warnings.append(
                f"Too many hashtags ({hashtag_count}). LinkedIn recommends max 3-{max_hashtags}."
            )

        # Check for mentions
        mention_count = 0
        if story.direct_people:
            mention_count += sum(
                1 for p in story.direct_people if p.get("linkedin_urn")
            )
        if story.indirect_people:
            mention_count += sum(
                1 for p in story.indirect_people if p.get("linkedin_urn")
            )

        # Word count
        word_count = len(text.split())

        return {
            "text": text,
            "char_count": char_count,
            "word_count": word_count,
            "has_image": has_image,
            "image_path": story.image_path if has_image else None,
            "hashtag_count": hashtag_count,
            "mention_count": mention_count,
            "warnings": warnings,
        }

    def _get_mentions_from_people_list(self, people_list: list[dict]) -> list[dict]:
        """
        Generate mentions list from a list of people dicts.

        Works with both direct_people and indirect_people formats.
        """
        mentions = []
        if people_list:
            for person in people_list:
                if person.get("linkedin_urn") or person.get("linkedin_profile"):
                    mentions.append(
                        {
                            "name": person.get("name", ""),
                            "urn": person.get("linkedin_urn", ""),
                            "type": "person",
                            "linkedin_url": person.get("linkedin_profile", ""),
                        }
                    )
        return mentions

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
        # Uses pre-compiled module-level pattern for efficiency
        return bool(_URN_PATTERN.match(urn))

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
            response = api_client.linkedin_request(
                method="POST",
                url=f"{self.BASE_URL}/ugcPosts",
                headers=self._get_headers(),
                json=payload,
                timeout=30,
                endpoint="oneoff_post",
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
            response = api_client.linkedin_request(
                method="GET",
                url=f"{self.BASE_URL}/userinfo",
                headers=self._get_headers(),
                timeout=10,
                endpoint="test_connection",
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
            response = api_client.linkedin_request(
                method="GET",
                url=f"{self.BASE_URL}/userinfo",
                headers=self._get_headers(),
                timeout=10,
                endpoint="profile_info",
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    def validate_before_publish(self, story: Story) -> PublishValidationResult:
        """
        Perform comprehensive validation before publishing to LinkedIn.

        Checks:
        1. Author verification - ensures access token matches configured author URN
        2. Mention URN format - validates all @mention URNs have proper format
        3. Mention person validation - cross-checks URNs against expected person data

        Args:
            story: The story to validate before publishing

        Returns:
            PublishValidationResult with validation status and any errors/warnings
        """
        result = PublishValidationResult()

        # Check 1: Verify credentials are configured
        if not Config.LINKEDIN_ACCESS_TOKEN:
            result.add_error("LINKEDIN_ACCESS_TOKEN is not configured")
            return result

        if not Config.LINKEDIN_AUTHOR_URN:
            result.add_error("LINKEDIN_AUTHOR_URN is not configured")
            return result

        # Check 2: Verify the access token belongs to the configured author
        try:
            profile = self.get_profile_info()
            if profile:
                # The userinfo endpoint returns 'sub' which is the person URN
                api_sub = profile.get("sub", "")
                api_name = profile.get("name", "")

                result.author_name = api_name
                result.author_urn_from_api = api_sub

                # Verify the configured URN matches the authenticated user
                # LinkedIn person URNs can be in different formats:
                # - urn:li:person:12345 (from sub)
                # - person:12345 (short format)
                configured_urn = Config.LINKEDIN_AUTHOR_URN

                # Normalize both for comparison
                configured_id = self._extract_urn_id(configured_urn)
                api_id = self._extract_urn_id(api_sub)

                if configured_id and api_id:
                    if configured_id != api_id:
                        result.add_error(
                            f"Author URN mismatch! Configured: {configured_urn}, "
                            f"Authenticated user: {api_sub} ({api_name})"
                        )
                    else:
                        result.author_verified = True
                        logger.info(f"Author verified: {api_name} ({api_sub})")
                else:
                    result.add_warning(
                        f"Could not verify author URN. Configured: {configured_urn}, "
                        f"API returned: {api_sub}"
                    )
            else:
                result.add_error(
                    "Failed to fetch authenticated user profile. "
                    "Access token may be invalid or expired."
                )
        except Exception as e:
            result.add_error(f"Author verification failed: {e}")

        # Check 3: Validate all @mention URNs
        all_people = []
        if story.direct_people:
            all_people.extend(story.direct_people)
        if story.indirect_people:
            all_people.extend(story.indirect_people)

        for person in all_people:
            name = person.get("name", "Unknown")
            urn = person.get("linkedin_urn", "")
            linkedin_url = person.get("linkedin_profile", "")
            confidence = person.get("match_confidence", "")

            mention_validation = {
                "name": name,
                "urn": urn,
                "linkedin_url": linkedin_url,
                "confidence": confidence,
                "urn_valid": False,
                "urn_format_ok": False,
                "cross_check_ok": False,
                "issues": [],
            }

            if urn:
                # Check URN format
                if self._is_valid_urn(urn):
                    mention_validation["urn_format_ok"] = True
                    mention_validation["urn_valid"] = True
                else:
                    mention_validation["issues"].append(
                        f"Invalid URN format: {urn} (must be urn:li:person:NUMERIC_ID)"
                    )
                    result.add_warning(
                        f"@mention for '{name}' has invalid URN format: {urn}"
                    )

                # Cross-check: If we have both URL and URN, verify they're consistent
                if linkedin_url and "/in/" in linkedin_url:
                    public_id = self._extract_public_id(linkedin_url)
                    if public_id:
                        # Attempt to verify the URN matches the profile
                        # This is a soft check - we can't always verify without API calls
                        mention_validation["linkedin_public_id"] = public_id

                        # If confidence is low or org_fallback, warn about potential mismatch
                        if confidence in ("low", "org_fallback", ""):
                            mention_validation["issues"].append(
                                f"Low confidence match - verify {name} is correct"
                            )
                            result.add_warning(
                                f"@mention for '{name}' has low confidence ({confidence})"
                            )
                        else:
                            mention_validation["cross_check_ok"] = True

            elif linkedin_url:
                # Has URL but no URN
                if "/in/" in linkedin_url:
                    mention_validation["issues"].append(
                        "Has LinkedIn profile URL but no URN for @mention"
                    )
                    result.add_warning(
                        f"'{name}' has profile URL but no URN - cannot @mention"
                    )

            result.mention_validations.append(mention_validation)

        # Summary logging
        valid_mentions = sum(
            1 for m in result.mention_validations if m.get("urn_valid")
        )
        total_people = len(result.mention_validations)

        if total_people > 0:
            logger.info(
                f"Mention validation: {valid_mentions}/{total_people} "
                f"people have valid URNs for @mentions"
            )

        return result

    def _extract_urn_id(self, urn: str) -> Optional[str]:
        """Extract the ID from a LinkedIn URN.

        LinkedIn URNs can have different ID formats:
        - Numeric: urn:li:person:123456789
        - Alphanumeric: urn:li:person:j1j3gunsBl
        - Short format: person:12345

        The API may also return just the bare ID (e.g., 'j1j3gunsBl').
        """
        if not urn:
            return None

        # Match patterns like: urn:li:person:12345 or person:12345 or urn:li:organization:ABC123
        # ID can be numeric or alphanumeric
        match = re.search(r"(?:urn:li:)?(?:person|organization):([a-zA-Z0-9_-]+)", urn)
        if match:
            return match.group(1)

        # If no URN pattern matched, check if it's already a bare ID (alphanumeric string)
        # This handles cases where the API returns just the ID without the URN prefix
        if re.match(r"^[a-zA-Z0-9_-]+$", urn):
            return urn

        return None

    def _extract_public_id(self, linkedin_url: str) -> Optional[str]:
        """Extract the public ID (username) from a LinkedIn profile URL."""
        return extract_linkedin_public_id(linkedin_url)

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

            # LinkedIn REST API version header format is YYYYMM01
            # Use a known valid version (LinkedIn's Marketing APIs)
            version = "202501"

            headers = {**self._get_headers(), "Linkedin-Version": version}

            response = api_client.linkedin_request(
                method="GET",
                url=rest_url,
                headers=headers,
                timeout=10,
                endpoint="verify_post",
            )

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

    def publish_immediately(
        self, story: Story, skip_validation: bool = False
    ) -> str | None:
        """
        Publish a story immediately, bypassing scheduling.
        Updates the story status to 'published' if successful.
        Returns the post ID if successful, None otherwise.

        Args:
            story: The story to publish
            skip_validation: If True, skip pre-publish validation (not recommended)

        Returns:
            Post ID if successful, None otherwise
        """
        # Run pre-publish validation unless explicitly skipped
        if not skip_validation:
            validation = self.validate_before_publish(story)
            if not validation.is_valid:
                logger.error(
                    f"Pre-publish validation failed for story {story.id}: "
                    f"{validation.errors}"
                )
                for error in validation.errors:
                    print(f"  ❌ {error}")
                return None

            # Log warnings but continue
            if validation.warnings:
                logger.warning(
                    f"Pre-publish warnings for story {story.id}: {validation.warnings}"
                )
                for warning in validation.warnings:
                    print(f"  ⚠️  {warning}")

            # Log successful validation
            if validation.author_verified:
                logger.info(
                    f"Publishing as: {validation.author_name} "
                    f"({validation.author_urn_from_api})"
                )

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
        return post_id

    # =========================================================================
    # Analytics Methods
    # =========================================================================

    def fetch_post_analytics(self, story: Story) -> dict | None:
        """
        Fetch analytics for a published LinkedIn post.

        Uses the organizationalEntityShareStatistics endpoint for organization posts,
        or socialActions for personal posts.

        Returns dict with keys: impressions, clicks, likes, comments, shares, engagement
        Returns None if fetch fails.
        """
        if not story.linkedin_post_id:
            logger.warning(f"Story {story.id} has no LinkedIn post ID")
            return None

        if not Config.LINKEDIN_ACCESS_TOKEN:
            logger.error("LinkedIn access token not configured")
            return None

        # Determine if this is an organization post or personal post
        is_org_post = Config.LINKEDIN_ORGANIZATION_URN is not None

        if is_org_post:
            return self._fetch_org_post_analytics(story)
        else:
            return self._fetch_social_actions(story)

    def _fetch_org_post_analytics(self, story: Story) -> dict | None:
        """
        Fetch analytics for an organization post using organizationalEntityShareStatistics.

        Requires rw_organization_admin permission.
        """
        if not Config.LINKEDIN_ORGANIZATION_URN:
            logger.warning("Organization URN not configured")
            return self._fetch_social_actions(story)

        try:
            # Determine numeric organization ID from URN if possible
            import re

            org_urn_value = str(Config.LINKEDIN_ORGANIZATION_URN)
            m = re.search(r"(\d+)$", org_urn_value)
            if m:
                org_id = m.group(1)
            else:
                # Fallback to sending the URN (some endpoints accept it), but it may fail
                org_id = None

            post_id = str(story.linkedin_post_id)
            post_urn = quote(post_id, safe="")

            # Determine if post is UGC or share type
            if "ugcPost" in post_id:
                post_param = f"ugcPosts=List({post_urn})"
            else:
                post_param = f"shares=List({post_urn})"

            # The API expects the full URN format: urn:li:organization:123456
            if org_id:
                org_urn_full = f"urn:li:organization:{org_id}"
            else:
                # Already have full URN
                org_urn_full = org_urn_value

            org_urn_encoded = quote(org_urn_full, safe="")
            url = (
                f"https://api.linkedin.com/rest/organizationalEntityShareStatistics"
                f"?q=organizationalEntity&organizationalEntity={org_urn_encoded}&{post_param}"
            )

            # LinkedIn REST API version header
            version = "202501"
            headers = {**self._get_headers(), "Linkedin-Version": version}

            response = api_client.linkedin_request(
                method="GET",
                url=url,
                headers=headers,
                timeout=15,
                endpoint="org_analytics",
            )

            if response.status_code == 200:
                data = response.json()
                elements = data.get("elements", [])
                if elements:
                    stats = elements[0].get("totalShareStatistics", {})
                    return {
                        "impressions": stats.get("impressionCount", 0),
                        "clicks": stats.get("clickCount", 0),
                        "likes": stats.get("likeCount", 0),
                        "comments": stats.get("commentCount", 0),
                        "shares": stats.get("shareCount", 0),
                        "engagement": stats.get("engagement", 0.0),
                    }
                else:
                    # No stats yet (post may be too new)
                    logger.debug(f"No analytics data yet for story {story.id}")
                    return {
                        "impressions": 0,
                        "clicks": 0,
                        "likes": 0,
                        "comments": 0,
                        "shares": 0,
                        "engagement": 0.0,
                    }
            elif response.status_code == 403:
                logger.warning(
                    "Organization analytics requires rw_organization_admin permission. "
                    "Falling back to socialActions."
                )
                return self._fetch_social_actions(story)
            else:
                logger.warning(
                    f"Org analytics fetch failed: {response.status_code} - "
                    f"{response.text[:200]}"
                )
                return self._fetch_social_actions(story)

        except Exception as e:
            logger.error(f"Error fetching org analytics: {e}")
            return self._fetch_social_actions(story)

    def _fetch_social_actions(self, story: Story) -> dict | None:
        """
        Fetch like/comment counts using socialActions endpoint.

        This works for both personal and organization posts but only provides
        likes and comments (not impressions, clicks, or shares).
        """
        try:
            # URL-encode the post URN (already validated non-None in caller)
            post_urn = quote(str(story.linkedin_post_id), safe="")

            url = f"https://api.linkedin.com/v2/socialActions/{post_urn}"

            headers = self._get_headers()

            response = api_client.linkedin_request(
                method="GET",
                url=url,
                headers=headers,
                timeout=15,
                endpoint="social_actions",
            )

            if response.status_code == 200:
                data = response.json()
                likes = data.get("likesSummary", {}).get("totalLikes", 0)
                comments = data.get("commentsSummary", {}).get(
                    "totalFirstLevelComments", 0
                )
                return {
                    "impressions": 0,  # Not available via socialActions
                    "clicks": 0,  # Not available via socialActions
                    "likes": likes,
                    "comments": comments,
                    "shares": 0,  # Not available via socialActions
                    "engagement": 0.0,  # Cannot calculate without impressions
                }
            else:
                logger.warning(
                    f"socialActions fetch failed: {response.status_code} - "
                    f"{response.text[:200]}"
                )
                return None

        except Exception as e:
            logger.error(f"Error fetching social actions: {e}")
            return None

    def update_story_analytics(self, story: Story) -> bool:
        """
        Fetch and update analytics for a published story.

        Returns True if analytics were successfully updated.
        """
        analytics = self.fetch_post_analytics(story)
        if analytics is None:
            return False

        story.linkedin_impressions = analytics["impressions"]
        story.linkedin_clicks = analytics["clicks"]
        story.linkedin_likes = analytics["likes"]
        story.linkedin_comments = analytics["comments"]
        story.linkedin_shares = analytics["shares"]
        story.linkedin_engagement = analytics["engagement"]
        story.linkedin_analytics_fetched_at = datetime.now()

        return self.db.update_story(story)

    def refresh_all_analytics(self) -> tuple[int, int]:
        """
        Refresh analytics for all published stories.

        Returns tuple of (success_count, failure_count).
        """
        published_stories = self.db.get_published_stories()

        if not published_stories:
            logger.info("No published stories to refresh analytics for")
            return (0, 0)

        success_count = 0
        failure_count = 0

        for story in published_stories:
            if self.update_story_analytics(story):
                success_count += 1
                logger.debug(
                    f"Updated analytics for story {story.id}: "
                    f"👁 {story.linkedin_impressions} | "
                    f"👍 {story.linkedin_likes} | "
                    f"💬 {story.linkedin_comments}"
                )
            else:
                failure_count += 1

        logger.info(
            f"Analytics refresh complete: {success_count} updated, "
            f"{failure_count} failed"
        )
        return (success_count, failure_count)


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests() -> bool:
    """Create unit tests for linkedin_publisher module."""
    import os
    import tempfile

    from test_framework import TestSuite

    suite = TestSuite("LinkedIn Publisher Tests", "linkedin_publisher.py")
    suite.start_suite()

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

    def test_is_valid_urn_valid_person():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            publisher = LinkedInPublisher(db)
            assert publisher._is_valid_urn("urn:li:person:12345") is True
        finally:
            os.unlink(db_path)

    def test_is_valid_urn_valid_organization():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            publisher = LinkedInPublisher(db)
            assert publisher._is_valid_urn("urn:li:organization:67890") is True
        finally:
            os.unlink(db_path)

    def test_is_valid_urn_invalid_format():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            publisher = LinkedInPublisher(db)
            # Non-numeric ID
            assert publisher._is_valid_urn("urn:li:person:abc") is False
            # Wrong prefix
            assert publisher._is_valid_urn("urn:li:company:12345") is False
            # Missing parts
            assert publisher._is_valid_urn("urn:li:person") is False
            # Empty
            assert publisher._is_valid_urn("") is False
        finally:
            os.unlink(db_path)

    def test_get_post_url():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            publisher = LinkedInPublisher(db)
            url = publisher._get_post_url("urn:li:share:7415014901590642688")
            assert "linkedin.com/feed/update" in url
            assert "urn:li:share:7415014901590642688" in url
        finally:
            os.unlink(db_path)

    def test_format_mentions_empty():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            publisher = LinkedInPublisher(db)
            result = publisher._format_mentions([])
            assert result == ""
        finally:
            os.unlink(db_path)

    def test_format_mentions_with_urn():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            publisher = LinkedInPublisher(db)
            mentions = [{"name": "Test Company", "urn": "urn:li:organization:12345"}]
            result = publisher._format_mentions(mentions)
            assert "Related:" in result
            assert "@[Test Company]" in result
            assert "urn:li:organization:12345" in result
        finally:
            os.unlink(db_path)

    def test_format_mentions_with_url_fallback():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            publisher = LinkedInPublisher(db)
            mentions = [
                {
                    "name": "Test Company",
                    "linkedin_url": "https://linkedin.com/company/test",
                }
            ]
            result = publisher._format_mentions(mentions)
            assert "Related:" in result
            assert "Test Company" in result
            assert "linkedin.com/company/test" in result
        finally:
            os.unlink(db_path)

    def test_verify_post_exists_no_token():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            publisher = LinkedInPublisher(db)
            # Without a token, should return False
            exists, data = publisher.verify_post_exists("")
            assert exists is False
            assert data is None
        finally:
            os.unlink(db_path)

    def test_fetch_post_analytics_no_post_id():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            publisher = LinkedInPublisher(db)
            story = Story(
                title="Test",
                summary="Summary",
                source_links=[],
                quality_score=7,
                linkedin_post_id=None,  # No post ID
            )
            result = publisher.fetch_post_analytics(story)
            assert result is None
        finally:
            os.unlink(db_path)

    def test_refresh_all_analytics_empty():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            publisher = LinkedInPublisher(db)
            # With no published stories, should return (0, 0)
            success, failure = publisher.refresh_all_analytics()
            assert success == 0
            assert failure == 0
        finally:
            os.unlink(db_path)

    def test_preview_post():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            publisher = LinkedInPublisher(db)
            # Story with good content
            story = Story(
                title="Test Story About Amazing AI Breakthrough in Healthcare",
                summary="This is a comprehensive test summary for the story that discusses "
                * 5,
                source_links=["https://example.com"],
                quality_score=8,
                hashtags=["#AI", "#Healthcare", "#Technology"],
                image_path="test_image.jpg",
            )
            preview = publisher.preview_post(story)
            assert "text" in preview
            assert "char_count" in preview
            assert preview["char_count"] > 0
            assert "word_count" in preview
            assert preview["word_count"] > 0
            assert "has_image" in preview
            # Image won't exist, so has_image should be False
            assert preview["has_image"] is False
            assert "hashtag_count" in preview
            assert preview["hashtag_count"] == 3
            assert "warnings" in preview
            assert isinstance(preview["warnings"], list)
        finally:
            os.unlink(db_path)

    def test_preview_post_warnings():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            publisher = LinkedInPublisher(db)
            # Story with issues that should generate warnings
            story = Story(
                title="Test",
                summary="Short",
                source_links=[],
                quality_score=5,
                hashtags=[],  # No hashtags
                image_path=None,  # No image
            )
            preview = publisher.preview_post(story)
            assert "warnings" in preview
            # Should have warnings about short post, no hashtags, and no image
            assert len(preview["warnings"]) > 0
            assert preview["has_image"] is False
            assert preview["hashtag_count"] == 0
        finally:
            os.unlink(db_path)

    suite.run_test(
        test_name="Publisher init",
        test_func=test_publisher_init,
        test_summary="Tests Publisher init functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function creates the expected object or result",
    )
    suite.run_test(
        test_name="Get headers",
        test_func=test_get_headers,
        test_summary="Tests Get headers functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Format post text",
        test_func=test_format_post_text,
        test_summary="Tests Format post text functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces correctly formatted output",
    )
    suite.run_test(
        test_name="Build image post payload",
        test_func=test_build_image_post_payload,
        test_summary="Tests Build image post payload functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function creates the expected object or result",
    )
    suite.run_test(
        test_name="Build article post payload",
        test_func=test_build_article_post_payload,
        test_summary="Tests Build article post payload functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function creates the expected object or result",
    )
    suite.run_test(
        test_name="Valid URN - person",
        test_func=test_is_valid_urn_valid_person,
        test_summary="Tests Valid URN with person scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function returns the expected successful result",
    )
    suite.run_test(
        test_name="Valid URN - organization",
        test_func=test_is_valid_urn_valid_organization,
        test_summary="Tests Valid URN with organization scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function returns the expected successful result",
    )
    suite.run_test(
        test_name="Invalid URN formats",
        test_func=test_is_valid_urn_invalid_format,
        test_summary="Tests Invalid URN formats functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function handles invalid input appropriately",
    )
    suite.run_test(
        test_name="Get post URL",
        test_func=test_get_post_url,
        test_summary="Tests Get post URL functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Format mentions - empty",
        test_func=test_format_mentions_empty,
        test_summary="Tests Format mentions with empty scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function returns the expected value",
    )
    suite.run_test(
        test_name="Format mentions - with URN",
        test_func=test_format_mentions_with_urn,
        test_summary="Tests Format mentions with with urn scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces correctly formatted output",
    )
    suite.run_test(
        test_name="Format mentions - URL fallback",
        test_func=test_format_mentions_with_url_fallback,
        test_summary="Tests Format mentions with url fallback scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces correctly formatted output",
    )
    suite.run_test(
        test_name="Verify post exists - no token",
        test_func=test_verify_post_exists_no_token,
        test_summary="Tests Verify post exists with no token scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function returns None as expected",
    )
    suite.run_test(
        test_name="Fetch analytics - no post ID",
        test_func=test_fetch_post_analytics_no_post_id,
        test_summary="Tests Fetch analytics with no post id scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function returns None as expected",
    )
    suite.run_test(
        test_name="Refresh analytics - empty",
        test_func=test_refresh_all_analytics_empty,
        test_summary="Tests Refresh analytics with empty scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function handles empty or missing input gracefully",
    )
    suite.run_test(
        test_name="Preview post",
        test_func=test_preview_post,
        test_summary="Tests Preview post functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Preview post - warnings",
        test_func=test_preview_post_warnings,
        test_summary="Tests Preview post with warnings scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )

    return suite.finish_suite()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)
