"""Company mention enrichment using AI for LinkedIn posts.

This module identifies real companies explicitly mentioned in news sources and adds them
to posts as professional, analytical context. It is conservative by design - when in doubt,
it defaults to NO_COMPANY_MENTION.
"""

import json
import logging
import time
from typing import TYPE_CHECKING

from google import genai  # type: ignore
from openai import OpenAI
import requests
import re

from api_client import api_client
from config import Config
from database import Database, Story

if TYPE_CHECKING:
    from linkedin_profile_lookup import LinkedInCompanyLookup

logger = logging.getLogger(__name__)

# Exact string that indicates no company mention should be added
NO_COMPANY_MENTION = "NO_COMPANY_MENTION"


def validate_linkedin_url(url: str, strict: bool = False) -> bool:
    """
    Validate that a LinkedIn URL appears to be a valid profile URL.

    By default (strict=False), only validates URL format since LinkedIn blocks
    unauthenticated HEAD/GET requests. Full validation happens later when
    visiting the profile with an authenticated browser session.

    When strict=True, performs HTTP validation (may fail due to LinkedIn blocks).

    Args:
        url: The LinkedIn profile URL to validate
        strict: If True, perform HTTP request validation (may be blocked by LinkedIn)

    Returns:
        True if the URL appears to be a valid LinkedIn profile URL
    """
    if not url or "linkedin.com/in/" not in url:
        return False

    # Extract the username/slug from the URL
    import re

    match = re.search(r"linkedin\.com/in/([\w\-]+)", url)
    if not match:
        return False

    slug = match.group(1)

    # Basic format validation - reject obviously invalid slugs
    if len(slug) < 2 or len(slug) > 100:
        return False

    # Reject common error page slugs
    invalid_slugs = {"login", "authwall", "error", "404", "unavailable", "uas"}
    if slug.lower() in invalid_slugs:
        return False

    # If not strict mode, accept the URL based on format alone
    if not strict:
        return True

    # Strict mode: perform HTTP validation (may be blocked by LinkedIn)
    try:
        # Use headers that mimic a browser to avoid blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        response = api_client.http_request(
            method="HEAD",
            url=url,
            headers=headers,
            timeout=10,
            allow_redirects=True,
            endpoint="linkedin_profile_validate",
        )

        # Check if we got a successful response
        if response.status_code == 200:
            # Check if we weren't redirected to a login or error page
            final_url = response.url if hasattr(response, "url") else url
            if "/login" in final_url or "/authwall" in final_url:
                logger.debug(f"LinkedIn URL redirected to login: {url}")
                return False
            return True

        # 405 Method Not Allowed - try GET instead
        if response.status_code == 405:
            response = api_client.http_request(
                method="GET",
                url=url,
                headers=headers,
                timeout=10,
                allow_redirects=True,
                endpoint="linkedin_profile_validate",
            )
            if response.status_code == 200:
                # Check we're still on a profile page
                if "/in/" in response.url and "/login" not in response.url:
                    return True

        logger.debug(
            f"LinkedIn URL validation failed with status {response.status_code}: {url}"
        )
        return False

    except requests.exceptions.Timeout:
        # Timeout might mean the URL exists but is slow - accept cautiously
        logger.debug(f"LinkedIn URL timeout (accepting cautiously): {url}")
        return True
    except requests.exceptions.RequestException as e:
        logger.debug(f"LinkedIn URL validation error ({type(e).__name__}): {url}")
        return False


class CompanyMentionEnricher:
    """Enrich stories with company mentions extracted from sources."""

    def __init__(
        self,
        database: Database,
        client: genai.Client,
        local_client: OpenAI | None = None,
    ):
        """Initialize the company mention enricher."""
        self.db = database
        self.client = client
        self.local_client = local_client

    def enrich_pending_stories(self) -> tuple[int, int]:
        """
        Enrich all pending stories with LinkedIn profiles for story_people and org_leaders.

        The enrichment process now works as follows:
        1. During story generation, story_people is populated with people mentioned
           in the story AND key leaders from mentioned organizations
        2. This method finds LinkedIn profiles for those people
        3. If story_people is empty, fall back to AI extraction

        Returns tuple of (enriched_count, skipped_count).
        """
        stories = self.db.get_stories_needing_enrichment()

        if not stories:
            logger.info("No stories pending enrichment")
            return (0, 0)

        total = len(stories)
        logger.info(f"Enriching {total} stories...")

        enriched = 0
        skipped = 0

        for i, story in enumerate(stories, 1):
            try:
                logger.info(f"[{i}/{total}] Enriching: {story.title}")

                # Check if we have story_people from story generation
                if story.story_people:
                    logger.info(
                        f"  Using {len(story.story_people)} people from story generation"
                    )

                    # Find LinkedIn profiles for story_people with enhanced context
                    people_for_lookup = [
                        {
                            "name": p.get("name", ""),
                            "title": p.get("position", ""),
                            "affiliation": p.get("company", ""),
                            "role_type": p.get("role_type", ""),
                            "department": p.get("department", ""),
                            "location": p.get("location", ""),
                        }
                        for p in story.story_people
                        if p.get("name")
                    ]

                    if people_for_lookup:
                        linkedin_profiles = self._find_linkedin_profiles_batch(
                            people_for_lookup,
                            story_title=story.title,
                            story_category=story.category,
                        )

                        # Update story_people with found LinkedIn profiles
                        if linkedin_profiles:
                            profiles_by_name = {
                                h.get("name", "").lower(): h for h in linkedin_profiles
                            }
                            updated_people = []
                            for person in story.story_people:
                                name_lower = person.get("name", "").lower()
                                if name_lower in profiles_by_name:
                                    profile = profiles_by_name[name_lower]
                                    person["linkedin_profile"] = profile.get(
                                        "linkedin_url", ""
                                    )
                                updated_people.append(person)
                            story.story_people = updated_people

                        logger.info(
                            f"  ✓ Found {len(linkedin_profiles)} LinkedIn profiles"
                        )
                        for profile in linkedin_profiles:
                            logger.info(
                                f"    → {profile.get('name', '')}: {profile.get('linkedin_url', '')}"
                            )

                    # Also populate organizations from story_people companies
                    orgs = list(
                        set(
                            p.get("company", "")
                            for p in story.story_people
                            if p.get("company")
                        )
                    )
                    if orgs:
                        story.organizations = orgs

                    story.enrichment_status = "enriched"
                    enriched += 1

                else:
                    # Fall back to AI extraction if no story_people
                    logger.info(
                        "  No story_people from story generation, using AI extraction..."
                    )
                    result = self._extract_orgs_and_people(story)

                    if result:
                        orgs = result.get("organizations", [])
                        people = result.get("story_people", [])

                        story.organizations = orgs
                        story.story_people = people
                        story.enrichment_status = "enriched"

                        if orgs or people:
                            logger.info(
                                f"  ✓ Found {len(orgs)} orgs, {len(people)} people"
                            )
                            if orgs:
                                logger.info(
                                    f"    Organizations: {', '.join(orgs[:3])}{'...' if len(orgs) > 3 else ''}"
                                )
                            if people:
                                names = [p.get("name", "") for p in people[:3]]
                                logger.info(
                                    f"    People: {', '.join(names)}{'...' if len(people) > 3 else ''}"
                                )
                            enriched += 1
                        else:
                            logger.info("  ⏭ No organizations or people found")
                            skipped += 1
                    else:
                        story.enrichment_status = "enriched"
                        logger.info("  ⏭ Could not extract data")
                        skipped += 1

                self.db.update_story(story)

            except KeyboardInterrupt:
                logger.warning(f"\nEnrichment interrupted by user at story {i}/{total}")
                break
            except Exception as e:
                logger.error(f"  ! Error: {e}")
                story.enrichment_status = "skipped"
                self.db.update_story(story)
                skipped += 1
                continue

        logger.info(f"Enrichment complete: {enriched} enriched, {skipped} skipped")
        return (enriched, skipped)

    def _extract_orgs_and_people(self, story: Story) -> dict | None:
        """Extract organizations and people from a story using AI with Google Search grounding.

        Uses Google Search to fetch the actual source article content for more thorough
        extraction of people mentioned in the story.
        """
        sources_str = (
            ", ".join(story.source_links[:5])
            if story.source_links
            else "No sources provided"
        )

        # Build a prompt that instructs the AI to search the source URL for people
        search_prompt = f"""Find all people mentioned in this news story by searching the source URL.

STORY TITLE: {story.title}

STORY SUMMARY: {story.summary}

SOURCE URL TO SEARCH: {sources_str}

TASK: Search the source URL and extract:
1. All researchers, scientists, professors named in the article
2. Any executives, leaders, or spokespersons quoted
3. Authors of the study if it's research
4. Anyone receiving awards or honors

Look for:
- Names in quotes (people who said something)
- Names linked to profile pages
- Names in bylines or "about the author" sections
- Names in research paper citations
- "Senior author", "lead researcher", "principal investigator" mentions

Return a JSON object:
{{
  "organizations": ["Org Name 1", "Org Name 2"],
  "story_people": [
    {{"name": "Full Name", "title": "Their Title/Role", "affiliation": "Their Institution"}}
  ]
}}

If nothing found, return: {{"organizations": [], "story_people": []}}

Return ONLY valid JSON, no explanation."""

        try:
            # Use Gemini with Google Search grounding to fetch source content
            response = api_client.gemini_generate(
                client=self.client,
                model=Config.MODEL_TEXT,
                contents=search_prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "max_output_tokens": 2000,
                },
                endpoint="source_extraction",
            )

            if not response.text:
                logger.warning("Empty response from source article extraction")
                # Fall back to non-grounded extraction
                return self._extract_orgs_and_people_fallback(story)

            response_text = response.text.strip()

            # Clean up response - sometimes AI adds markdown code blocks
            if response_text.startswith("```"):
                response_text = re.sub(r"^```\w*\n?", "", response_text)
                response_text = re.sub(r"\n?```$", "", response_text)

            data = json.loads(response_text)
            orgs = data.get("organizations", [])
            people = data.get("story_people", [])

            logger.info(
                f"  Extracted from source: {len(orgs)} orgs, {len(people)} people"
            )
            for person in people:
                logger.info(
                    f"    → {person.get('name', 'Unknown')} ({person.get('title', '')})"
                )

            return {
                "organizations": orgs,
                "story_people": people,
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse enrichment JSON: {e}")
            return self._extract_orgs_and_people_fallback(story)
        except Exception as e:
            logger.error(f"Error extracting orgs/people with search: {e}")
            return self._extract_orgs_and_people_fallback(story)

    def _extract_orgs_and_people_fallback(self, story: Story) -> dict | None:
        """Fallback extraction without Google Search (uses only summary text)."""
        sources_str = (
            ", ".join(story.source_links[:5])
            if story.source_links
            else "No sources provided"
        )
        prompt = Config.STORY_ENRICHMENT_PROMPT.format(
            story_title=story.title,
            story_summary=story.summary,
            story_sources=sources_str,
        )

        try:
            response_text = self._get_ai_response(prompt)
            if not response_text:
                return None

            # Clean up response - sometimes AI adds markdown code blocks
            response_text = response_text.strip()
            if response_text.startswith("```"):
                response_text = re.sub(r"^```\w*\n?", "", response_text)
                response_text = re.sub(r"\n?```$", "", response_text)

            data = json.loads(response_text)
            return {
                "organizations": data.get("organizations", []),
                "story_people": data.get("story_people", []),
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse enrichment JSON (fallback): {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting orgs/people (fallback): {e}")
            return None

    def find_org_leaders(self) -> tuple[int, int]:
        """
        Find key leaders for organizations in enriched stories.
        Returns tuple of (enriched_count, skipped_count).
        """
        # Get stories that have organizations but no org_leaders yet
        stories = self._get_stories_needing_org_leaders()

        if not stories:
            logger.info("No stories need organization leader enrichment")
            return (0, 0)

        total = len(stories)
        logger.info(f"Finding organization leaders for {total} stories...")

        enriched = 0
        skipped = 0

        for i, story in enumerate(stories, 1):
            try:
                logger.info(f"[{i}/{total}] Finding leaders for: {story.title}")

                all_leaders = []
                for org in story.organizations:
                    # Pass story context for context-aware leader selection
                    leaders = self._get_org_leaders(
                        org,
                        story_category=story.category,
                        story_title=story.title,
                    )
                    if leaders:
                        all_leaders.extend(leaders)
                        logger.info(f"  ✓ {org}: {len(leaders)} leaders found")
                    else:
                        logger.info(f"  ⏭ {org}: No leaders found")

                # Look up LinkedIn profiles for org_leaders using enhanced matching
                if all_leaders:
                    # Build enhanced lookup list with all available context
                    people_for_lookup = [
                        {
                            "name": leader.get("name", ""),
                            "title": leader.get("title", ""),
                            "affiliation": leader.get("organization", ""),
                            "role_type": leader.get("role_type", ""),
                            "department": leader.get("department", ""),
                            "location": leader.get("location", ""),
                        }
                        for leader in all_leaders
                        if leader.get("name")
                    ]

                    if people_for_lookup:
                        linkedin_profiles = self._find_linkedin_profiles_batch(
                            people_for_lookup,
                            story_title=story.title,
                            story_category=story.category,
                        )

                        # Update org_leaders with found LinkedIn profiles
                        if linkedin_profiles:
                            profiles_by_name = {
                                p.get("name", "").lower(): p for p in linkedin_profiles
                            }
                            for leader in all_leaders:
                                name_lower = leader.get("name", "").lower()
                                if name_lower in profiles_by_name:
                                    profile = profiles_by_name[name_lower]
                                    leader["linkedin_profile"] = profile.get(
                                        "linkedin_url", ""
                                    )
                            logger.info(
                                f"  ✓ Found {len(linkedin_profiles)} LinkedIn profiles for leaders"
                            )

                story.org_leaders = all_leaders
                self.db.update_story(story)

                if all_leaders:
                    enriched += 1
                else:
                    skipped += 1

            except KeyboardInterrupt:
                logger.warning(f"\nLeader enrichment interrupted at story {i}/{total}")
                break
            except Exception as e:
                logger.error(f"  ! Error: {e}")
                skipped += 1
                continue

        logger.info(
            f"Leader enrichment complete: {enriched} enriched, {skipped} skipped"
        )
        return (enriched, skipped)

    def _get_org_leaders(
        self,
        organization_name: str,
        story_category: str = "",
        story_title: str = "",
    ) -> list[dict]:
        """Get key leaders for an organization using AI with story context.

        Args:
            organization_name: Name of the organization to find leaders for
            story_category: Category of the story (Research, Business, etc.)
            story_title: Title of the story for additional context

        Returns:
            List of leader dictionaries with name, title, organization, role_type, etc.
        """
        # Format prompt with story context for context-aware leader selection
        prompt = Config.ORG_LEADERS_PROMPT.format(
            organization_name=organization_name,
            story_category=story_category or "General",
            story_title=story_title or "N/A",
        )

        try:
            response_text = self._get_ai_response(prompt)
            if not response_text:
                return []

            # Clean up response
            response_text = response_text.strip()
            if response_text.startswith("```"):
                response_text = re.sub(r"^```\w*\n?", "", response_text)
                response_text = re.sub(r"\n?```$", "", response_text)

            data = json.loads(response_text)
            leaders = data.get("leaders", [])

            # Ensure each leader has required fields with defaults
            for leader in leaders:
                leader.setdefault("role_type", "executive")
                leader.setdefault("department", "")
                leader.setdefault("location", "")
                leader.setdefault("linkedin_profile", "")

            return leaders

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse leaders JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting org leaders: {e}")
            return []

    def _get_stories_needing_org_leaders(self) -> list[Story]:
        """Get stories with organizations but no org_leaders yet."""
        stories = []
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM stories
                WHERE enrichment_status = 'enriched'
                AND organizations IS NOT NULL
                AND organizations != '[]'
                AND (org_leaders IS NULL OR org_leaders = '[]')
                ORDER BY id DESC
            """)
            for row in cursor.fetchall():
                stories.append(Story.from_row(row))
        return stories

    def _find_linkedin_profiles_batch(
        self,
        people: list[dict],
        story_title: str = "",
        story_category: str = "",
    ) -> list[dict]:
        """
        Search for LinkedIn profiles for a batch of people using Gemini with Google Search.

        Uses Google Search grounding to find real LinkedIn profile URLs.
        Returns list of profiles with verified URLs.

        Args:
            people: List of person dictionaries with name, title, affiliation
            story_title: Optional story title for context
            story_category: Optional story category for context

        Returns:
            List of validated profile dictionaries
        """
        if not people:
            return []

        # Build the people list for the prompt with enhanced context
        people_lines = []
        for i, person in enumerate(people, 1):
            name = person.get("name", "")
            if not name:
                continue
            title = person.get("title", "")
            affiliation = person.get("affiliation", "")
            role_type = person.get("role_type", "")
            department = person.get("department", "")
            location = person.get("location", "")

            # Build a detailed line for each person
            parts = [f"{i}. {name}"]
            if title:
                parts.append(f"({title})")
            if affiliation:
                parts.append(f"at {affiliation}")
            if department:
                parts.append(f"in {department}")
            if location:
                parts.append(f"from {location}")
            if role_type:
                parts.append(f"[{role_type}]")

            people_lines.append(" ".join(parts))

        if not people_lines:
            return []

        people_list_text = "\n".join(people_lines)

        # Build story context for better matching
        story_context = ""
        if story_title or story_category:
            context_parts = []
            if story_category:
                context_parts.append(f"Category: {story_category}")
            if story_title:
                context_parts.append(f"Title: {story_title}")
            story_context = "\n".join(context_parts)
        else:
            story_context = "No additional story context available"

        prompt = Config.LINKEDIN_PROFILE_SEARCH_PROMPT.format(
            people_list=people_list_text,
            story_context=story_context,
        )

        try:
            # Use Gemini with Google Search grounding to find real profiles
            response = api_client.gemini_generate(
                client=self.client,
                model=Config.MODEL_TEXT,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "max_output_tokens": 2000,
                },
                endpoint="linkedin_profile_search",
            )

            if not response.text:
                logger.warning("Empty response from LinkedIn profile search")
                return []

            # Parse the JSON response
            text = response.text.strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            profiles = json.loads(text)

            if not isinstance(profiles, list):
                logger.warning(f"Unexpected response format: {type(profiles)}")
                return []

            # Validate and filter results - now includes URL validation
            valid_profiles = []
            for profile in profiles:
                url = profile.get("linkedin_url", "")
                if url and "linkedin.com/in/" in url:
                    # Validate the URL actually returns a valid profile
                    if validate_linkedin_url(url):
                        # Extract username from URL
                        username = (
                            url.split("linkedin.com/in/")[-1].rstrip("/").split("?")[0]
                        )
                        valid_profiles.append(
                            {
                                "name": profile.get("name", ""),
                                "title": profile.get("title", ""),
                                "affiliation": profile.get("affiliation", ""),
                                "handle": f"@{username}" if username else None,
                                "linkedin_url": url,
                            }
                        )
                        logger.debug(f"LinkedIn URL validated: {url}")
                    else:
                        logger.info(
                            f"  ⚠ LinkedIn URL validation failed, skipping: {url}"
                        )

                    # Small delay between validation requests to avoid rate limiting
                    time.sleep(0.5)

            return valid_profiles

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LinkedIn profile response: {e}")
            return []
        except Exception as e:
            logger.error(f"Error finding LinkedIn profiles: {e}")
            return []

    def _find_profile_with_fallback(
        self,
        person: dict,
        org_linkedin_urls: dict[str, str],
        lookup: "LinkedInCompanyLookup",
    ) -> tuple[str, str, bool]:
        """
        Find LinkedIn profile for a person with organization fallback.

        This method implements the high-precision profile matching strategy:
        1. Search for person's LinkedIn profile
        2. If found with high confidence, use it
        3. If not found OR low confidence, fall back to organization profile

        Args:
            person: Person dictionary with name, company, position, etc.
            org_linkedin_urls: Cache of organization name -> LinkedIn URL
            lookup: LinkedInCompanyLookup instance

        Returns:
            Tuple of (linkedin_url, match_type, is_person_profile)
            - linkedin_url: The URL (person or org)
            - match_type: "person", "org_fallback", or "none"
            - is_person_profile: True if this is a personal profile
        """
        from profile_matcher import (
            ProfileMatcher,
            PersonContext,
            RoleType,
            create_person_context,
        )

        name = person.get("name", "")
        org_name = person.get("company", "") or person.get("organization", "")

        if not name:
            return ("", "none", False)

        # Create person context from dictionary
        person_context = create_person_context(person)

        # Get org LinkedIn URL (cached or look up)
        org_url = None
        if org_name:
            if org_name in org_linkedin_urls:
                org_url = org_linkedin_urls[org_name]
            else:
                # Look up org LinkedIn page
                url, slug = lookup.search_company(org_name)
                if url:
                    org_linkedin_urls[org_name] = url
                    org_url = url
                    logger.debug(f"Found org LinkedIn page: {org_name} -> {url}")

        # Use ProfileMatcher for high-precision matching
        matcher = ProfileMatcher(linkedin_lookup=lookup)
        result = matcher.match_person(person_context, org_fallback_url=org_url)

        if result.is_person_profile() and result.matched_profile:
            return (result.matched_profile.linkedin_url, "person", True)
        elif result.confidence.value == "org_fallback" and result.org_linkedin_url:
            logger.info(f"  → Using org fallback for {name}: {result.org_linkedin_url}")
            return (result.org_linkedin_url, "org_fallback", False)
        else:
            return ("", "none", False)

    def find_profiles_with_fallback(self, story: Story) -> dict:
        """
        Find LinkedIn profiles for all people in a story with org fallback.

        This enhanced method uses multi-signal scoring and falls back to
        organization profiles when personal profiles can't be confidently matched.

        Args:
            story: Story object with story_people, org_leaders, and organizations

        Returns:
            Dictionary with:
            - person_profiles: List of people with matched personal profiles
            - org_fallbacks: List of people using org profile fallback
            - org_urls: Mapping of org name -> LinkedIn URL
        """
        from linkedin_profile_lookup import LinkedInCompanyLookup

        result = {
            "person_profiles": [],
            "org_fallbacks": [],
            "org_urls": {},
            "total_matched": 0,
        }

        all_people = (story.story_people or []) + (story.org_leaders or [])
        if not all_people:
            return result

        logger.info(f"Finding profiles with fallback for {len(all_people)} people")

        # Pre-fetch org LinkedIn pages
        org_urls = {}
        with LinkedInCompanyLookup(genai_client=self.client) as lookup:
            # Look up organization pages first (for fallback)
            for org in story.organizations or []:
                url, slug = lookup.search_company(org)
                if url:
                    org_urls[org] = url
                    logger.debug(f"  Org page: {org} -> {url}")

            # Now match each person
            for person in all_people:
                name = person.get("name", "")
                if not name:
                    continue

                linkedin_url, match_type, is_person = self._find_profile_with_fallback(
                    person, org_urls, lookup
                )

                if is_person:
                    result["person_profiles"].append(
                        {
                            **person,
                            "linkedin_profile": linkedin_url,
                            "profile_type": "person",
                        }
                    )
                elif match_type == "org_fallback":
                    result["org_fallbacks"].append(
                        {
                            **person,
                            "linkedin_profile": linkedin_url,
                            "profile_type": "org_fallback",
                        }
                    )

        result["org_urls"] = org_urls
        result["total_matched"] = len(result["person_profiles"]) + len(
            result["org_fallbacks"]
        )
        return result

    def populate_linkedin_mentions(self) -> tuple[int, int]:
        """
        Look up and store URNs directly in story_people.linkedin_urn and org_leaders.linkedin_urn.

        This streamlined approach stores URNs within story_people and org_leaders rather
        than duplicating data in a separate linkedin_mentions column.

        Returns tuple of (enriched_count, skipped_count).
        """
        from linkedin_profile_lookup import LinkedInCompanyLookup

        # Get stories that have profiles but no URNs
        stories = self._get_stories_needing_urns()

        if not stories:
            logger.info("No stories need LinkedIn URN lookup")
            return (0, 0)

        total = len(stories)
        logger.info(f"Looking up LinkedIn URNs for {total} stories...")

        enriched = 0
        skipped = 0

        # Use a single browser session for efficiency
        with LinkedInCompanyLookup(genai_client=self.client) as lookup:
            for i, story in enumerate(stories, 1):
                try:
                    logger.info(f"[{i}/{total}] Looking up URNs for: {story.title}")

                    # Combine story_people and org_leaders for lookup
                    all_people = (story.story_people or []) + (story.org_leaders or [])
                    if not all_people:
                        logger.info("  ⏭ No people to look up")
                        skipped += 1
                        continue

                    # Find people needing URN lookup:
                    # 1. Has personal profile URL but no URN
                    # 2. Has organization profile (wrong type - needs personal profile search)
                    people_needing_lookup = []
                    for p in all_people:
                        profile = p.get("linkedin_profile", "")
                        urn = p.get("linkedin_urn", "")
                        profile_type = p.get("linkedin_profile_type", "")

                        # Has personal profile but no URN
                        if profile and "/in/" in profile and not urn:
                            people_needing_lookup.append(
                                {"person": p, "needs_search": False}
                            )
                        # Has organization profile - needs personal profile search
                        elif (
                            profile_type == "organization"
                            or "urn:li:organization:" in str(urn)
                        ):
                            people_needing_lookup.append(
                                {"person": p, "needs_search": True}
                            )

                    if not people_needing_lookup:
                        logger.info("  ⏭ All people already have personal URNs")
                        skipped += 1
                        continue

                    logger.info(
                        f"  Found {len(people_needing_lookup)} people to lookup"
                    )

                    urns_found = 0
                    for item in people_needing_lookup:
                        person = item["person"]
                        needs_search = item["needs_search"]
                        name = person.get("name", "Unknown")
                        company = person.get("company", "")
                        position = person.get("position", "")

                        import time

                        if needs_search:
                            # Search for personal profile using Google Search
                            logger.info(
                                f"    🔍 Searching for personal profile: {name}"
                            )
                            personal_url = self._search_personal_linkedin(
                                lookup, name, company, position
                            )
                            if personal_url:
                                # Update to personal profile
                                person["linkedin_profile"] = personal_url
                                person["linkedin_profile_type"] = "personal"
                                # Clear wrong org URN
                                person["linkedin_urn"] = None
                                person.pop("linkedin_slug", None)
                                logger.info(
                                    f"    ✓ Found personal profile: {personal_url}"
                                )
                                # Now look up the URN
                                urn = lookup.lookup_person_urn(personal_url)
                                if urn:
                                    person["linkedin_urn"] = urn
                                    urns_found += 1
                                    logger.info(f"    ✓ {name}: {urn}")
                                time.sleep(2.0)
                            else:
                                logger.info(f"    ✗ {name}: No personal profile found")
                        else:
                            # Already has personal profile URL, just look up URN
                            url = person.get("linkedin_profile", "")
                            urn = lookup.lookup_person_urn(url)
                            if urn:
                                person["linkedin_urn"] = urn
                                urns_found += 1
                                logger.info(f"    ✓ {name}: {urn}")
                            else:
                                # Profile URL might be invalid/removed - search for new one
                                logger.info(
                                    f"    ⚠ {name}: Existing profile invalid, searching..."
                                )
                                personal_url = self._search_personal_linkedin(
                                    lookup, name, company, position
                                )
                                if personal_url:
                                    person["linkedin_profile"] = personal_url
                                    logger.info(
                                        f"    ✓ Found new profile: {personal_url}"
                                    )
                                    urn = lookup.lookup_person_urn(personal_url)
                                    if urn:
                                        person["linkedin_urn"] = urn
                                        urns_found += 1
                                        logger.info(f"    ✓ {name}: {urn}")
                                    time.sleep(2.0)
                                else:
                                    logger.info(f"    ✗ {name}: No URN found")
                            time.sleep(2.0)

                    # Save updated story_people and org_leaders
                    self.db.update_story(story)

                    if urns_found > 0:
                        logger.info(
                            f"  ✓ Updated {urns_found}/{len(people_needing_lookup)} URNs"
                        )
                        enriched += 1
                    else:
                        logger.info("  ⏭ No URNs found")
                        skipped += 1

                except KeyboardInterrupt:
                    logger.warning(f"\nURN lookup interrupted at story {i}/{total}")
                    break
                except Exception as e:
                    logger.error(f"  ! Error: {e}")
                    skipped += 1
                    continue

        logger.info(f"URN lookup complete: {enriched} enriched, {skipped} skipped")
        return (enriched, skipped)

    def _search_personal_linkedin(
        self, lookup, name: str, company: str, position: str
    ) -> str | None:
        """Search for a person's personal LinkedIn profile URL using browser-based Google Search."""
        import re as regex
        import urllib.parse

        # Get browser driver from lookup
        driver = lookup._get_uc_driver()
        if not driver:
            logger.warning("Cannot search for LinkedIn - browser driver not available")
            return None

        # Clean name - remove parentheses for cleaner search
        clean_name = regex.sub(r"\([^)]*\)", "", name).strip()

        # Extract first and last name for validation
        name_parts = clean_name.lower().split()
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[-1] if len(name_parts) > 1 else ""

        # Also extract the part in parentheses if present (e.g., "Harry (Shih-I) Tan" -> "Shih-I Tan")
        alt_name = None
        paren_match = regex.search(r"\(([^)]+)\)", name)
        if paren_match:
            parts = name.split()
            if len(parts) > 1:
                last_name_orig = parts[-1]
                alt_name = f"{paren_match.group(1)} {last_name_orig}"

        # Extract location hint from company name
        location = self._extract_location_hint(company)

        # Build search query - use "linkedin" in quotes to ensure it's always included
        if location:
            query = f'"{clean_name}" {location} "linkedin" site:linkedin.com/in'
        else:
            query = f'"{clean_name}" "{company}" "linkedin" site:linkedin.com/in'

        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"

        try:
            logger.debug(f"Searching Google: {query}")
            driver.get(search_url)
            time.sleep(2)

            # Extract LinkedIn URLs from search results
            page_source = driver.page_source
            urls = regex.findall(
                r"https://www\.linkedin\.com/in/[\w\-]+/?", page_source
            )

            # Deduplicate
            seen = set()
            unique_urls = []
            for url in urls:
                url_clean = url.rstrip("/")
                if url_clean not in seen:
                    seen.add(url_clean)
                    unique_urls.append(url_clean)

            # Validate each URL to ensure name matches before accepting
            for url in unique_urls:
                # Try to get the profile name from the page to validate
                try:
                    driver.get(url)
                    time.sleep(1.5)
                    
                    # Extract the profile name from the page title or h1
                    # LinkedIn profile titles are typically "FirstName LastName - Title | LinkedIn"
                    page_title = driver.title.lower() if driver.title else ""
                    
                    # Also try to get the main profile name heading
                    profile_name = ""
                    try:
                        from selenium.webdriver.common.by import By
                        # Try multiple selectors for the profile name
                        for selector in ["h1", ".text-heading-xlarge", "[data-generated-suggestion-target]"]:
                            try:
                                name_elem = driver.find_element(By.CSS_SELECTOR, selector)
                                if name_elem and name_elem.text:
                                    profile_name = name_elem.text.lower().strip()
                                    break
                            except Exception:
                                continue
                    except Exception:
                        pass
                    
                    # Use profile name if found, otherwise fall back to page title
                    name_text = profile_name if profile_name else page_title
                    
                    # For validation, check if BOTH first and last name appear in the profile name
                    # Use word boundary matching to avoid partial matches (e.g., "ning" in "engineering")
                    if first_name and last_name:
                        # Check for whole word matches
                        first_match = regex.search(rf"\b{regex.escape(first_name)}\b", name_text)
                        last_match = regex.search(rf"\b{regex.escape(last_name)}\b", name_text)
                        
                        if first_match and last_match:
                            logger.info(f"Found validated LinkedIn URL: {url} (name: {name_text[:50]})")
                            return url
                        else:
                            logger.debug(
                                f"Skipping {url} - name mismatch (expected '{first_name} {last_name}', found '{name_text[:50]}')"
                            )
                    elif first_name:
                        first_match = regex.search(rf"\b{regex.escape(first_name)}\b", name_text)
                        if first_match:
                            logger.info(
                                f"Found LinkedIn URL (first name match): {url}"
                            )
                            return url
                    else:
                        # No name parts to validate, accept first result
                        logger.debug(f"Found LinkedIn URL (no validation): {url}")
                        return url
                except Exception as e:
                    logger.debug(f"Error validating {url}: {e}")
                    continue

            # Try alternate search with alt_name if first search failed
            if alt_name:
                if location:
                    query2 = f'"{alt_name}" {location} "linkedin" site:linkedin.com/in'
                else:
                    query2 = f'"{alt_name}" "{company}" "linkedin" site:linkedin.com/in'

                search_url2 = (
                    f"https://www.google.com/search?q={urllib.parse.quote(query2)}"
                )
                logger.debug(f"Retry search: {query2}")
                driver.get(search_url2)
                time.sleep(2)

                page_source = driver.page_source
                urls = regex.findall(
                    r"https://www\.linkedin\.com/in/[\w\-]+/?", page_source
                )
                for url in urls:
                    url_clean = url.rstrip("/")
                    if url_clean not in seen:
                        logger.debug(f"Found LinkedIn URL (alt): {url_clean}")
                        return url_clean

            # Try with position as last resort
            if position:
                query3 = f'"{clean_name}" {position} "linkedin" site:linkedin.com/in'
                search_url3 = (
                    f"https://www.google.com/search?q={urllib.parse.quote(query3)}"
                )
                logger.debug(f"Retry with position: {query3}")
                driver.get(search_url3)
                time.sleep(2)

                page_source = driver.page_source
                urls = regex.findall(
                    r"https://www\.linkedin\.com/in/[\w\-]+/?", page_source
                )
                for url in urls:
                    url_clean = url.rstrip("/")
                    if url_clean not in seen:
                        logger.debug(f"Found LinkedIn URL (position): {url_clean}")
                        return url_clean

            logger.debug(f"No LinkedIn profile found for {name}")
            return None

        except Exception as e:
            logger.warning(f"Error searching for personal LinkedIn: {e}")
            return None

    def _extract_location_hint(self, company: str) -> str:
        """Extract a location hint from the company/institution name."""
        location_hints = {
            "Toronto": "Toronto",
            "Stanford": "Stanford",
            "MIT": "Massachusetts",
            "Harvard": "Boston",
            "Berkeley": "Berkeley",
            "UCLA": "Los Angeles",
            "UIUC": "Urbana",
            "Illinois Urbana-Champaign": "Urbana",
            "Urbana-Champaign": "Urbana",
            "Northwestern": "Evanston",
            "Michigan": "Ann Arbor",
            "Penn State": "Pennsylvania",
            "Waterloo": "Waterloo",
            "UBC": "Vancouver",
            "McGill": "Montreal",
            "Caltech": "Pasadena",
            "Princeton": "Princeton",
            "Yale": "New Haven",
            "Columbia": "New York",
            "Cornell": "Ithaca",
            "Duke": "Durham",
            "Rice": "Houston",
            "Georgia Tech": "Atlanta",
            "Carnegie Mellon": "Pittsburgh",
            "CMU": "Pittsburgh",
            "Wisconsin": "Madison",
            "Washington": "Seattle",
            "UW": "Seattle",
            "Texas": "Austin",
            "UT Austin": "Austin",
            "Chicago": "Chicago",
            "Oxford": "Oxford",
            "Cambridge": "Cambridge",
            "ETH": "Zurich",
            "EPFL": "Lausanne",
            "Imperial": "London",
            "UCL": "London",
        }
        for key, location in location_hints.items():
            if key.lower() in company.lower():
                return location
        return ""

    def _get_stories_needing_urns(self) -> list[Story]:
        """Get stories with people needing personal LinkedIn URNs.

        Includes people who:
        1. Have linkedin_profile but no linkedin_urn
        2. Have organization URN instead of personal URN (wrong type)
        """
        stories = []
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            # Stories with story_people or org_leaders containing linkedin_profile
            cursor.execute("""
                SELECT * FROM stories
                WHERE (story_people LIKE '%linkedin_profile%')
                   OR (org_leaders LIKE '%linkedin_profile%')
                ORDER BY id DESC
            """)
            for row in cursor.fetchall():
                story = Story.from_row(row)
                all_people = (story.story_people or []) + (story.org_leaders or [])
                if all_people:
                    needs_lookup = False
                    for p in all_people:
                        has_profile = bool(p.get("linkedin_profile"))
                        has_urn = bool(p.get("linkedin_urn"))
                        profile_type = p.get("linkedin_profile_type", "")
                        urn = p.get("linkedin_urn", "")

                        # Need lookup if: has personal profile URL but no URN
                        if (
                            has_profile
                            and "/in/" in p.get("linkedin_profile", "")
                            and not has_urn
                        ):
                            needs_lookup = True
                            break
                        # Or if: has organization URN (wrong - person needs personal URN)
                        if profile_type == "organization" or (
                            has_urn and "urn:li:organization:" in str(urn)
                        ):
                            needs_lookup = True
                            break
                    if needs_lookup:
                        stories.append(story)
        return stories

    # =========================================================================
    # LEGACY METHODS (kept for backward compatibility)
    # =========================================================================

    def enrich_individuals_for_stories(self) -> tuple[int, int]:
        """
        Find key individuals for stories that have been enriched with company mentions.
        Returns tuple of (enriched_count, skipped_count).
        """
        # Get stories that have company mentions but no individuals yet
        stories = self._get_stories_needing_individual_enrichment()

        if not stories:
            logger.info("No stories need individual enrichment")
            return (0, 0)

        total = len(stories)
        logger.info(f"Finding key individuals for {total} stories...")

        enriched = 0
        skipped = 0

        for i, story in enumerate(stories, 1):
            try:
                logger.info(f"[{i}/{total}] Finding individuals for: {story.title}")

                # Extract company name from enrichment (may be None)
                company_name = self._extract_company_name(
                    story.company_mention_enrichment
                )

                # Find key individuals (works with or without company name)
                individuals = self._find_key_individuals(story, company_name)
                if not individuals:
                    logger.info("  ⏭ No individuals identified")
                    skipped += 1
                    continue

                story.individuals = [
                    ind.get("name", "") for ind in individuals if ind.get("name")
                ]
                logger.info(
                    f"  ✓ Found {len(story.individuals)} individuals: {', '.join(story.individuals)}"
                )

                # Find LinkedIn profiles for each individual
                linkedin_profiles = []
                for ind in individuals:
                    name = ind.get("name", "")
                    title = ind.get("title", "")
                    if name:
                        profile = self._find_linkedin_profile(name, company_name, title)
                        if profile:
                            linkedin_profiles.append(profile)
                            logger.info(
                                f"    → LinkedIn: {name} - {profile.get('linkedin_url', 'URN only')}"
                            )

                story.linkedin_profiles = linkedin_profiles
                self.db.update_story(story)
                enriched += 1

            except KeyboardInterrupt:
                logger.warning(
                    f"\nIndividual enrichment interrupted at story {i}/{total}"
                )
                break
            except Exception as e:
                logger.error(f"  ! Error: {e}")
                skipped += 1
                continue

        logger.info(
            f"Individual enrichment complete: {enriched} enriched, {skipped} skipped"
        )
        return (enriched, skipped)

    def _get_stories_needing_individual_enrichment(self) -> list[Story]:
        """Get enriched stories that don't have individuals yet."""
        stories = []
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            # Get all enriched stories (approved + has image) that don't have individuals yet
            cursor.execute("""
                SELECT * FROM stories
                WHERE enrichment_status = 'enriched'
                AND verification_status = 'approved'
                AND image_path IS NOT NULL
                AND (individuals IS NULL OR individuals = '[]')
                ORDER BY id DESC
            """)
            for row in cursor.fetchall():
                stories.append(Story.from_row(row))
        return stories

    def _extract_company_name(self, mention: str | None) -> str | None:
        """Extract company name from the enrichment mention string."""
        if not mention:
            return None
        # Try to extract from standard format: "Company Name is the primary subject..."
        # or "This development from Company Name demonstrates..."
        try:
            if " is the primary" in mention or " is " in mention:
                return mention.split(" is ", 1)[0].strip().strip('"').strip("'")
            if "from " in mention:
                after_from = mention.split("from ", 1)[1]
                # Get text until next space or punctuation
                company = re.match(r"([A-Za-z0-9\s&\-\.]+)", after_from)
                if company:
                    return company.group(1).strip()
        except Exception:
            pass
        # Fallback: return the whole mention if it's short enough
        if len(mention) < 50:
            return mention
        return None

    def _find_key_individuals(
        self, story: Story, company_name: str | None = None
    ) -> list[dict]:
        """Use AI to find key individuals associated with the story (and optionally company)."""
        # Build company context for prompt
        if company_name:
            company_context = f"COMPANY: {company_name}\n\nFocus on individuals related to this company."
        else:
            company_context = "NOTE: No specific company identified. Look for individuals mentioned in the story itself."

        prompt = Config.INDIVIDUAL_EXTRACTION_PROMPT.format(
            story_title=story.title,
            story_summary=story.summary,
            company_context=company_context,
        )

        try:
            response_text = self._get_ai_response(prompt)
            if not response_text:
                return []

            # Parse JSON response
            # Clean up response - sometimes AI adds markdown code blocks
            response_text = response_text.strip()
            if response_text.startswith("```"):
                response_text = re.sub(r"^```\w*\n?", "", response_text)
                response_text = re.sub(r"\n?```$", "", response_text)

            data = json.loads(response_text)
            individuals = data.get("individuals", [])

            # Validate and clean up
            valid_individuals = []
            for ind in individuals:
                if isinstance(ind, dict) and ind.get("name"):
                    valid_individuals.append(
                        {
                            "name": ind.get("name", "").strip(),
                            "title": ind.get("title", "").strip(),
                            "source": ind.get("source", "unknown"),
                        }
                    )
            return valid_individuals[:5]  # Max 5 individuals

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse individuals JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Error finding individuals: {e}")
            return []

    def _get_ai_response(self, prompt: str) -> str | None:
        """Get response from AI (local or Gemini)."""
        if self.local_client:
            try:
                response = self.local_client.chat.completions.create(
                    model=Config.LM_STUDIO_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content
                if content:
                    return content.strip()
            except Exception as e:
                logger.warning(f"Local AI failed: {e}. Falling back to Gemini.")

        # Fallback to Gemini
        response = api_client.gemini_generate(
            client=self.client,
            model=Config.MODEL_TEXT,
            contents=prompt,
            endpoint="ai_response",
        )
        return response.text.strip() if response.text else None

    def _find_linkedin_profile(
        self, name: str, company: str | None = None, title: str = ""
    ) -> dict | None:
        """Search for a person's LinkedIn profile."""
        if not Config.LINKEDIN_ACCESS_TOKEN:
            logger.debug("No LinkedIn token - skipping profile search")
            return None

        headers = {
            "Authorization": f"Bearer {Config.LINKEDIN_ACCESS_TOKEN}",
            "X-Restli-Protocol-Version": "2.0.0",
        }

        # Try LinkedIn People Search API
        # Note: This requires specific LinkedIn API permissions (r_liteprofile, etc.)
        try:
            # Search by name and optionally company
            search_query = f"{name} {company}" if company else name

            # LinkedIn's People Search requires Marketing API or specific permissions
            # As a fallback, we construct a likely LinkedIn URL
            # Real implementation would use LinkedIn's People Search API

            # Construct a search URL that the user can use
            import urllib.parse

            encoded_query = urllib.parse.quote(search_query)
            search_url = f"https://www.linkedin.com/search/results/people/?keywords={encoded_query}"

            return {
                "name": name,
                "title": title,
                "company": company or "",
                "linkedin_url": search_url,
                "urn": None,  # Would be populated if we had People Search API access
            }

        except Exception as e:
            logger.debug(f"LinkedIn profile search failed for {name}: {e}")
            return None

    def _enrich_story(self, story: Story) -> tuple[str, str]:
        """
        Enrich a single story with company mentions.
        Returns tuple of (mention_or_reason, status).
        Status can be: "completed", "skipped", or "error"
        """
        try:
            # Build enrichment prompt
            prompt = self._build_enrichment_prompt(story)

            # Get company mention from AI
            mention = self._get_company_mention(prompt)

            # Validate the response
            mention = self._validate_mention(mention)

            return (mention, "completed")

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.error(
                    f"Enrichment failed: API quota exceeded (429 RESOURCE_EXHAUSTED)"
                )
                return ("API quota exceeded", "error")
            logger.error(f"Enrichment error for story {story.id}: {e}")
            return (str(e), "error")

    def _get_company_mention(self, prompt: str) -> str:
        """Get company mention from AI model.
        Returns the mention text or NO_COMPANY_MENTION.
        """
        # Use local LLM if available
        if self.local_client:
            try:
                logger.info("Using local LLM for company mention enrichment...")
                response = self.local_client.chat.completions.create(
                    model=Config.LM_STUDIO_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content
                if content:
                    return content.strip()
            except Exception as e:
                logger.warning(f"Local enrichment failed: {e}. Falling back to Gemini.")

        # Fallback to Gemini
        response = api_client.gemini_generate(
            client=self.client,
            model=Config.MODEL_TEXT,
            contents=prompt,
            endpoint="company_mention",
        )
        if not response.text:
            logger.warning("Empty response from Gemini during enrichment")
            return NO_COMPANY_MENTION
        return response.text.strip()

    def _build_enrichment_prompt(self, story: Story) -> str:
        """Build the enrichment prompt for a story."""
        sources_str = (
            ", ".join(story.source_links[:5])
            if story.source_links
            else "No sources provided"
        )
        return Config.COMPANY_MENTION_PROMPT.format(
            story_title=story.title,
            story_summary=story.summary,
            story_sources=sources_str,
        )

    def _validate_mention(self, mention: str) -> str:
        """
        Validate mention output.
        Returns either a valid sentence or NO_COMPANY_MENTION.
        """
        mention = mention.strip()

        # If AI returned multiple lines, take the first non-empty line that looks like a sentence
        if "\n" in mention:
            lines = [line.strip() for line in mention.split("\n") if line.strip()]
            # Find the first line that looks like a valid sentence (ends with punctuation)
            for line in lines:
                if line.endswith((".", "!", "?")) and not line.startswith(
                    ("NO_COMPANY", "If", "Note:", "Respond")
                ):
                    mention = line
                    break
            else:
                # No valid sentence found in any line
                logger.warning(
                    f"Invalid mention (no valid sentence in multi-line response): {mention[:50]}"
                )
                return NO_COMPANY_MENTION

        # If it's the no-mention marker, return it as-is
        if mention == NO_COMPANY_MENTION:
            return NO_COMPANY_MENTION

        # Validate that it's a single sentence
        # A valid sentence should:
        # 1. Not be empty
        # 2. End with a period (or be very short and professional)
        # 3. Not contain newlines (single sentence rule)
        # 4. Be reasonable length (max 350 chars for longer org names)

        if not mention:
            return NO_COMPANY_MENTION

        if "\n" in mention:
            logger.warning(f"Invalid mention (contains newlines): {mention[:50]}")
            return NO_COMPANY_MENTION

        if len(mention) > 350:
            logger.warning(f"Invalid mention (too long): {mention[:50]}")
            return NO_COMPANY_MENTION

        # Ensure it ends with proper punctuation
        if not mention.endswith((".", "!", "?")):
            # If it doesn't end with punctuation, check if it looks like a sentence
            # Default to rejection if ambiguous
            logger.warning(f"Invalid mention (no ending punctuation): {mention}")
            return NO_COMPANY_MENTION

        # Check for multiple sentences - but allow abbreviations like Inc., Ltd., Corp., etc.
        # Remove common abbreviations before counting periods
        temp_mention = mention
        for abbrev in [
            "Inc.",
            "Ltd.",
            "Corp.",
            "Co.",
            "LLC.",
            "L.L.C.",
            "P.L.C.",
            "S.A.",
            "N.V.",
            "GmbH.",
            "AG.",
            "Dr.",
            "Prof.",
            "Mr.",
            "Ms.",
            "Mrs.",
        ]:
            temp_mention = temp_mention.replace(abbrev, "ABBREV")

        sentence_count = (
            temp_mention.count(".") + temp_mention.count("!") + temp_mention.count("?")
        )
        if sentence_count > 1:
            logger.warning(f"Invalid mention (multiple sentences): {mention[:50]}")
            return NO_COMPANY_MENTION

        # Check for list indicators (these shouldn't be in a single sentence)
        # But allow hyphens in company names
        if any(indicator in mention for indicator in ["1.", "2.", "•", "* "]):
            logger.warning(f"Invalid mention (contains list): {mention[:50]}")
            return NO_COMPANY_MENTION

        # Check for inappropriate elements
        if any(
            bad in mention.lower()
            for bad in ["@", "#", "no_company_mention", "hashtag", "tag"]
        ):
            logger.warning(f"Invalid mention (inappropriate elements): {mention[:50]}")
            return NO_COMPANY_MENTION

        # If all validation passes, it's acceptable
        return mention

    def _resolve_linkedin_urn(self, company_name: str) -> str | None:
        """Try to resolve a LinkedIn organization URN for the company name.

        Uses the LinkedIn Organizations endpoint with `q=vanityName` and a few
        heuristic attempts (original name and a slugified form).
        Returns a URN string like 'urn:li:organization:12345' on success or None.
        """
        if not Config.LINKEDIN_ACCESS_TOKEN:
            return None

        headers = {
            "Authorization": f"Bearer {Config.LINKEDIN_ACCESS_TOKEN}",
            "X-Restli-Protocol-Version": "2.0.0",
        }

        # Build simple slug candidate
        slug = re.sub(r"[^a-z0-9-]", "", company_name.lower().replace(" ", "-"))

        candidates = [company_name, slug]

        for candidate in candidates:
            if not candidate:
                continue
            params = {"q": "vanityName", "vanityName": candidate}
            try:
                resp = api_client.linkedin_request(
                    method="GET",
                    url="https://api.linkedin.com/v2/organizations",
                    headers=headers,
                    params=params,
                    timeout=10,
                    endpoint="org_lookup",
                )
            except requests.RequestException:
                continue

            if resp.status_code != 200:
                continue

            data = resp.json()
            elements = data.get("elements", [])
            if not elements:
                continue

            # Iterate returned elements and verify match heuristically
            for elem in elements:
                org_id = elem.get("id") or elem.get("organization", {}).get("id")
                if not org_id:
                    continue

                try:
                    if self._confirm_organization(company_name, elem):
                        return f"urn:li:organization:{org_id}"
                except Exception:
                    # If verification fails, still fall back to first id
                    return f"urn:li:organization:{org_id}"

        return None

    def _confirm_organization(self, company_name: str, org_elem: dict) -> bool:
        """Heuristic check whether org_elem matches company_name.

        Checks localizedName, vanityName and some token overlap. Conservative by design.
        """
        try:
            name = company_name.lower().strip()
            localized = "".join(org_elem.get("localizedName", "") or "").lower()
            vanity = "".join(org_elem.get("vanityName", "") or "").lower()

            # Exact match or substring match
            if localized and (
                name == localized or name in localized or localized in name
            ):
                return True
            if vanity and (name == vanity or name in vanity or vanity in name):
                return True

            # Token overlap (at least 2 tokens in common)
            def tokens(s: str):
                return {t for t in re.split(r"\W+", s) if t}

            name_tokens = tokens(name)
            if localized:
                loc_tokens = tokens(localized)
                if len(name_tokens & loc_tokens) >= 2:
                    return True
            if vanity:
                van_tokens = tokens(vanity)
                if len(name_tokens & van_tokens) >= 2:
                    return True

        except Exception as e:
            logger.debug(f"Error in _confirm_organization: {e}")
        return False

    def get_enrichment_stats(self) -> dict:
        """Get enrichment statistics."""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()

            # Count total enriched stories
            cursor.execute(
                "SELECT COUNT(*) FROM stories WHERE enrichment_status = 'enriched'"
            )
            total_enriched = cursor.fetchone()[0]

            # Count stories with organizations
            cursor.execute(
                """SELECT COUNT(*) FROM stories
                   WHERE enrichment_status = 'enriched'
                   AND organizations IS NOT NULL
                   AND organizations != '[]'"""
            )
            with_orgs = cursor.fetchone()[0]

            # Count stories with people (story_people or org_leaders)
            cursor.execute(
                """SELECT COUNT(*) FROM stories
                   WHERE enrichment_status = 'enriched'
                   AND (
                       (story_people IS NOT NULL AND story_people != '[]')
                       OR (org_leaders IS NOT NULL AND org_leaders != '[]')
                   )"""
            )
            with_people = cursor.fetchone()[0]

            # Count stories with LinkedIn URNs in story_people or org_leaders
            cursor.execute(
                """SELECT COUNT(*) FROM stories
                   WHERE enrichment_status = 'enriched'
                   AND (story_people LIKE '%linkedin_urn%' OR org_leaders LIKE '%linkedin_urn%')"""
            )
            with_urns = cursor.fetchone()[0]

            # Count pending
            cursor.execute(
                "SELECT COUNT(*) FROM stories WHERE enrichment_status = 'pending'"
            )
            pending = cursor.fetchone()[0]

        return {
            "pending": pending,
            "total_enriched": total_enriched,
            "with_orgs": with_orgs,
            "with_people": with_people,
            "with_urns": with_urns,
            # Legacy fields for backward compatibility
            "with_mentions": with_orgs,
            "no_mentions": total_enriched - with_orgs,
        }

    def force_set_mention(self, story_id: int, mention: str | None) -> bool:
        """
        Manually set company mention for a story.
        Pass None or empty string to clear mention.
        Returns True if successful.
        """
        story = self.db.get_story(story_id)
        if not story:
            return False

        # Validate the mention if provided
        if mention:
            mention = self._validate_mention(mention)
            if mention == NO_COMPANY_MENTION:
                mention = None

        story.company_mention_enrichment = mention
        story.enrichment_status = "completed"
        return self.db.update_story(story)


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for company_mention_enricher module."""
    import os
    import tempfile

    from test_framework import TestSuite

    suite = TestSuite("Company Mention Enricher Tests")

    def test_confirm_organization_exact_match():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            elem = {
                "localizedName": "SOLVE Chemistry",
                "vanityName": "solve-chemistry",
                "id": 123,
            }
            assert enricher._confirm_organization("SOLVE Chemistry", elem)
        finally:
            os.unlink(db_path)

    def test_confirm_organization_token_overlap():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            elem = {
                "localizedName": "Greentown Labs",
                "vanityName": "greentownlabs",
                "id": 456,
            }
            assert enricher._confirm_organization("Greentown Labs", elem)
        finally:
            os.unlink(db_path)

    suite.add_test("Confirm org - exact match", test_confirm_organization_exact_match)
    suite.add_test(
        "Confirm org - token overlap", test_confirm_organization_token_overlap
    )

    def test_validate_mention_valid_sentence():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            result = enricher._validate_mention(
                "This work integrates BASF's established catalysis technology."
            )
            assert (
                result
                == "This work integrates BASF's established catalysis technology."
            )
        finally:
            os.unlink(db_path)

    def test_validate_mention_no_company():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            result = enricher._validate_mention(NO_COMPANY_MENTION)
            assert result == NO_COMPANY_MENTION
        finally:
            os.unlink(db_path)

    def test_validate_mention_empty():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            result = enricher._validate_mention("")
            assert result == NO_COMPANY_MENTION
        finally:
            os.unlink(db_path)

    def test_validate_mention_with_newlines():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            result = enricher._validate_mention("Line 1.\nLine 2.")
            assert result == NO_COMPANY_MENTION
        finally:
            os.unlink(db_path)

    def test_validate_mention_with_hashtag():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            result = enricher._validate_mention(
                "This mentions Dow. #ChemicalEngineering"
            )
            assert result == NO_COMPANY_MENTION
        finally:
            os.unlink(db_path)

    def test_validate_mention_with_list():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            result = enricher._validate_mention("Companies: 1. Dow 2. BASF.")
            assert result == NO_COMPANY_MENTION
        finally:
            os.unlink(db_path)

    def test_validate_mention_missing_punctuation():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            result = enricher._validate_mention("This mentions Dow")
            assert result == NO_COMPANY_MENTION
        finally:
            os.unlink(db_path)

    def test_build_enrichment_prompt():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            story = Story(
                title="BASF Innovation",
                summary="Development by BASF team",
                source_links=["https://example.com/basf"],
            )
            prompt = enricher._build_enrichment_prompt(story)
            assert "BASF Innovation" in prompt
            assert "Development by BASF team" in prompt
            assert "https://example.com/basf" in prompt
        finally:
            os.unlink(db_path)

    def test_get_enrichment_stats():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            stats = enricher.get_enrichment_stats()
            assert "pending" in stats
            assert "total_enriched" in stats
        finally:
            os.unlink(db_path)

    suite.add_test(
        "Validate mention - valid sentence", test_validate_mention_valid_sentence
    )
    suite.add_test(
        "Validate mention - NO_COMPANY_MENTION", test_validate_mention_no_company
    )
    suite.add_test("Validate mention - empty string", test_validate_mention_empty)
    suite.add_test(
        "Validate mention - with newlines", test_validate_mention_with_newlines
    )
    suite.add_test(
        "Validate mention - with hashtag", test_validate_mention_with_hashtag
    )
    suite.add_test("Validate mention - with list", test_validate_mention_with_list)
    suite.add_test(
        "Validate mention - missing punctuation",
        test_validate_mention_missing_punctuation,
    )
    suite.add_test("Build enrichment prompt", test_build_enrichment_prompt)
    suite.add_test("Get enrichment stats", test_get_enrichment_stats)

    return suite
