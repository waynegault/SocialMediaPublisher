"""Company mention enrichment using AI for LinkedIn posts.

This module identifies real companies explicitly mentioned in news sources and adds them
to posts as professional, analytical context. It is conservative by design - when in doubt,
it defaults to NO_COMPANY_MENTION.
"""

import json
import logging
import time

from google import genai  # type: ignore
from openai import OpenAI
import requests
import re

from config import Config
from database import Database, Story

logger = logging.getLogger(__name__)

# Exact string that indicates no company mention should be added
NO_COMPANY_MENTION = "NO_COMPANY_MENTION"


def validate_linkedin_url(url: str) -> bool:
    """
    Validate that a LinkedIn URL returns a valid profile page.

    Performs a HEAD request to check if the URL:
    1. Is accessible (returns 200 OK)
    2. Does not redirect to a login/error page

    Args:
        url: The LinkedIn profile URL to validate

    Returns:
        True if the URL appears to be a valid, accessible profile
    """
    if not url or "linkedin.com/in/" not in url:
        return False

    try:
        # Use headers that mimic a browser to avoid blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)

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
            response = requests.get(
                url, headers=headers, timeout=10, allow_redirects=True
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
        Enrich all pending stories with LinkedIn profiles for relevant_people.

        The enrichment process now works as follows:
        1. During story generation, relevant_people is populated with people mentioned
           in the story AND key leaders from mentioned organizations
        2. This method finds LinkedIn profiles for those people
        3. If relevant_people is empty, fall back to AI extraction

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

                # Check if we have relevant_people from story generation
                if story.relevant_people:
                    logger.info(
                        f"  Using {len(story.relevant_people)} people from story generation"
                    )

                    # Find LinkedIn profiles for relevant_people
                    people_for_lookup = [
                        {
                            "name": p.get("name", ""),
                            "title": p.get("position", ""),
                            "affiliation": p.get("company", ""),
                        }
                        for p in story.relevant_people
                        if p.get("name")
                    ]

                    if people_for_lookup:
                        linkedin_handles = self._find_linkedin_profiles_batch(
                            people_for_lookup
                        )

                        # Update relevant_people with found LinkedIn profiles
                        if linkedin_handles:
                            profiles_by_name = {
                                h.get("name", "").lower(): h for h in linkedin_handles
                            }
                            updated_people = []
                            for person in story.relevant_people:
                                name_lower = person.get("name", "").lower()
                                if name_lower in profiles_by_name:
                                    profile = profiles_by_name[name_lower]
                                    person["linkedin_profile"] = profile.get(
                                        "linkedin_url", ""
                                    )
                                updated_people.append(person)
                            story.relevant_people = updated_people

                        story.linkedin_handles = linkedin_handles
                        logger.info(
                            f"  ✓ Found {len(linkedin_handles)} LinkedIn profiles"
                        )
                        for handle in linkedin_handles:
                            logger.info(
                                f"    → {handle.get('name', '')}: {handle.get('linkedin_url', '')}"
                            )

                    # Also populate organizations from relevant_people companies
                    orgs = list(
                        set(
                            p.get("company", "")
                            for p in story.relevant_people
                            if p.get("company")
                        )
                    )
                    if orgs:
                        story.organizations = orgs

                    story.enrichment_status = "enriched"
                    enriched += 1

                else:
                    # Fall back to AI extraction if no relevant_people
                    logger.info(
                        "  No relevant_people from story generation, using AI extraction..."
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
        """Extract organizations and people from a story using AI."""
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
            logger.warning(f"Failed to parse enrichment JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting orgs/people: {e}")
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
                    leaders = self._get_org_leaders(org)
                    if leaders:
                        all_leaders.extend(leaders)
                        logger.info(f"  ✓ {org}: {len(leaders)} leaders found")
                    else:
                        logger.info(f"  ⏭ {org}: No leaders found")

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

    def _get_org_leaders(self, organization_name: str) -> list[dict]:
        """Get key leaders for an organization using AI."""
        prompt = Config.ORG_LEADERS_PROMPT.format(organization_name=organization_name)

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
            return data.get("leaders", [])

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

    def find_linkedin_handles(self) -> tuple[int, int]:
        """
        Find LinkedIn handles for all people (story_people + org_leaders).
        Returns tuple of (enriched_count, skipped_count).
        """
        # Get stories that have people but no linkedin_handles yet
        stories = self._get_stories_needing_linkedin_handles()

        if not stories:
            logger.info("No stories need LinkedIn handle enrichment")
            return (0, 0)

        total = len(stories)
        logger.info(f"Finding LinkedIn handles for {total} stories...")

        enriched = 0
        skipped = 0

        for i, story in enumerate(stories, 1):
            try:
                logger.info(
                    f"[{i}/{total}] Finding LinkedIn handles for: {story.title}"
                )

                # Combine all people
                all_people = []
                for person in story.story_people:
                    all_people.append(
                        {
                            "name": person.get("name", ""),
                            "title": person.get("title", ""),
                            "affiliation": person.get("affiliation", ""),
                        }
                    )
                for leader in story.org_leaders:
                    all_people.append(
                        {
                            "name": leader.get("name", ""),
                            "title": leader.get("title", ""),
                            "affiliation": leader.get("organization", ""),
                        }
                    )

                # Use batch search with Google Search grounding
                linkedin_handles = self._find_linkedin_profiles_batch(all_people)
                for handle in linkedin_handles:
                    name = handle.get("name", "Unknown")
                    url = handle.get("linkedin_url", "")
                    if url:
                        logger.info(f"    → {name}: {url}")

                story.linkedin_handles = linkedin_handles
                self.db.update_story(story)

                if linkedin_handles:
                    logger.info(f"  ✓ Found {len(linkedin_handles)} LinkedIn handles")
                    enriched += 1
                else:
                    logger.info("  ⏭ No LinkedIn handles found")
                    skipped += 1

            except KeyboardInterrupt:
                logger.warning(
                    f"\nLinkedIn enrichment interrupted at story {i}/{total}"
                )
                break
            except Exception as e:
                logger.error(f"  ! Error: {e}")
                skipped += 1
                continue

        logger.info(
            f"LinkedIn enrichment complete: {enriched} enriched, {skipped} skipped"
        )
        return (enriched, skipped)

    def _find_linkedin_profiles_batch(self, people: list[dict]) -> list[dict]:
        """
        Search for LinkedIn profiles for a batch of people using Gemini with Google Search.

        Uses Google Search grounding to find real LinkedIn profile URLs.
        Returns list of profiles with verified URLs.
        """
        if not people:
            return []

        # Build the people list for the prompt
        people_lines = []
        for i, person in enumerate(people, 1):
            name = person.get("name", "")
            if not name:
                continue
            title = person.get("title", "")
            affiliation = person.get("affiliation", "")
            if title and affiliation:
                people_lines.append(f"{i}. {name} - {title} at {affiliation}")
            elif affiliation:
                people_lines.append(f"{i}. {name} - {affiliation}")
            elif title:
                people_lines.append(f"{i}. {name} - {title}")
            else:
                people_lines.append(f"{i}. {name}")

        if not people_lines:
            return []

        people_list_text = "\n".join(people_lines)
        prompt = Config.LINKEDIN_PROFILE_SEARCH_PROMPT.format(
            people_list=people_list_text
        )

        try:
            # Use Gemini with Google Search grounding to find real profiles
            response = self.client.models.generate_content(
                model=Config.MODEL_TEXT,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "max_output_tokens": 2000,
                },
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

    def _get_stories_needing_linkedin_handles(self) -> list[Story]:
        """Get stories with people but no linkedin_handles yet."""
        stories = []
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM stories
                WHERE enrichment_status = 'enriched'
                AND (
                    (story_people IS NOT NULL AND story_people != '[]')
                    OR (org_leaders IS NOT NULL AND org_leaders != '[]')
                )
                AND (linkedin_handles IS NULL OR linkedin_handles = '[]')
                ORDER BY id DESC
            """)
            for row in cursor.fetchall():
                stories.append(Story.from_row(row))
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
        response = self.client.models.generate_content(
            model=Config.MODEL_TEXT, contents=prompt
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
        response = self.client.models.generate_content(
            model=Config.MODEL_TEXT, contents=prompt
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
                resp = requests.get(
                    "https://api.linkedin.com/v2/organizations",
                    headers=headers,
                    params=params,
                    timeout=10,
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

            # Count stories with LinkedIn handles
            cursor.execute(
                """SELECT COUNT(*) FROM stories
                   WHERE enrichment_status = 'enriched'
                   AND linkedin_handles IS NOT NULL
                   AND linkedin_handles != '[]'"""
            )
            with_handles = cursor.fetchone()[0]

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
            "with_handles": with_handles,
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
