"""Unified data models for Social Media Publisher.

This module provides shared data classes used across the LinkedIn lookup
and publishing components, reducing duplication and ensuring consistency.

Models:
- LinkedInProfile: Unified representation of a LinkedIn person profile
- LinkedInOrganization: Unified representation of a LinkedIn organization/company

These models consolidate:
- LinkedInPerson from linkedin_voyager_client.py
- LinkedInProfileResult from linkedin_rapidapi_client.py
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class LinkedInProfile:
    """
    Unified LinkedIn profile representation.

    This class combines fields from:
    - LinkedInPerson (linkedin_voyager_client.py)
    - LinkedInProfileResult (linkedin_rapidapi_client.py)

    Provides a consistent interface for all LinkedIn profile data regardless
    of the source (Voyager API, RapidAPI, browser search, etc.).
    """

    # Core identifiers
    urn_id: str = ""  # LinkedIn URN ID (e.g., "ACoAABxxxxxx")
    public_id: str = ""  # Vanity URL slug (e.g., "john-smith-123")
    linkedin_url: str = ""  # Full profile URL

    # Name fields
    first_name: str = ""
    last_name: str = ""
    full_name: str = ""  # Combined name or "name" field

    # Professional info
    headline: str = ""  # Current job title/description
    job_title: str = ""  # Extracted job title
    company: str = ""  # Current company name
    company_linkedin_url: str = ""  # Company's LinkedIn page URL

    # Location and media
    location: str = ""
    about: str = ""  # Bio/summary
    profile_image_url: str = ""

    # Matching metadata
    distance: int = 0  # Connection distance (1=1st, 2=2nd, 3=3rd+)
    match_score: float = 0.0
    match_signals: list[str] = field(default_factory=list)
    confidence: str = ""  # "high", "medium", "low"

    # Additional data (from RapidAPI)
    company_domain: str = ""
    company_industry: str = ""
    connection_count: int = 0

    def __post_init__(self) -> None:
        """Auto-populate derived fields."""
        # Build full_name if not provided
        if not self.full_name and (self.first_name or self.last_name):
            self.full_name = f"{self.first_name} {self.last_name}".strip()

        # Build linkedin_url if not provided
        if self.public_id and not self.linkedin_url:
            self.linkedin_url = f"https://www.linkedin.com/in/{self.public_id}"

    @property
    def name(self) -> str:
        """Alias for full_name for backward compatibility."""
        return self.full_name

    @property
    def profile_url(self) -> str:
        """Alias for linkedin_url for backward compatibility."""
        return self.linkedin_url

    @property
    def mention_urn(self) -> str:
        """Get URN format for @mentions."""
        return f"urn:li:person:{self.urn_id}" if self.urn_id else ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching."""
        return {
            "urn_id": self.urn_id,
            "public_id": self.public_id,
            "linkedin_url": self.linkedin_url,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "headline": self.headline,
            "job_title": self.job_title,
            "company": self.company,
            "company_linkedin_url": self.company_linkedin_url,
            "location": self.location,
            "about": self.about,
            "profile_image_url": self.profile_image_url,
            "distance": self.distance,
            "match_score": self.match_score,
            "match_signals": self.match_signals,
            "confidence": self.confidence,
            "company_domain": self.company_domain,
            "company_industry": self.company_industry,
            "connection_count": self.connection_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinkedInProfile":
        """Create from dictionary (cache retrieval)."""
        return cls(
            urn_id=data.get("urn_id", ""),
            public_id=data.get("public_id", ""),
            linkedin_url=data.get("linkedin_url", ""),
            first_name=data.get("first_name", ""),
            last_name=data.get("last_name", ""),
            full_name=data.get("full_name", ""),
            headline=data.get("headline", ""),
            job_title=data.get("job_title", ""),
            company=data.get("company", ""),
            company_linkedin_url=data.get("company_linkedin_url", ""),
            location=data.get("location", ""),
            about=data.get("about", ""),
            profile_image_url=data.get("profile_image_url", ""),
            distance=data.get("distance", 0),
            match_score=data.get("match_score", 0.0),
            match_signals=data.get("match_signals", []),
            confidence=data.get("confidence", ""),
            company_domain=data.get("company_domain", ""),
            company_industry=data.get("company_industry", ""),
            connection_count=data.get("connection_count", 0),
        )

    @classmethod
    def from_voyager_person(cls, person: Any) -> "LinkedInProfile":
        """
        Create from LinkedInPerson (linkedin_voyager_client.py).

        Args:
            person: A LinkedInPerson instance

        Returns:
            LinkedInProfile with mapped fields
        """
        return cls(
            urn_id=getattr(person, "urn_id", ""),
            public_id=getattr(person, "public_id", ""),
            linkedin_url=getattr(person, "profile_url", ""),
            first_name=getattr(person, "first_name", ""),
            last_name=getattr(person, "last_name", ""),
            full_name=getattr(person, "name", ""),
            headline=getattr(person, "headline", ""),
            location=getattr(person, "location", ""),
            distance=getattr(person, "distance", 0),
            match_score=getattr(person, "match_score", 0.0),
            match_signals=getattr(person, "match_signals", []),
        )

    @classmethod
    def from_rapidapi_result(cls, result: Any) -> "LinkedInProfile":
        """
        Create from LinkedInProfileResult (linkedin_rapidapi_client.py).

        Args:
            result: A LinkedInProfileResult instance

        Returns:
            LinkedInProfile with mapped fields
        """
        return cls(
            public_id=getattr(result, "public_id", ""),
            linkedin_url=getattr(result, "linkedin_url", ""),
            first_name=getattr(result, "first_name", ""),
            last_name=getattr(result, "last_name", ""),
            full_name=getattr(result, "full_name", ""),
            headline=getattr(result, "headline", ""),
            job_title=getattr(result, "job_title", ""),
            company=getattr(result, "company", ""),
            company_linkedin_url=getattr(result, "company_linkedin_url", ""),
            location=getattr(result, "location", ""),
            about=getattr(result, "about", ""),
            profile_image_url=getattr(result, "profile_image_url", ""),
            match_score=getattr(result, "match_score", 0.0),
            company_domain=getattr(result, "company_domain", ""),
            company_industry=getattr(result, "company_industry", ""),
            connection_count=getattr(result, "connection_count", 0),
        )


@dataclass
class LinkedInOrganization:
    """
    Unified LinkedIn organization/company representation.

    Provides a consistent interface for organization data regardless
    of the lookup source.
    """

    # Core identifiers
    urn_id: str = ""  # LinkedIn URN ID
    public_id: str = ""  # Vanity URL slug
    linkedin_url: str = ""  # Full organization page URL

    # Organization info
    name: str = ""
    page_type: str = "company"  # "company" or "school"
    industry: str = ""
    description: str = ""
    location: str = ""
    employee_count: str = ""
    website: str = ""
    logo_url: str = ""

    def __post_init__(self) -> None:
        """Auto-populate derived fields."""
        if self.public_id and not self.linkedin_url:
            self.linkedin_url = (
                f"https://www.linkedin.com/{self.page_type}/{self.public_id}"
            )

    @property
    def profile_url(self) -> str:
        """Alias for linkedin_url for backward compatibility."""
        return self.linkedin_url

    @property
    def mention_urn(self) -> str:
        """Get URN format for @mentions."""
        return f"urn:li:organization:{self.urn_id}" if self.urn_id else ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching."""
        return {
            "urn_id": self.urn_id,
            "public_id": self.public_id,
            "linkedin_url": self.linkedin_url,
            "name": self.name,
            "page_type": self.page_type,
            "industry": self.industry,
            "description": self.description,
            "location": self.location,
            "employee_count": self.employee_count,
            "website": self.website,
            "logo_url": self.logo_url,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinkedInOrganization":
        """Create from dictionary (cache retrieval)."""
        return cls(
            urn_id=data.get("urn_id", ""),
            public_id=data.get("public_id", ""),
            linkedin_url=data.get("linkedin_url", ""),
            name=data.get("name", ""),
            page_type=data.get("page_type", "company"),
            industry=data.get("industry", ""),
            description=data.get("description", ""),
            location=data.get("location", ""),
            employee_count=data.get("employee_count", ""),
            website=data.get("website", ""),
            logo_url=data.get("logo_url", ""),
        )

    @classmethod
    def from_voyager_org(cls, org: Any) -> "LinkedInOrganization":
        """
        Create from LinkedInOrganization (linkedin_voyager_client.py).

        Args:
            org: A LinkedInOrganization instance from voyager client

        Returns:
            LinkedInOrganization with mapped fields
        """
        return cls(
            urn_id=getattr(org, "urn_id", ""),
            public_id=getattr(org, "public_id", ""),
            linkedin_url=getattr(org, "profile_url", ""),
            name=getattr(org, "name", ""),
            page_type=getattr(org, "page_type", "company"),
            industry=getattr(org, "industry", ""),
            description=getattr(org, "description", ""),
            location=getattr(org, "location", ""),
            employee_count=getattr(org, "employee_count", ""),
        )


# =============================================================================
# Type aliases for backward compatibility
# =============================================================================

# These aliases allow gradual migration from old class names
LinkedInPerson = LinkedInProfile  # Alias for linkedin_voyager_client.py compatibility
LinkedInProfileResult = (
    LinkedInProfile  # Alias for linkedin_rapidapi_client.py compatibility
)
