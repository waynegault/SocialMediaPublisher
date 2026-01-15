"""Centralized domain credibility management for Social Media Publisher.

This module provides a single source of truth for domain credibility tiers,
replacing duplicate domain lists that were previously in searcher.py and
source_verifier.py.

Usage:
    from domain_credibility import (
        TIER_1_DOMAINS,
        TIER_2_DOMAINS,
        TIER_3_DOMAINS,
        ALL_REPUTABLE_DOMAINS,
        LOW_CREDIBILITY_DOMAINS,
        get_domain_tier,
        is_reputable_domain,
    )
"""

from typing import Literal
from urllib.parse import urlparse


# Tier 1: Most authoritative sources
# Scientific publishers, major news organizations, top universities, government
TIER_1_DOMAINS = frozenset(
    {
        # Major scientific publishers
        "nature.com",
        "science.org",
        "sciencedirect.com",
        "ieee.org",
        "springer.com",
        "wiley.com",
        "acs.org",
        "rsc.org",
        "elsevier.com",
        "cell.com",
        # Major news organizations
        "nytimes.com",
        "washingtonpost.com",
        "bbc.com",
        "bbc.co.uk",
        "reuters.com",
        "apnews.com",
        "bloomberg.com",
        "wsj.com",
        "economist.com",
        "ft.com",
        "theguardian.com",
        # Top universities
        "mit.edu",
        "stanford.edu",
        "harvard.edu",
        "berkeley.edu",
        "caltech.edu",
        "princeton.edu",
        "yale.edu",
        "columbia.edu",
        "cam.ac.uk",
        "ox.ac.uk",
        "ethz.ch",
        "mpg.de",
        # Government / Research institutions
        "nih.gov",
        "nasa.gov",
        "doe.gov",
        "energy.gov",
        "noaa.gov",
        "nist.gov",
        "arxiv.org",
        "pubmed.gov",
    }
)

# Tier 2: High-quality but less authoritative sources
# Quality tech publications, industry publications, other reputable universities
TIER_2_DOMAINS = frozenset(
    {
        # Quality tech publications
        "techcrunch.com",
        "wired.com",
        "arstechnica.com",
        "theverge.com",
        "engadget.com",
        "zdnet.com",
        "cnet.com",
        "venturebeat.com",
        # Industry publications
        "forbes.com",
        "businessinsider.com",
        "fortune.com",
        "cnbc.com",
        # Energy/sustainability specific
        "greentechmedia.com",
        "cleantechnica.com",
        "renewableenergyworld.com",
        "pv-magazine.com",
        "hydrogeninsight.com",
        "rechargenews.com",
        "spglobal.com",
        "woodmac.com",
        # Other reputable universities
        "cornell.edu",
        "uchicago.edu",
        "upenn.edu",
        "duke.edu",
        "northwestern.edu",
        "ucla.edu",
        "umich.edu",
        "gatech.edu",
        "cmu.edu",
        "utexas.edu",
    }
)

# Tier 3: Acceptable but less reliable sources
# Company blogs/newsrooms (primary sources but biased), regional news, international
TIER_3_DOMAINS = frozenset(
    {
        # Regional news
        "latimes.com",
        "chicagotribune.com",
        "bostonglobe.com",
        "sfchronicle.com",
        # International quality sources
        "lemonde.fr",
        "spiegel.de",
        "zeit.de",
        "nikkei.com",
        "scmp.com",
    }
)

# Subdomain patterns for tier 3 (company blogs, newsrooms)
TIER_3_SUBDOMAIN_PATTERNS = frozenset(
    {
        "newsroom.",
        "blog.",
        "press.",
        "media.",
    }
)

# Domains that reduce credibility (tabloids, known misinformation, etc.)
LOW_CREDIBILITY_DOMAINS = frozenset(
    {
        "dailymail.co.uk",
        "thesun.co.uk",
        "nypost.com",
        "breitbart.com",
        "infowars.com",
        "naturalnews.com",
    }
)

# Combined set of all reputable domains (tiers 1-3)
ALL_REPUTABLE_DOMAINS = TIER_1_DOMAINS | TIER_2_DOMAINS | TIER_3_DOMAINS


def extract_domain(url: str) -> str:
    """
    Extract and normalize domain from a URL.

    Args:
        url: Full URL string

    Returns:
        Lowercase domain without www. prefix
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def get_domain_tier(url: str) -> Literal[0, 1, 2, 3, 4]:
    """
    Get the credibility tier for a domain.

    Args:
        url: Full URL string

    Returns:
        1 = Tier 1 (most authoritative)
        2 = Tier 2 (high quality)
        3 = Tier 3 (acceptable)
        4 = Low credibility (tabloids, misinformation)
        0 = Unknown/unranked
    """
    domain = extract_domain(url)
    if not domain:
        return 0

    # Check domain and parent domains
    domain_parts = domain.split(".")
    for i in range(len(domain_parts) - 1):
        check_domain = ".".join(domain_parts[i:])

        # Check for low credibility first
        if check_domain in LOW_CREDIBILITY_DOMAINS:
            return 4

        if check_domain in TIER_1_DOMAINS:
            return 1

        if check_domain in TIER_2_DOMAINS:
            return 2

        if check_domain in TIER_3_DOMAINS:
            return 3

    # Check for academic/government TLDs
    if domain.endswith(".edu") or domain.endswith(".ac.uk"):
        return 2  # Default academic to tier 2

    if domain.endswith(".gov") or domain.endswith(".gov.uk"):
        return 1  # Government sources are tier 1

    # Check for tier 3 subdomain patterns (newsroom., blog., etc.)
    # Only match patterns at the START of the domain (e.g., blog.company.com)
    for pattern in TIER_3_SUBDOMAIN_PATTERNS:
        if domain.startswith(pattern):
            return 3

    return 0  # Unknown


def is_reputable_domain(url: str) -> bool:
    """
    Check if a URL is from a reputable domain (tier 1, 2, or 3).

    Args:
        url: Full URL string

    Returns:
        True if the domain is in any reputable tier
    """
    tier = get_domain_tier(url)
    return tier in (1, 2, 3)


def get_credibility_score(url: str) -> float:
    """
    Get a numeric credibility score for a domain.

    Args:
        url: Full URL string

    Returns:
        Float between 0.0 and 1.0
        1.0 = Tier 1 (most credible)
        0.8 = Tier 2
        0.6 = Tier 3
        0.3 = Unknown
        0.1 = Low credibility
    """
    tier = get_domain_tier(url)
    tier_scores = {
        1: 1.0,
        2: 0.8,
        3: 0.6,
        4: 0.1,  # Low credibility
        0: 0.3,  # Unknown
    }
    return tier_scores.get(tier, 0.3)


def is_academic_domain(url: str) -> bool:
    """Check if URL is from an academic institution."""
    domain = extract_domain(url)
    return domain.endswith(".edu") or domain.endswith(".ac.uk")


def is_government_domain(url: str) -> bool:
    """Check if URL is from a government source."""
    domain = extract_domain(url)
    return domain.endswith(".gov") or domain.endswith(".gov.uk")


def is_primary_source(url: str) -> bool:
    """Check if URL is a primary source (company newsroom, blog, etc.).
    
    Only matches subdomain patterns at the START of the domain.
    e.g., "newsroom.company.com" matches, but "random-blog.xyz" does not.
    """
    domain = extract_domain(url)
    return any(domain.startswith(pattern) for pattern in TIER_3_SUBDOMAIN_PATTERNS)


# ============================================================================
# Module Tests
# ============================================================================


def _create_module_tests():
    """Create test suite for domain credibility module."""
    from test_framework import TestSuite

    suite = TestSuite("Domain Credibility")

    # Test tier 1 domains
    suite.add_test(
        "Tier 1 domain - Nature",
        lambda: get_domain_tier("https://www.nature.com/article") == 1,
    )

    suite.add_test(
        "Tier 1 domain - NIH gov",
        lambda: get_domain_tier("https://www.nih.gov/news") == 1,
    )

    # Test tier 2 domains
    suite.add_test(
        "Tier 2 domain - TechCrunch",
        lambda: get_domain_tier("https://techcrunch.com/2024/01/01/story") == 2,
    )

    # Test tier 3 domains
    suite.add_test(
        "Tier 3 domain - LA Times",
        lambda: get_domain_tier("https://www.latimes.com/news") == 3,
    )

    # Test low credibility
    suite.add_test(
        "Low credibility - Daily Mail",
        lambda: get_domain_tier("https://www.dailymail.co.uk/news") == 4,
    )

    # Test unknown domain
    suite.add_test(
        "Unknown domain",
        lambda: get_domain_tier("https://random-blog.com/post") == 0,
    )

    # Test is_reputable_domain
    suite.add_test(
        "is_reputable_domain - MIT",
        lambda: is_reputable_domain("https://news.mit.edu/story") is True,
    )

    suite.add_test(
        "is_reputable_domain - random site",
        lambda: is_reputable_domain("https://randomsite.com/post") is False,
    )

    # Test academic domain detection
    suite.add_test(
        "is_academic_domain - Stanford",
        lambda: is_academic_domain("https://news.stanford.edu/article") is True,
    )

    # Test government domain detection
    suite.add_test(
        "is_government_domain - NASA",
        lambda: is_government_domain("https://www.nasa.gov/news") is True,
    )

    # Test credibility score
    suite.add_test(
        "credibility_score - Tier 1",
        lambda: get_credibility_score("https://nature.com/article") == 1.0,
    )

    suite.add_test(
        "credibility_score - Low credibility",
        lambda: get_credibility_score("https://dailymail.co.uk/news") == 0.1,
    )

    # Test extract_domain
    suite.add_test(
        "extract_domain - with www",
        lambda: extract_domain("https://www.nature.com/article") == "nature.com",
    )

    suite.add_test(
        "extract_domain - without www",
        lambda: extract_domain("https://techcrunch.com/news") == "techcrunch.com",
    )

    # Test primary source detection
    suite.add_test(
        "is_primary_source - newsroom",
        lambda: is_primary_source("https://newsroom.google.com/article") is True,
    )

    return suite


if __name__ == "__main__":
    # Run tests when executed directly
    suite = _create_module_tests()
    from test_framework import run_all_suites

    run_all_suites([suite], verbose=True)
