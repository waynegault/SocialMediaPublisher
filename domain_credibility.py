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


def _create_module_tests() -> bool:
    """Create test suite for domain credibility module."""
    from test_framework import TestSuite

    suite = TestSuite("Domain Credibility", "domain_credibility.py")
    suite.start_suite()

    def test_tier1_nature():
        assert get_domain_tier("https://www.nature.com/article") == 1

    def test_tier1_nih():
        assert get_domain_tier("https://www.nih.gov/news") == 1

    def test_tier2_techcrunch():
        assert get_domain_tier("https://techcrunch.com/2024/01/01/story") == 2

    def test_tier3_latimes():
        assert get_domain_tier("https://www.latimes.com/news") == 3

    def test_low_credibility_dailymail():
        assert get_domain_tier("https://www.dailymail.co.uk/news") == 4

    def test_unknown_domain():
        assert get_domain_tier("https://random-blog.com/post") == 0

    def test_reputable_mit():
        assert is_reputable_domain("https://news.mit.edu/story") is True

    def test_not_reputable_random():
        assert is_reputable_domain("https://randomsite.com/post") is False

    def test_academic_stanford():
        assert is_academic_domain("https://news.stanford.edu/article") is True

    def test_government_nasa():
        assert is_government_domain("https://www.nasa.gov/news") is True

    def test_score_tier1():
        assert get_credibility_score("https://nature.com/article") == 1.0

    def test_score_low_credibility():
        assert get_credibility_score("https://dailymail.co.uk/news") == 0.1

    def test_extract_domain_with_www():
        assert extract_domain("https://www.nature.com/article") == "nature.com"

    def test_extract_domain_without_www():
        assert extract_domain("https://techcrunch.com/news") == "techcrunch.com"

    def test_primary_source_newsroom():
        assert is_primary_source("https://newsroom.google.com/article") is True

    suite.run_test(

        test_name="Tier 1 domain - Nature",

        test_func=test_tier1_nature,

        test_summary="Tier 1 domain behavior with Nature input",

        method_description="Testing Tier 1 domain with Nature input using equality assertions",

        expected_outcome="Tier 1 domain returns the expected value",

    )
    suite.run_test(
        test_name="Tier 1 domain - NIH gov",
        test_func=test_tier1_nih,
        test_summary="Tier 1 domain behavior with NIH gov input",
        method_description="Testing Tier 1 domain with NIH gov input using equality assertions",
        expected_outcome="Tier 1 domain returns the expected value",
    )
    suite.run_test(
        test_name="Tier 2 domain - TechCrunch",
        test_func=test_tier2_techcrunch,
        test_summary="Tier 2 domain behavior with TechCrunch input",
        method_description="Testing Tier 2 domain with TechCrunch input using equality assertions",
        expected_outcome="Tier 2 domain returns the expected value",
    )
    suite.run_test(
        test_name="Tier 3 domain - LA Times",
        test_func=test_tier3_latimes,
        test_summary="Tier 3 domain behavior with LA Times input",
        method_description="Testing Tier 3 domain with LA Times input using equality assertions",
        expected_outcome="Tier 3 domain returns the expected value",
    )
    suite.run_test(
        test_name="Low credibility - Daily Mail",
        test_func=test_low_credibility_dailymail,
        test_summary="Low credibility behavior with Daily Mail input",
        method_description="Testing Low credibility with Daily Mail input using equality assertions",
        expected_outcome="Low credibility returns the expected value",
    )
    suite.run_test(
        test_name="Unknown domain",
        test_func=test_unknown_domain,
        test_summary="Verify Unknown domain produces correct results",
        method_description="Testing Unknown domain using equality assertions",
        expected_outcome="Unknown domain returns the expected value",
    )
    suite.run_test(
        test_name="is_reputable_domain - MIT",
        test_func=test_reputable_mit,
        test_summary="is_reputable_domain behavior with MIT input",
        method_description="Testing is_reputable_domain with MIT input using boolean return verification",
        expected_outcome="Function returns True for MIT input",
    )
    suite.run_test(
        test_name="is_reputable_domain - random site",
        test_func=test_not_reputable_random,
        test_summary="is_reputable_domain behavior with random site input",
        method_description="Testing is_reputable_domain with random site input using boolean return verification",
        expected_outcome="Function returns False for random site input",
    )
    suite.run_test(
        test_name="is_academic_domain - Stanford",
        test_func=test_academic_stanford,
        test_summary="is_academic_domain behavior with Stanford input",
        method_description="Testing is_academic_domain with Stanford input using boolean return verification",
        expected_outcome="Function returns True for Stanford input",
    )
    suite.run_test(
        test_name="is_government_domain - NASA",
        test_func=test_government_nasa,
        test_summary="is_government_domain behavior with NASA input",
        method_description="Testing is_government_domain with NASA input using boolean return verification",
        expected_outcome="Function returns True for NASA input",
    )
    suite.run_test(
        test_name="credibility_score - Tier 1",
        test_func=test_score_tier1,
        test_summary="credibility_score behavior with Tier 1 input",
        method_description="Testing credibility_score with Tier 1 input using equality assertions",
        expected_outcome="credibility_score returns the expected value",
    )
    suite.run_test(
        test_name="credibility_score - Low credibility",
        test_func=test_score_low_credibility,
        test_summary="credibility_score behavior with Low credibility input",
        method_description="Testing credibility_score with Low credibility input using equality assertions",
        expected_outcome="credibility_score returns the expected value",
    )
    suite.run_test(
        test_name="extract_domain - with www",
        test_func=test_extract_domain_with_www,
        test_summary="extract_domain behavior with with www input",
        method_description="Testing extract_domain with with www input using equality assertions",
        expected_outcome="extract_domain returns the expected value",
    )
    suite.run_test(
        test_name="extract_domain - without www",
        test_func=test_extract_domain_without_www,
        test_summary="extract_domain behavior with without www input",
        method_description="Testing extract_domain with without www input using equality assertions",
        expected_outcome="extract_domain returns the expected value",
    )
    suite.run_test(
        test_name="is_primary_source - newsroom",
        test_func=test_primary_source_newsroom,
        test_summary="is_primary_source behavior with newsroom input",
        method_description="Testing is_primary_source with newsroom input using boolean return verification",
        expected_outcome="Function returns True for newsroom input",
    )
    return suite.finish_suite()


if __name__ == "__main__":
    _create_module_tests()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)
