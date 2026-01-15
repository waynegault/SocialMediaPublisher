"""
Organization Aliases Module.

Provides centralized mappings of organization name abbreviations and variations
to their canonical forms. This prevents duplicate searches and lookups for
organizations known by multiple names.

Includes:
- University abbreviations (MIT, UCLA, CMU, etc.)
- Company aliases (AWS, MSFT, Meta/Facebook, etc.)
- Research institution mappings (Argonne, etc.)
"""

from typing import Optional

# =============================================================================
# Organization Name Aliases
# =============================================================================

# Maps common abbreviations/variations to canonical organization names
# This prevents duplicate searches for "UChicago" vs "University of Chicago"
ORG_ALIASES: dict[str, str] = {
    # Universities - abbreviations
    "uchicago": "university of chicago",
    "u of c": "university of chicago",
    "mit": "massachusetts institute of technology",
    "caltech": "california institute of technology",
    "cal tech": "california institute of technology",
    "stanford": "stanford university",
    "harvard": "harvard university",
    "yale": "yale university",
    "princeton": "princeton university",
    "columbia": "columbia university",
    "cornell": "cornell university",
    "upenn": "university of pennsylvania",
    "penn": "university of pennsylvania",
    "berkeley": "university of california berkeley",
    "uc berkeley": "university of california berkeley",
    "ucla": "university of california los angeles",
    "usc": "university of southern california",
    "nyu": "new york university",
    "ut austin": "university of texas at austin",
    "ut": "university of texas at austin",
    "gatech": "georgia institute of technology",
    "georgia tech": "georgia institute of technology",
    "purdue": "purdue university",
    "umich": "university of michigan",
    "u michigan": "university of michigan",
    "northwestern": "northwestern university",
    "vanderbilt": "vanderbilt university",
    "vandy": "vanderbilt university",
    "iowa state": "iowa state university",
    "penn state": "pennsylvania state university",
    "ohio state": "ohio state university",
    "osu": "ohio state university",
    "uva": "university of virginia",
    "unc": "university of north carolina",
    "duke": "duke university",
    "rice": "rice university",
    "cmu": "carnegie mellon university",
    "carnegie mellon": "carnegie mellon university",
    "jhu": "johns hopkins university",
    "johns hopkins": "johns hopkins university",
    "ucl": "university college london",
    "oxford": "university of oxford",
    "cambridge": "university of cambridge",
    "eth": "eth zurich",
    "epfl": "ecole polytechnique federale de lausanne",
    "tsinghua": "tsinghua university",
    "peking": "peking university",
    "pku": "peking university",
    # UChicago variations (Pritzker School etc.)
    "uchicago pritzker school of molecular engineering": "university of chicago",
    "pritzker school of molecular engineering": "university of chicago",
    "pritzker molecular engineering": "university of chicago",
    # University of Illinois variations
    "uic": "university of illinois chicago",
    "uiuc": "university of illinois urbana champaign",
    "u of i": "university of illinois urbana champaign",
    "university of illinois at chicago": "university of illinois chicago",
    "university of illinois at urbana champaign": "university of illinois urbana champaign",
    "university of illinois at urbana-champaign": "university of illinois urbana champaign",
    # National Labs
    "anl": "argonne national laboratory",
    "argonne": "argonne national laboratory",
    "llnl": "lawrence livermore national laboratory",
    "lbnl": "lawrence berkeley national laboratory",
    "lanl": "los alamos national laboratory",
    "ornl": "oak ridge national laboratory",
    "pnnl": "pacific northwest national laboratory",
    "sandia": "sandia national laboratories",
    "nrel": "national renewable energy laboratory",
    "inl": "idaho national laboratory",
    "fermilab": "fermi national accelerator laboratory",
    "slac": "stanford linear accelerator center",
    # Companies - Tech Giants
    "google": "google llc",
    "meta": "meta platforms",
    "facebook": "meta platforms",
    "fb": "meta platforms",
    "amazon": "amazon.com",
    "aws": "amazon web services",
    "microsoft": "microsoft corporation",
    "msft": "microsoft corporation",
    "apple": "apple inc",
    "ibm": "international business machines",
    "intel": "intel corporation",
    "nvidia": "nvidia corporation",
    "amd": "advanced micro devices",
    "tesla": "tesla inc",
    "spacex": "space exploration technologies",
    # Companies - Enterprise/Consulting
    "ge": "general electric",
    "gm": "general motors",
    "jpmorgan": "jpmorgan chase",
    "jp morgan": "jpmorgan chase",
    "goldman": "goldman sachs",
    "gs": "goldman sachs",
    "mckinsey": "mckinsey & company",
    "mck": "mckinsey & company",
    "bcg": "boston consulting group",
    "bain": "bain & company",
    "deloitte": "deloitte consulting",
    "pwc": "pricewaterhousecoopers",
    "ey": "ernst & young",
    "kpmg": "kpmg international",
    "accenture": "accenture plc",
}

# Reverse mapping: canonical name -> list of aliases
_CANONICAL_TO_ALIASES: Optional[dict[str, list[str]]] = None


def get_canonical_name(org_name: str) -> str:
    """
    Get the canonical organization name for an alias.

    Args:
        org_name: Organization name or alias

    Returns:
        Canonical name if found, otherwise the original name (lowercased)
    """
    normalized = org_name.lower().strip()
    return ORG_ALIASES.get(normalized, normalized)


def get_aliases(canonical_name: str) -> list[str]:
    """
    Get all aliases for a canonical organization name.

    Args:
        canonical_name: The canonical name to look up

    Returns:
        List of aliases including the canonical name itself
    """
    global _CANONICAL_TO_ALIASES

    if _CANONICAL_TO_ALIASES is None:
        # Build reverse mapping on first call
        _CANONICAL_TO_ALIASES = {}
        for alias, canonical in ORG_ALIASES.items():
            if canonical not in _CANONICAL_TO_ALIASES:
                _CANONICAL_TO_ALIASES[canonical] = [canonical]
            _CANONICAL_TO_ALIASES[canonical].append(alias)

    normalized = canonical_name.lower().strip()
    # Check if this is an alias first
    if normalized in ORG_ALIASES:
        normalized = ORG_ALIASES[normalized]

    return _CANONICAL_TO_ALIASES.get(normalized, [normalized])


def is_known_organization(org_name: str) -> bool:
    """
    Check if an organization name (or alias) is in the known aliases.

    Args:
        org_name: Organization name to check

    Returns:
        True if found in aliases, False otherwise
    """
    normalized = org_name.lower().strip()
    if normalized in ORG_ALIASES:
        return True
    # Also check if it's a canonical name
    return normalized in set(ORG_ALIASES.values())


__all__ = [
    "ORG_ALIASES",
    "get_canonical_name",
    "get_aliases",
    "is_known_organization",
]
