"""Named Entity Recognition Engine for Social Media Publisher.

This module provides NER capabilities for extracting and disambiguating
entities (people, companies, organizations) from news stories to improve
@mention reliability on LinkedIn.

Features:
- spaCy-based NER with custom domain patterns
- Entity disambiguation using knowledge graphs
- Company/person cache with fuzzy matching
- Chemical engineering domain-specific rules

TASK 1.4: Entity Extraction & Named Entity Recognition
"""

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Optional, Sequence

logger = logging.getLogger(__name__)

# Type hints for optional dependencies
if TYPE_CHECKING:
    import spacy
    from spacy.tokens import Doc, Span
    from spacy.language import Language
    from sentence_transformers import SentenceTransformer
    import numpy as np

# Optional spaCy import - gracefully degrade if not installed
try:
    import spacy
    from spacy.tokens import Doc, Span
    from spacy.language import Language

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None  # type: ignore[assignment]
    Doc = None  # type: ignore[assignment, misc]
    Span = None  # type: ignore[assignment, misc]
    Language = None  # type: ignore[assignment, misc]
    logger.warning(
        "spaCy not installed - pip install spacy && python -m spacy download en_core_web_sm"
    )

# Optional sentence-transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore[assignment, misc]
    np = None  # type: ignore[assignment]
    logger.debug("sentence-transformers not available for semantic similarity")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Entity:
    """Represents an extracted entity."""

    text: str  # Original text as found
    label: str  # Entity type: PERSON, ORG, COMPANY, TITLE, etc.
    start_char: int = 0  # Character offset in source text
    end_char: int = 0
    confidence: float = 1.0  # Confidence score 0-1
    normalized: str = ""  # Normalized/canonical form
    metadata: dict = field(default_factory=dict)  # Additional info

    def __post_init__(self) -> None:
        if not self.normalized:
            self.normalized = self.text


@dataclass
class PersonEntity(Entity):
    """A person entity with additional fields."""

    title: str = ""  # Job title if found
    affiliation: str = ""  # Company/organization
    linkedin_profile: str = ""
    linkedin_urn: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.label = "PERSON"


@dataclass
class OrganizationEntity(Entity):
    """An organization/company entity."""

    org_type: str = ""  # company, university, government, ngo, etc.
    industry: str = ""
    headquarters: str = ""
    linkedin_page: str = ""
    linkedin_urn: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.label = "ORG"


@dataclass
class ExtractionResult:
    """Results from entity extraction."""

    persons: list[PersonEntity] = field(default_factory=list)
    organizations: list[OrganizationEntity] = field(default_factory=list)
    all_entities: list[Entity] = field(default_factory=list)
    text: str = ""
    confidence_score: float = 0.0

    @property
    def mention_candidates(self) -> list[PersonEntity | OrganizationEntity]:
        """Get entities suitable for @mentions (people and orgs with profiles)."""
        candidates: list[PersonEntity | OrganizationEntity] = []
        candidates.extend(self.persons)
        candidates.extend(self.organizations)
        return sorted(candidates, key=lambda e: e.confidence, reverse=True)


# =============================================================================
# Chemical Engineering Domain Patterns
# =============================================================================


# Common chemical engineering terms that might be confused with entities
CHEM_ENG_TERMS = {
    "process engineering",
    "chemical engineering",
    "process safety",
    "reaction engineering",
    "separation processes",
    "thermodynamics",
    "fluid dynamics",
    "heat transfer",
    "mass transfer",
    "unit operations",
    "catalysis",
    "polymerization",
    "distillation",
    "crystallization",
    "filtration",
    "drying",
    "extraction",
    "absorption",
    "adsorption",
    "reactor design",
    "process control",
    "process optimization",
    "plant design",
    "piping",
    "instrumentation",
    "hazop",
    "p&id",
    "pfd",
}

# Industry-specific company suffixes
COMPANY_SUFFIXES = {
    "ltd",
    "limited",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "llc",
    "llp",
    "plc",
    "gmbh",
    "ag",
    "sa",
    "nv",
    "bv",
    "pty",
    "co",
    "company",
    "group",
    "holdings",
    "industries",
    "chemicals",
    "petrochemicals",
    "energy",
    "engineering",
    "technologies",
}

# Academic/research institution patterns
ACADEMIC_PATTERNS = {
    "university",
    "college",
    "institute",
    "laboratory",
    "lab",
    "center",
    "centre",
    "school",
    "faculty",
    "department",
    "research",
    "mit",  # Massachusetts Institute of Technology
    "caltech",  # California Institute of Technology
}

# Professional titles that indicate a person
PROFESSIONAL_TITLES = {
    "dr",
    "dr.",
    "prof",
    "prof.",
    "professor",
    "mr",
    "mr.",
    "mrs",
    "mrs.",
    "ms",
    "ms.",
    "sir",
    "dame",
    "ceo",
    "cto",
    "cfo",
    "coo",
    "president",
    "chairman",
    "chairwoman",
    "director",
    "manager",
    "engineer",
    "scientist",
    "researcher",
    "analyst",
    "vp",
    "vice president",
    "head",
    "chief",
    "lead",
    "senior",
    "principal",
    "fellow",
}

# Major chemical/energy companies (known entities for disambiguation)
KNOWN_COMPANIES = {
    "basf": "BASF SE",
    "dow": "Dow Inc.",
    "dupont": "DuPont",
    "shell": "Shell plc",
    "bp": "BP plc",
    "exxon": "ExxonMobil",
    "exxonmobil": "ExxonMobil",
    "chevron": "Chevron Corporation",
    "total": "TotalEnergies",
    "totalenergies": "TotalEnergies",
    "sinopec": "Sinopec",
    "petrochina": "PetroChina",
    "sabic": "SABIC",
    "lyondellbasell": "LyondellBasell",
    "ineos": "INEOS",
    "linde": "Linde plc",
    "air liquide": "Air Liquide",
    "praxair": "Linde plc",  # Merged with Linde
    "honeywell": "Honeywell",
    "emerson": "Emerson Electric",
    "siemens": "Siemens AG",
    "abb": "ABB Ltd",
    "yokogawa": "Yokogawa Electric",
    "aspentech": "AspenTech",
    "aveva": "AVEVA",
    "icheme": "Institution of Chemical Engineers",
    "aiche": "American Institute of Chemical Engineers",
    "rsc": "Royal Society of Chemistry",
    "acs": "American Chemical Society",
}

# Known universities for disambiguation
KNOWN_UNIVERSITIES = {
    "mit": "Massachusetts Institute of Technology",
    "stanford": "Stanford University",
    "berkeley": "University of California, Berkeley",
    "caltech": "California Institute of Technology",
    "cambridge": "University of Cambridge",
    "oxford": "University of Oxford",
    "imperial": "Imperial College London",
    "eth": "ETH Zurich",
    "delft": "Delft University of Technology",
    "tu delft": "Delft University of Technology",
    "manchester": "University of Manchester",
    "ucl": "University College London",
    "georgia tech": "Georgia Institute of Technology",
    "purdue": "Purdue University",
    "texas a&m": "Texas A&M University",
}

# =============================================================================
# Invalid Entity Names (shared across modules)
# =============================================================================
# These are terms that appear as "affiliations" or "names" but aren't real entities.
# Used by linkedin_profile_lookup.py and company_mention_enricher.py

INVALID_ORG_NAMES: set[str] = {
    # Generic terms / research topics (not organizations)
    "molecular",
    "chemistry",
    "physics",
    "biology",
    "research",
    "science",
    "engineering",
    "technology",
    "materials",
    "nanotechnology",
    "computational",
    "theoretical",
    "applied",
    "advanced",
    "fundamental",
    # Materials and topics (not organizations)
    "mxenes",
    "graphene",
    "nanoparticles",
    "nanomaterials",
    "quantum",
    "polymers",
    "catalysis",
    "electrochemistry",
    "spectroscopy",
    "synthesis",
    # Government entities (too generic)
    "government",
    "the government",
    "uk government",
    "us government",
    "federal government",
    "state government",
    "local government",
    "ministry",
    "department",
    "agency",
    # Retail (often false positives from articles)
    "aldi",
    "asda",
    "tesco",
    "sainsburys",
    "morrisons",
    "lidl",
    "waitrose",
    "co-op",
    "walmart",
    "target",
    "costco",
    "supermarket",
    "retail",
    "store",
    "shop",
    # Common Asian surnames (false positives from parsing)
    "jiang",
    "wang",
    "zhang",
    "chen",
    "liu",
    "li",
    "yang",
    "huang",
    "zhou",
    "wu",
    # Other invalid patterns
    "n/a",
    "none",
    "unknown",
    "various",
    "multiple",
    "other",
    "others",
    "tba",
    "independent",
    "freelance",
    "self-employed",
    "retired",
}

# Patterns that indicate a headline/description rather than an org name
INVALID_ORG_PATTERNS: list[str] = [
    r"^new\s+",  # Headlines often start with "New ..."
    r"smash",  # "Smashes", "Smashing" - headline verbs
    r"breakthrough",
    r"discover",
    r"announc",  # "Announces", "Announced"
    r"reveal",
    r"launch",
    r"unveil",
    r"develop",
    r"creat",  # "Creates", "Created"
    r"powered",  # "Gold-Powered" etc.
    r"benchmark",
    r"record",
    r"-old\b",  # "decade-old", "year-old"
]

# Generic placeholders that aren't actual person names
INVALID_PERSON_NAMES: set[str] = {
    "individual researcher",
    "researcher",
    "professor",
    "scientist",
    "engineer",
    "author",
    "contributor",
    "correspondent",
    "editor",
    "staff",
    "staff writer",
    "team",
    "anonymous",
    "unknown",
}

# Single-word org abbreviations that ARE valid (exceptions)
VALID_SINGLE_WORD_ORGS: set[str] = {
    "mit",
    "nasa",
    "ibm",
    "gsk",
    "rspca",
    "bva",
    "basf",
    "dow",
    "shell",
    "bp",
    "sabic",
    "ineos",
    "linde",
}


# =============================================================================
# NER Engine
# =============================================================================


class NEREngine:
    """Named Entity Recognition engine for story content.

    Uses spaCy for base NER with custom rules for the chemical engineering
    domain. Falls back to regex-based extraction if spaCy is not available.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        use_transformers: bool = False,
    ) -> None:
        """Initialize the NER engine.

        Args:
            model_name: spaCy model to use (default: en_core_web_sm)
            use_transformers: Whether to use sentence-transformers for similarity
        """
        self.model_name = model_name
        self.nlp: Optional[Any] = None  # spacy.Language when available
        self.similarity_model: Optional[Any] = (
            None  # SentenceTransformer when available
        )

        # Entity cache for fuzzy matching
        self._entity_cache: dict[str, Entity] = {}

        # Initialize spaCy if available
        if SPACY_AVAILABLE and spacy is not None:
            try:
                self.nlp = spacy.load(model_name)
                self._add_custom_patterns()
                logger.info(f"NER engine initialized with spaCy model: {model_name}")
            except OSError:
                logger.warning(
                    f"spaCy model '{model_name}' not found. "
                    f"Run: python -m spacy download {model_name}"
                )
        else:
            logger.warning("spaCy not available - using regex fallback")

        # Initialize transformer model for semantic similarity
        if (
            use_transformers
            and TRANSFORMERS_AVAILABLE
            and SentenceTransformer is not None
        ):
            try:
                self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Sentence transformer loaded for entity similarity")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")

    def _add_custom_patterns(self) -> None:
        """Add custom entity patterns to the spaCy pipeline."""
        if not self.nlp:
            return

        # Add entity ruler for domain-specific patterns
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        else:
            ruler = self.nlp.get_pipe("entity_ruler")

        patterns = []

        # Add known companies
        for short, full in KNOWN_COMPANIES.items():
            patterns.append({"label": "ORG", "pattern": short.upper()})
            patterns.append({"label": "ORG", "pattern": short.title()})
            patterns.append({"label": "ORG", "pattern": full})

        # Add known universities
        for short, full in KNOWN_UNIVERSITIES.items():
            patterns.append({"label": "ORG", "pattern": short.upper()})
            patterns.append({"label": "ORG", "pattern": full})

        # Add professional body patterns
        patterns.extend(
            [
                {"label": "ORG", "pattern": "IChemE"},
                {"label": "ORG", "pattern": "AIChE"},
                {"label": "ORG", "pattern": "Institution of Chemical Engineers"},
                {"label": "ORG", "pattern": "American Institute of Chemical Engineers"},
            ]
        )

        # Add title + name patterns for better person detection
        for title in ["Dr.", "Prof.", "Professor", "CEO", "CTO", "Director"]:
            patterns.append(
                {
                    "label": "PERSON",
                    "pattern": [
                        {"TEXT": title},
                        {"IS_TITLE": True},
                        {"IS_TITLE": True},
                    ],
                }
            )

        ruler.add_patterns(patterns)

    def extract_entities(self, text: str) -> ExtractionResult:
        """Extract entities from text.

        Args:
            text: Text to extract entities from

        Returns:
            ExtractionResult with extracted entities
        """
        if not text:
            return ExtractionResult(text=text)

        if self.nlp:
            return self._extract_with_spacy(text)
        else:
            return self._extract_with_regex(text)

    def _extract_with_spacy(self, text: str) -> ExtractionResult:
        """Extract entities using spaCy NER."""
        assert self.nlp is not None, "spaCy nlp object not initialized"
        doc = self.nlp(text)
        result = ExtractionResult(text=text)

        for ent in doc.ents:
            entity = self._process_spacy_entity(ent)
            if entity:
                result.all_entities.append(entity)
                if isinstance(entity, PersonEntity):
                    result.persons.append(entity)
                elif isinstance(entity, OrganizationEntity):
                    result.organizations.append(entity)

        # Post-process to find additional context (titles, affiliations)
        self._enrich_entities(result, doc)

        # Calculate confidence score
        if result.all_entities:
            result.confidence_score = sum(
                e.confidence for e in result.all_entities
            ) / len(result.all_entities)

        return result

    def _process_spacy_entity(self, ent: Any) -> Optional[Entity]:
        """Process a spaCy entity span into our Entity type."""
        text = ent.text.strip()

        # Skip single characters or very short entities
        if len(text) < 2:
            return None

        # Skip common chemical engineering terms mistaken as entities
        if text.lower() in CHEM_ENG_TERMS:
            return None

        if ent.label_ == "PERSON":
            person = PersonEntity(
                text=text,
                label="PERSON",
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=0.85,
            )
            # Check for title prefix
            person.title = self._extract_title_from_context(ent)
            return person

        elif ent.label_ in ("ORG", "COMPANY", "GPE"):
            # Determine organization type
            org_type = self._classify_org_type(text)

            # Normalize known entities
            normalized = self._normalize_organization(text)

            org = OrganizationEntity(
                text=text,
                label="ORG",
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=0.8,
                normalized=normalized,
                org_type=org_type,
            )
            return org

        return None

    def _extract_title_from_context(self, ent: Any) -> str:
        """Extract job title from surrounding context."""
        # Look at tokens before the entity
        doc = ent.doc
        start_idx = ent.start

        # Check 3 tokens before for title indicators
        title_parts = []
        for i in range(max(0, start_idx - 3), start_idx):
            token = doc[i]
            if token.text.lower().rstrip(".,") in PROFESSIONAL_TITLES:
                title_parts.append(token.text)

        return " ".join(title_parts)

    def _classify_org_type(self, text: str) -> str:
        """Classify an organization's type."""
        text_lower = text.lower()

        # Check for academic patterns
        for pattern in ACADEMIC_PATTERNS:
            if pattern in text_lower:
                return "university"

        # Check for company patterns
        for suffix in COMPANY_SUFFIXES:
            if text_lower.endswith(suffix) or f" {suffix}" in text_lower:
                return "company"

        # Check known companies
        if text_lower in KNOWN_COMPANIES:
            return "company"

        # Check known universities
        if text_lower in KNOWN_UNIVERSITIES:
            return "university"

        return "organization"

    def _normalize_organization(self, text: str) -> str:
        """Normalize an organization name to its canonical form."""
        text_lower = text.lower().strip()

        # Check known mappings
        if text_lower in KNOWN_COMPANIES:
            return KNOWN_COMPANIES[text_lower]

        if text_lower in KNOWN_UNIVERSITIES:
            return KNOWN_UNIVERSITIES[text_lower]

        # Remove common suffixes for matching
        for suffix in ["ltd", "limited", "inc", "corp", "plc"]:
            if text_lower.endswith(f" {suffix}"):
                base = text_lower[: -(len(suffix) + 1)]
                if base in KNOWN_COMPANIES:
                    return KNOWN_COMPANIES[base]

        return text

    def _enrich_entities(self, result: ExtractionResult, doc: Any) -> None:
        """Enrich entities with additional context from the document."""
        # Find person-organization associations
        for person in result.persons:
            # Look for "of [ORG]" or "at [ORG]" patterns after person name
            person_end = person.end_char
            context_window = result.text[person_end : person_end + 100]

            # Pattern: ", CEO of BASF" or "at MIT"
            affiliation_match = re.search(
                r"(?:,?\s*(?:of|at|from|with)\s+)([A-Z][A-Za-z\s&]+?)(?:[,.\s]|$)",
                context_window,
            )
            if affiliation_match:
                person.affiliation = affiliation_match.group(1).strip()

            # Look for title patterns after comma
            title_match = re.search(
                r",\s*((?:chief|senior|lead|principal|head|director|manager|vp|vice president|president|ceo|cto|cfo)[^,\.]*)",
                context_window,
                re.IGNORECASE,
            )
            if title_match and not person.title:
                person.title = title_match.group(1).strip()

    def _extract_with_regex(self, text: str) -> ExtractionResult:
        """Fallback regex-based entity extraction."""
        result = ExtractionResult(text=text)

        # Extract persons (Title + Name pattern)
        person_pattern = re.compile(
            r"\b((?:Dr\.?|Prof\.?|Professor|Mr\.?|Mrs\.?|Ms\.?)\s+)?"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
        )

        for match in person_pattern.finditer(text):
            title = (match.group(1) or "").strip()
            name = match.group(2).strip()

            # Skip if it looks like a company
            if any(
                name.lower().endswith(suffix)
                for suffix in ["inc", "ltd", "corp", "llc"]
            ):
                continue

            person = PersonEntity(
                text=name,
                label="PERSON",
                start_char=match.start(2),
                end_char=match.end(2),
                confidence=0.6,
                title=title.rstrip("."),
            )
            result.persons.append(person)
            result.all_entities.append(person)

        # Extract organizations
        org_pattern = re.compile(
            r"\b([A-Z][A-Za-z&\s]+(?:Inc\.?|Ltd\.?|Corp\.?|LLC|University|Institute|"
            r"Laboratory|College|Company|Group|Holdings|Technologies))\b"
        )

        for match in org_pattern.finditer(text):
            org_text = match.group(1).strip()

            org = OrganizationEntity(
                text=org_text,
                label="ORG",
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.5,
                normalized=self._normalize_organization(org_text),
                org_type=self._classify_org_type(org_text),
            )
            result.organizations.append(org)
            result.all_entities.append(org)

        # Look for known companies by name
        for short, full in KNOWN_COMPANIES.items():
            pattern = re.compile(rf"\b{re.escape(short)}\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                # Avoid duplicates
                if not any(e.start_char == match.start() for e in result.organizations):
                    org = OrganizationEntity(
                        text=match.group(),
                        label="ORG",
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.9,
                        normalized=full,
                        org_type="company",
                    )
                    result.organizations.append(org)
                    result.all_entities.append(org)

        return result

    def find_similar_entity(
        self,
        query: str,
        candidates: list[str],
        threshold: float = 0.8,
    ) -> Optional[str]:
        """Find the most similar entity from candidates using fuzzy matching.

        Args:
            query: Entity name to match
            candidates: List of candidate entity names
            threshold: Minimum similarity score (0-1)

        Returns:
            Best matching candidate or None if below threshold
        """
        if not candidates:
            return None

        # Use semantic similarity if available
        if self.similarity_model and TRANSFORMERS_AVAILABLE:
            return self._semantic_similarity_match(query, candidates, threshold)

        # Fallback to fuzzy string matching
        return self._fuzzy_match(query, candidates, threshold)

    def _fuzzy_match(
        self,
        query: str,
        candidates: list[str],
        threshold: float,
    ) -> Optional[str]:
        """Find similar entity using fuzzy string matching."""
        query_lower = query.lower()
        best_match = None
        best_score = 0.0

        for candidate in candidates:
            # Exact match (case-insensitive)
            if candidate.lower() == query_lower:
                return candidate

            # Sequence matcher score
            score = SequenceMatcher(None, query_lower, candidate.lower()).ratio()

            # Bonus for prefix match
            if candidate.lower().startswith(query_lower[:3]):
                score += 0.1

            if score > best_score:
                best_score = score
                best_match = candidate

        return best_match if best_score >= threshold else None

    def _semantic_similarity_match(
        self,
        query: str,
        candidates: list[str],
        threshold: float,
    ) -> Optional[str]:
        """Find similar entity using semantic similarity."""
        if not self.similarity_model or np is None:
            return self._fuzzy_match(query, candidates, threshold)

        # Encode query and candidates
        query_embedding = self.similarity_model.encode([query])
        candidate_embeddings = self.similarity_model.encode(candidates)

        # Compute cosine similarities
        similarities = np.dot(query_embedding, candidate_embeddings.T)[0]

        # Find best match
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        return candidates[best_idx] if best_score >= threshold else None

    def deduplicate_entities(
        self,
        entities: Sequence[Entity],
        similarity_threshold: float = 0.85,
    ) -> list[Entity]:
        """Remove duplicate entities using fuzzy matching.

        Args:
            entities: List of entities to deduplicate
            similarity_threshold: Similarity threshold for considering duplicates

        Returns:
            Deduplicated list of entities
        """
        if not entities:
            return []

        unique: list[Entity] = []
        seen_normalized: set[str] = set()

        for entity in entities:
            normalized = entity.normalized.lower()

            # Skip exact duplicates
            if normalized in seen_normalized:
                continue

            # Check fuzzy similarity with existing entities
            is_duplicate = False
            for existing in unique:
                score = SequenceMatcher(
                    None, normalized, existing.normalized.lower()
                ).ratio()
                if score >= similarity_threshold:
                    # Keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        unique.remove(existing)
                        unique.append(entity)
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(entity)
                seen_normalized.add(normalized)

        return unique


# =============================================================================
# Integration Functions
# =============================================================================


def extract_entities_from_story(
    title: str,
    summary: str,
    ner_engine: Optional[NEREngine] = None,
) -> ExtractionResult:
    """Extract entities from a story's title and summary.

    Args:
        title: Story title
        summary: Story summary/content
        ner_engine: Optional pre-initialized NER engine

    Returns:
        Combined extraction result from title and summary
    """
    if ner_engine is None:
        ner_engine = NEREngine()

    # Extract from title (higher weight)
    title_result = ner_engine.extract_entities(title)
    for entity in title_result.all_entities:
        entity.confidence = min(1.0, entity.confidence + 0.1)  # Boost title entities

    # Extract from summary
    summary_result = ner_engine.extract_entities(summary)

    # Combine results
    combined = ExtractionResult(text=f"{title}\n\n{summary}")
    combined.all_entities = title_result.all_entities + summary_result.all_entities
    combined.persons = title_result.persons + summary_result.persons
    combined.organizations = title_result.organizations + summary_result.organizations

    # Deduplicate
    combined.persons = [
        e
        for e in ner_engine.deduplicate_entities(combined.persons)
        if isinstance(e, PersonEntity)
    ]
    combined.organizations = [
        e
        for e in ner_engine.deduplicate_entities(combined.organizations)
        if isinstance(e, OrganizationEntity)
    ]
    # Cast to list[Entity] to satisfy type checker
    combined.all_entities = list[Entity](combined.persons) + list[Entity](
        combined.organizations
    )  # type: ignore[valid-type]

    # Calculate overall confidence
    if combined.all_entities:
        combined.confidence_score = sum(
            e.confidence for e in combined.all_entities
        ) / len(combined.all_entities)

    return combined


# =============================================================================
# Module-level convenience instance
# =============================================================================

# Lazy-loaded singleton
_ner_engine: Optional[NEREngine] = None


def get_ner_engine() -> NEREngine:
    """Get or create the singleton NER engine."""
    global _ner_engine
    if _ner_engine is None:
        _ner_engine = NEREngine()
    return _ner_engine


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for ner_engine module."""
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from test_framework import TestSuite

    from test_framework import TestSuite

    suite = TestSuite("NER Engine Tests")

    def test_entity_creation():
        """Test Entity dataclass creation."""
        entity = Entity(
            text="BASF",
            label="ORG",
            start_char=0,
            end_char=4,
            confidence=0.9,
        )
        assert entity.text == "BASF"
        assert entity.label == "ORG"
        assert entity.normalized == "BASF"

    def test_entity_normalized_default():
        """Test Entity normalized defaults to text."""
        entity = Entity(text="MIT", label="ORG")
        assert entity.normalized == "MIT"

    def test_person_entity_creation():
        """Test PersonEntity dataclass creation."""
        person = PersonEntity(
            text="Dr. John Smith",
            label="PERSON",
            title="CEO",
            affiliation="BASF",
        )
        assert person.text == "Dr. John Smith"
        assert person.label == "PERSON"
        assert person.title == "CEO"

    def test_organization_entity_creation():
        """Test OrganizationEntity dataclass creation."""
        org = OrganizationEntity(
            text="BASF SE",
            label="ORG",
            org_type="company",
            industry="chemicals",
        )
        assert org.text == "BASF SE"
        assert org.label == "ORG"
        assert org.org_type == "company"

    def test_extraction_result_creation():
        """Test ExtractionResult dataclass creation."""
        result = ExtractionResult(
            text="Sample text",
            confidence_score=0.85,
        )
        assert result.text == "Sample text"
        assert result.persons == []
        assert result.organizations == []

    def test_extraction_result_mention_candidates():
        """Test ExtractionResult mention_candidates property."""
        person = PersonEntity(text="John", label="PERSON", confidence=0.9)
        org = OrganizationEntity(text="BASF", label="ORG", confidence=0.8)
        result = ExtractionResult(persons=[person], organizations=[org])
        candidates = result.mention_candidates
        assert len(candidates) == 2
        assert candidates[0].confidence >= candidates[1].confidence

    def test_chem_eng_terms_defined():
        """Test CHEM_ENG_TERMS set is populated."""
        assert len(CHEM_ENG_TERMS) > 0
        assert "distillation" in CHEM_ENG_TERMS
        assert "catalysis" in CHEM_ENG_TERMS

    def test_company_suffixes_defined():
        """Test COMPANY_SUFFIXES set is populated."""
        assert len(COMPANY_SUFFIXES) > 0
        assert "ltd" in COMPANY_SUFFIXES
        assert "inc" in COMPANY_SUFFIXES

    def test_academic_patterns_defined():
        """Test ACADEMIC_PATTERNS set is populated."""
        assert len(ACADEMIC_PATTERNS) > 0
        assert "university" in ACADEMIC_PATTERNS
        assert "institute" in ACADEMIC_PATTERNS

    def test_professional_titles_defined():
        """Test PROFESSIONAL_TITLES set is populated."""
        assert len(PROFESSIONAL_TITLES) > 0
        assert "ceo" in PROFESSIONAL_TITLES
        assert "professor" in PROFESSIONAL_TITLES

    def test_known_companies_defined():
        """Test KNOWN_COMPANIES dict is populated."""
        assert len(KNOWN_COMPANIES) > 0
        assert "basf" in KNOWN_COMPANIES
        assert KNOWN_COMPANIES["basf"] == "BASF SE"

    def test_known_universities_defined():
        """Test KNOWN_UNIVERSITIES dict is populated."""
        assert len(KNOWN_UNIVERSITIES) > 0
        assert "mit" in KNOWN_UNIVERSITIES
        assert "Massachusetts Institute of Technology" in KNOWN_UNIVERSITIES["mit"]

    def test_ner_engine_creation():
        """Test NEREngine class creation."""
        engine = NEREngine()
        assert engine is not None
        assert engine.model_name == "en_core_web_sm"

    def test_ner_engine_extract_empty():
        """Test NEREngine extract with empty text."""
        engine = NEREngine()
        result = engine.extract_entities("")
        assert result.text == ""
        assert len(result.all_entities) == 0

    def test_ner_engine_extract_basic():
        """Test NEREngine basic extraction."""
        engine = NEREngine()
        result = engine.extract_entities("BASF announced a new partnership with MIT.")
        # Should find entities (depends on spaCy being installed)
        assert result is not None
        assert result.text == "BASF announced a new partnership with MIT."

    def test_get_ner_engine_singleton():
        """Test get_ner_engine returns singleton."""
        global _ner_engine
        _ner_engine = None  # Reset for test
        e1 = get_ner_engine()
        e2 = get_ner_engine()
        assert e1 is e2
        _ner_engine = None  # Cleanup

    suite.add_test("Entity creation", test_entity_creation)
    suite.add_test("Entity normalized default", test_entity_normalized_default)
    suite.add_test("PersonEntity creation", test_person_entity_creation)
    suite.add_test("OrganizationEntity creation", test_organization_entity_creation)
    suite.add_test("ExtractionResult creation", test_extraction_result_creation)
    suite.add_test(
        "ExtractionResult mention_candidates", test_extraction_result_mention_candidates
    )
    suite.add_test("CHEM_ENG_TERMS defined", test_chem_eng_terms_defined)
    suite.add_test("COMPANY_SUFFIXES defined", test_company_suffixes_defined)
    suite.add_test("ACADEMIC_PATTERNS defined", test_academic_patterns_defined)
    suite.add_test("PROFESSIONAL_TITLES defined", test_professional_titles_defined)
    suite.add_test("KNOWN_COMPANIES defined", test_known_companies_defined)
    suite.add_test("KNOWN_UNIVERSITIES defined", test_known_universities_defined)
    suite.add_test("NEREngine creation", test_ner_engine_creation)
    suite.add_test("NEREngine extract empty", test_ner_engine_extract_empty)
    suite.add_test("NEREngine extract basic", test_ner_engine_extract_basic)
    suite.add_test("get_ner_engine singleton", test_get_ner_engine_singleton)

    return suite
