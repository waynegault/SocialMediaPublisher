"""Retrieval-Augmented Generation (RAG) Engine for Personalized Content.

This module provides a RAG system to personalize content generation
based on the author's experience, resume, projects, and target jobs.

Features:
- Index author's resume, projects, publications, and target job descriptions
- Vector embeddings using ChromaDB for context retrieval
- Personal context injection into summary generation prompts
- Semantic search for relevant experience

TASK 1.1: RAG for Personalized Content Generation
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger(__name__)

# Type hints for optional dependencies
if TYPE_CHECKING:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

# Optional ChromaDB import
try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None  # type: ignore[assignment]
    Settings = None  # type: ignore[assignment, misc]
    logger.debug("chromadb not available - pip install chromadb")

# Optional sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore[assignment, misc]
    logger.debug("sentence-transformers not available for embeddings")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Document:
    """A document for indexing in the RAG system."""

    id: str
    content: str
    doc_type: str  # resume, project, publication, job_description, skill, etc.
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""

    documents: list[Document] = field(default_factory=list)
    query: str = ""
    scores: list[float] = field(default_factory=list)

    @property
    def context(self) -> str:
        """Get concatenated context from retrieved documents."""
        return "\n\n---\n\n".join(doc.content for doc in self.documents)

    @property
    def top_document(self) -> Optional[Document]:
        """Get the most relevant document."""
        return self.documents[0] if self.documents else None


@dataclass
class PersonalContext:
    """Author's personal context for content generation."""

    name: str = ""
    current_title: str = ""
    target_role: str = ""
    years_experience: int = 0
    key_skills: list[str] = field(default_factory=list)
    industries: list[str] = field(default_factory=list)
    companies: list[str] = field(default_factory=list)
    achievements: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Convert to a prompt-friendly context string."""
        lines = []

        if self.name:
            lines.append(f"Author: {self.name}")
        if self.current_title:
            lines.append(f"Current Role: {self.current_title}")
        if self.target_role:
            lines.append(f"Target Role: {self.target_role}")
        if self.years_experience:
            lines.append(f"Experience: {self.years_experience} years")
        if self.key_skills:
            lines.append(f"Key Skills: {', '.join(self.key_skills[:10])}")
        if self.industries:
            lines.append(f"Industries: {', '.join(self.industries)}")
        if self.companies:
            lines.append(f"Companies: {', '.join(self.companies[:5])}")
        if self.achievements:
            lines.append("Key Achievements:")
            for achievement in self.achievements[:3]:
                lines.append(f"  - {achievement}")
        if self.education:
            lines.append(f"Education: {', '.join(self.education)}")

        return "\n".join(lines)


# =============================================================================
# RAG Engine
# =============================================================================


class RAGEngine:
    """Retrieval-Augmented Generation engine for personalized content.

    Uses ChromaDB for vector storage and sentence-transformers for embeddings.
    Falls back to keyword matching if dependencies are not available.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "personal_context",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialize the RAG engine.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model for embeddings
        """
        self.persist_directory = persist_directory or str(
            Path(__file__).parent / ".rag_data"
        )
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        # Initialize components
        self.chroma_client: "Optional[Any]" = None  # chromadb.Client when available
        self.collection: "Optional[Any]" = None  # chromadb.Collection when available
        self.embedding_model: "Optional[Any]" = (
            None  # SentenceTransformer when available
        )

        # Document cache for fallback
        self._documents: dict[str, Document] = {}

        # Personal context
        self.personal_context = PersonalContext()

        self._initialize()

    def _initialize(self) -> None:
        """Initialize ChromaDB and embedding model."""
        # Initialize embedding model
        if EMBEDDINGS_AVAILABLE and SentenceTransformer is not None:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")

        # Initialize ChromaDB
        if CHROMADB_AVAILABLE and chromadb is not None and Settings is not None:
            try:
                os.makedirs(self.persist_directory, exist_ok=True)
                self.chroma_client = chromadb.Client(
                    Settings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=self.persist_directory,
                        anonymized_telemetry=False,
                    )
                )
                assert self.chroma_client is not None  # Type narrowing for Pylance
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "Personal context for RAG"},
                )
                logger.info(f"ChromaDB initialized at: {self.persist_directory}")
            except Exception as e:
                logger.warning(f"Failed to initialize ChromaDB: {e}")
        else:
            logger.info("Using in-memory document storage (ChromaDB not available)")

    def add_document(
        self,
        doc_id: str,
        content: str,
        doc_type: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a document to the RAG index.

        Args:
            doc_id: Unique document identifier
            content: Document content
            doc_type: Type of document (resume, project, etc.)
            metadata: Additional metadata
        """
        metadata = metadata or {}
        metadata["doc_type"] = doc_type

        doc = Document(
            id=doc_id,
            content=content,
            doc_type=doc_type,
            metadata=metadata,
        )

        # Store in cache
        self._documents[doc_id] = doc

        # Add to ChromaDB if available
        if self.collection is not None:
            try:
                embedding = None
                if self.embedding_model:
                    embedding = self.embedding_model.encode(content).tolist()

                self.collection.upsert(
                    ids=[doc_id],
                    documents=[content],
                    metadatas=[metadata],
                    embeddings=[embedding] if embedding else None,
                )
                logger.debug(f"Added document to ChromaDB: {doc_id}")
            except Exception as e:
                logger.warning(f"Failed to add document to ChromaDB: {e}")

    def add_resume(self, resume_text: str, metadata: Optional[dict] = None) -> None:
        """Add resume content to the index.

        Args:
            resume_text: Full resume text
            metadata: Additional metadata
        """
        self.add_document(
            doc_id="resume_main",
            content=resume_text,
            doc_type="resume",
            metadata=metadata,
        )

        # Parse resume sections
        self._parse_resume_sections(resume_text)

    def _parse_resume_sections(self, resume_text: str) -> None:
        """Parse resume into sections for granular retrieval."""
        # Simple section detection
        sections = {
            "experience": [],
            "education": [],
            "skills": [],
            "projects": [],
            "achievements": [],
        }

        current_section = None
        lines = resume_text.split("\n")

        for line in lines:
            line_lower = line.lower().strip()

            # Detect section headers
            if "experience" in line_lower or "work history" in line_lower:
                current_section = "experience"
            elif "education" in line_lower:
                current_section = "education"
            elif "skill" in line_lower:
                current_section = "skills"
            elif "project" in line_lower:
                current_section = "projects"
            elif "achievement" in line_lower or "accomplishment" in line_lower:
                current_section = "achievements"
            elif current_section and line.strip():
                sections[current_section].append(line.strip())

        # Add each section as a separate document
        for section_name, content_lines in sections.items():
            if content_lines:
                content = "\n".join(content_lines)
                self.add_document(
                    doc_id=f"resume_{section_name}",
                    content=content,
                    doc_type=f"resume_{section_name}",
                )

    def add_project(
        self,
        project_name: str,
        description: str,
        technologies: Optional[list[str]] = None,
        outcomes: Optional[list[str]] = None,
    ) -> None:
        """Add a project to the index.

        Args:
            project_name: Name of the project
            description: Project description
            technologies: Technologies used
            outcomes: Project outcomes/achievements
        """
        content_parts = [f"Project: {project_name}", f"Description: {description}"]

        if technologies:
            content_parts.append(f"Technologies: {', '.join(technologies)}")
        if outcomes:
            content_parts.append("Outcomes:")
            content_parts.extend([f"  - {o}" for o in outcomes])

        content = "\n".join(content_parts)
        self.add_document(
            doc_id=f"project_{project_name.lower().replace(' ', '_')}",
            content=content,
            doc_type="project",
            metadata={
                "project_name": project_name,
                "technologies": technologies or [],
            },
        )

    def add_job_description(
        self,
        job_title: str,
        company: str,
        description: str,
        requirements: Optional[list[str]] = None,
    ) -> None:
        """Add a target job description to the index.

        Args:
            job_title: Job title
            company: Company name
            description: Job description
            requirements: Job requirements
        """
        content_parts = [
            f"Job Title: {job_title}",
            f"Company: {company}",
            f"Description: {description}",
        ]

        if requirements:
            content_parts.append("Requirements:")
            content_parts.extend([f"  - {r}" for r in requirements])

        content = "\n".join(content_parts)
        self.add_document(
            doc_id=f"job_{job_title.lower().replace(' ', '_')}_{company.lower().replace(' ', '_')}",
            content=content,
            doc_type="job_description",
            metadata={"job_title": job_title, "company": company},
        )

    def add_skill(
        self, skill_name: str, description: str, level: str = "expert"
    ) -> None:
        """Add a skill with context.

        Args:
            skill_name: Name of the skill
            description: Description of experience with this skill
            level: Proficiency level
        """
        content = f"Skill: {skill_name}\nLevel: {level}\nExperience: {description}"
        self.add_document(
            doc_id=f"skill_{skill_name.lower().replace(' ', '_')}",
            content=content,
            doc_type="skill",
            metadata={"skill_name": skill_name, "level": level},
        )

    def retrieve(
        self,
        query: str,
        n_results: int = 3,
        doc_types: Optional[list[str]] = None,
    ) -> RetrievalResult:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query
            n_results: Number of results to return
            doc_types: Filter by document types

        Returns:
            RetrievalResult with matching documents
        """
        result = RetrievalResult(query=query)

        # Try ChromaDB first
        if self.collection is not None:
            try:
                where_filter = None
                if doc_types:
                    where_filter = {"doc_type": {"$in": doc_types}}

                query_embedding = None
                if self.embedding_model:
                    query_embedding = self.embedding_model.encode(query).tolist()

                if query_embedding:
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        where=where_filter,
                    )
                else:
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=n_results,
                        where=where_filter,
                    )

                # Parse results
                if results["ids"] and results["ids"][0]:
                    for i, doc_id in enumerate(results["ids"][0]):
                        doc = Document(
                            id=doc_id,
                            content=results["documents"][0][i]
                            if results["documents"]
                            else "",
                            doc_type=results["metadatas"][0][i].get(
                                "doc_type", "unknown"
                            )
                            if results["metadatas"]
                            else "unknown",
                            metadata=results["metadatas"][0][i]
                            if results["metadatas"]
                            else {},
                        )
                        result.documents.append(doc)

                        if results["distances"]:
                            # Convert distance to similarity score
                            result.scores.append(1 - results["distances"][0][i])

                return result

            except Exception as e:
                logger.warning(f"ChromaDB query failed, falling back to keyword: {e}")

        # Fallback to keyword matching
        return self._keyword_retrieve(query, n_results, doc_types)

    def _keyword_retrieve(
        self,
        query: str,
        n_results: int,
        doc_types: Optional[list[str]],
    ) -> RetrievalResult:
        """Fallback keyword-based retrieval."""
        result = RetrievalResult(query=query)
        query_words = set(query.lower().split())

        scored_docs: list[tuple[Document, float]] = []

        for doc in self._documents.values():
            # Filter by type if specified
            if doc_types and doc.doc_type not in doc_types:
                continue

            # Simple keyword matching score
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / len(query_words) if query_words else 0

            if score > 0:
                scored_docs.append((doc, score))

        # Sort by score and take top results
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        for doc, score in scored_docs[:n_results]:
            result.documents.append(doc)
            result.scores.append(score)

        return result

    def get_context_for_story(
        self,
        title: str,
        summary: str,
        max_context_length: int = 1500,
    ) -> str:
        """Get relevant personal context for a story.

        Args:
            title: Story title
            summary: Story summary
            max_context_length: Maximum context length in characters

        Returns:
            Relevant personal context string
        """
        # Combine title and summary for query
        query = f"{title} {summary}"

        # Retrieve relevant documents
        result = self.retrieve(query, n_results=3)

        # Build context
        context_parts = []

        # Add personal context summary
        if self.personal_context.name:
            context_parts.append(self.personal_context.to_prompt_context())
            context_parts.append("")

        # Add retrieved documents
        for doc in result.documents:
            doc_context = f"[{doc.doc_type.upper()}]\n{doc.content}"
            context_parts.append(doc_context)

        context = "\n\n".join(context_parts)

        # Truncate if too long
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        return context

    def generate_personalized_prompt(
        self,
        title: str,
        summary: str,
        base_prompt: str,
    ) -> str:
        """Generate a personalized prompt with RAG context.

        Args:
            title: Story title
            summary: Story summary
            base_prompt: Base prompt template

        Returns:
            Personalized prompt with context injected
        """
        context = self.get_context_for_story(title, summary)

        if context:
            personalized_prompt = f"""
AUTHOR'S PROFESSIONAL CONTEXT:
{context}

STORY TO PERSONALIZE:
Title: {title}
Summary: {summary}

INSTRUCTIONS:
{base_prompt}

When writing, naturally connect the story to the author's experience where relevant.
Reference specific skills, projects, or achievements that relate to the topic.
Maintain authenticity - only reference experiences that genuinely connect to the story.
"""
        else:
            personalized_prompt = f"""
STORY:
Title: {title}
Summary: {summary}

INSTRUCTIONS:
{base_prompt}
"""

        return personalized_prompt

    def load_personal_context(self, file_path: str) -> None:
        """Load personal context from a JSON file.

        Args:
            file_path: Path to JSON file with personal context
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.personal_context = PersonalContext(
                name=data.get("name", ""),
                current_title=data.get("current_title", ""),
                target_role=data.get("target_role", ""),
                years_experience=data.get("years_experience", 0),
                key_skills=data.get("key_skills", []),
                industries=data.get("industries", []),
                companies=data.get("companies", []),
                achievements=data.get("achievements", []),
                education=data.get("education", []),
            )

            # Index achievements and skills
            for i, achievement in enumerate(self.personal_context.achievements):
                self.add_document(
                    doc_id=f"achievement_{i}",
                    content=achievement,
                    doc_type="achievement",
                )

            for skill in self.personal_context.key_skills:
                self.add_skill(skill, f"Expert in {skill}")

            logger.info(f"Loaded personal context from: {file_path}")

        except Exception as e:
            logger.warning(f"Failed to load personal context: {e}")

    def save_index(self) -> None:
        """Persist the index to disk."""
        if self.chroma_client:
            try:
                self.chroma_client.persist()
                logger.info("Persisted ChromaDB index")
            except Exception as e:
                logger.warning(f"Failed to persist index: {e}")

    def clear_index(self) -> None:
        """Clear all indexed documents."""
        self._documents.clear()

        if self.collection and self.chroma_client:
            try:
                # Delete and recreate collection
                self.chroma_client.delete_collection(self.collection_name)
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Personal context for RAG"},
                )
                logger.info("Cleared RAG index")
            except Exception as e:
                logger.warning(f"Failed to clear ChromaDB collection: {e}")


# =============================================================================
# Integration Functions
# =============================================================================


def get_personalized_context(
    title: str,
    summary: str,
    rag_engine: Optional[RAGEngine] = None,
) -> str:
    """Get personalized context for a story.

    Args:
        title: Story title
        summary: Story summary
        rag_engine: Optional pre-initialized RAG engine

    Returns:
        Personalized context string
    """
    if rag_engine is None:
        rag_engine = RAGEngine()

    return rag_engine.get_context_for_story(title, summary)


def create_personalized_prompt(
    title: str,
    summary: str,
    base_prompt: str,
    rag_engine: Optional[RAGEngine] = None,
) -> str:
    """Create a personalized prompt with RAG context.

    Args:
        title: Story title
        summary: Story summary
        base_prompt: Base prompt template
        rag_engine: Optional pre-initialized RAG engine

    Returns:
        Personalized prompt
    """
    if rag_engine is None:
        rag_engine = RAGEngine()

    return rag_engine.generate_personalized_prompt(title, summary, base_prompt)


# =============================================================================
# Module-level convenience instance
# =============================================================================

_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Get or create the singleton RAG engine."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for rag_engine module."""
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from test_framework import TestSuite

    from test_framework import TestSuite
    import tempfile

    suite = TestSuite("RAG Engine Tests")

    def test_document_creation():
        """Test Document dataclass creation."""
        doc = Document(
            id="doc1",
            content="Sample document content",
            doc_type="resume",
            metadata={"key": "value"},
        )
        assert doc.id == "doc1"
        assert doc.content == "Sample document content"
        assert doc.doc_type == "resume"

    def test_retrieval_result_creation():
        """Test RetrievalResult dataclass creation."""
        result = RetrievalResult(query="test query")
        assert result.query == "test query"
        assert result.documents == []
        assert result.scores == []

    def test_retrieval_result_context():
        """Test RetrievalResult context property."""
        doc1 = Document(id="1", content="First doc", doc_type="test")
        doc2 = Document(id="2", content="Second doc", doc_type="test")
        result = RetrievalResult(query="test", documents=[doc1, doc2])
        context = result.context
        assert "First doc" in context
        assert "Second doc" in context

    def test_retrieval_result_top_document():
        """Test RetrievalResult top_document property."""
        doc = Document(id="1", content="Top doc", doc_type="test")
        result = RetrievalResult(query="test", documents=[doc])
        assert result.top_document is doc

    def test_retrieval_result_top_document_empty():
        """Test RetrievalResult top_document when empty."""
        result = RetrievalResult(query="test")
        assert result.top_document is None

    def test_personal_context_creation():
        """Test PersonalContext dataclass creation."""
        ctx = PersonalContext(
            name="John Doe",
            current_title="Senior Engineer",
            target_role="Lead Engineer",
            years_experience=10,
        )
        assert ctx.name == "John Doe"
        assert ctx.years_experience == 10

    def test_personal_context_to_prompt_context():
        """Test PersonalContext to_prompt_context method."""
        ctx = PersonalContext(
            name="Jane Doe",
            current_title="Process Engineer",
            key_skills=["Python", "Chemical Engineering"],
        )
        prompt = ctx.to_prompt_context()
        assert "Jane Doe" in prompt
        assert "Process Engineer" in prompt
        assert "Python" in prompt

    def test_personal_context_empty():
        """Test PersonalContext with no data."""
        ctx = PersonalContext()
        prompt = ctx.to_prompt_context()
        assert prompt == ""

    def test_rag_engine_creation():
        """Test RAGEngine class creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = RAGEngine(persist_directory=tmpdir)
            assert engine is not None
            assert engine.persist_directory == tmpdir

    def test_rag_engine_add_document():
        """Test RAGEngine add_document method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = RAGEngine(persist_directory=tmpdir)
            engine.add_document(
                doc_id="test_doc",
                content="Test content about chemical engineering",
                doc_type="test",
            )
            assert "test_doc" in engine._documents

    def test_rag_engine_add_skill():
        """Test RAGEngine add_skill method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = RAGEngine(persist_directory=tmpdir)
            engine.add_skill("Python", "10 years of experience", level="expert")
            assert any("python" in k for k in engine._documents.keys())

    def test_rag_engine_add_project():
        """Test RAGEngine add_project method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = RAGEngine(persist_directory=tmpdir)
            engine.add_project(
                project_name="Test Project",
                description="A test project description",
                technologies=["Python", "SQL"],
            )
            assert any("project" in k for k in engine._documents.keys())

    def test_rag_engine_keyword_retrieve():
        """Test RAGEngine keyword-based retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = RAGEngine(persist_directory=tmpdir)
            engine.add_document("doc1", "Python programming language", "skill")
            engine.add_document("doc2", "Java development", "skill")
            result = engine._keyword_retrieve("Python", n_results=2, doc_types=None)
            assert len(result.documents) > 0

    def test_get_rag_engine_singleton():
        """Test get_rag_engine returns singleton."""
        global _rag_engine
        _rag_engine = None  # Reset for test
        r1 = get_rag_engine()
        r2 = get_rag_engine()
        assert r1 is r2
        _rag_engine = None  # Cleanup

    suite.add_test("Document creation", test_document_creation)
    suite.add_test("RetrievalResult creation", test_retrieval_result_creation)
    suite.add_test("RetrievalResult context", test_retrieval_result_context)
    suite.add_test("RetrievalResult top_document", test_retrieval_result_top_document)
    suite.add_test(
        "RetrievalResult top_document empty", test_retrieval_result_top_document_empty
    )
    suite.add_test("PersonalContext creation", test_personal_context_creation)
    suite.add_test(
        "PersonalContext to_prompt_context", test_personal_context_to_prompt_context
    )
    suite.add_test("PersonalContext empty", test_personal_context_empty)
    suite.add_test("RAGEngine creation", test_rag_engine_creation)
    suite.add_test("RAGEngine add_document", test_rag_engine_add_document)
    suite.add_test("RAGEngine add_skill", test_rag_engine_add_skill)
    suite.add_test("RAGEngine add_project", test_rag_engine_add_project)
    suite.add_test("RAGEngine keyword retrieve", test_rag_engine_keyword_retrieve)
    suite.add_test("get_rag_engine singleton", test_get_rag_engine_singleton)

    return suite
