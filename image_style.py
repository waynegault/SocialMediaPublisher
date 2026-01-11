"""Image style consistency for AI-generated images.

This module provides visual style templates and consistency enforcement
to build a recognizable brand identity in LinkedIn posts.

Features:
- Predefined style presets for different content types
- Color palette management
- Style application to prompts
- Style performance tracking
- A/B testing integration for styles

Example:
    manager = StyleManager()
    enhanced_prompt = manager.apply_style(
        "A photo of an engineer working on solar panels",
        style_preset="technical_professional"
    )
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ColorPalette:
    """Color palette for consistent brand imagery."""

    primary: str = "#1E3A5F"  # Deep blue
    secondary: str = "#2ECC71"  # Green accent
    accent: str = "#E74C3C"  # Red for highlights
    background: str = "#F5F5F5"  # Light gray
    text: str = "#2C3E50"  # Dark gray
    name: str = "default"

    def to_prompt_description(self) -> str:
        """Convert palette to natural language for prompts."""
        return (
            f"professional color palette with deep blue ({self.primary}) "
            f"and green ({self.secondary}) accents"
        )


@dataclass
class StylePreset:
    """A complete style preset for image generation."""

    name: str
    description: str
    prompt_prefix: str = ""  # Added before user prompt
    prompt_suffix: str = ""  # Added after user prompt
    negative_prompt: str = ""  # Things to avoid
    lighting: str = ""  # Lighting description
    camera_angle: str = ""  # Camera angle/perspective
    color_palette: ColorPalette = field(default_factory=ColorPalette)
    tags: list[str] = field(default_factory=list)
    
    # Performance tracking
    usage_count: int = 0
    total_engagement: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def apply_to_prompt(self, base_prompt: str) -> str:
        """Apply this style to a base prompt."""
        parts = []
        
        if self.prompt_prefix:
            parts.append(self.prompt_prefix)
        
        parts.append(base_prompt)
        
        if self.lighting:
            parts.append(self.lighting)
            
        if self.camera_angle:
            parts.append(self.camera_angle)
            
        if self.prompt_suffix:
            parts.append(self.prompt_suffix)
        
        return ", ".join(parts)

    @property
    def average_engagement(self) -> float:
        """Calculate average engagement per use."""
        if self.usage_count == 0:
            return 0.0
        return self.total_engagement / self.usage_count


@dataclass
class StyleUsage:
    """Track usage of a style preset."""

    preset_name: str
    applied_at: datetime
    original_prompt: str
    styled_prompt: str
    engagement_score: float | None = None
    image_quality_score: float | None = None


# =============================================================================
# Predefined Style Presets
# =============================================================================


def _create_technical_professional() -> StylePreset:
    """Create a technical professional style preset."""
    return StylePreset(
        name="technical_professional",
        description="Clean, professional technical imagery for engineering content",
        prompt_prefix="professional industrial photography, technical documentation style",
        prompt_suffix="high resolution, sharp focus, clean composition, professional lighting",
        negative_prompt="cartoon, anime, blurry, low quality, text, watermark, distorted",
        lighting="soft natural lighting with subtle shadows",
        camera_angle="eye level professional shot",
        color_palette=ColorPalette(
            primary="#1E3A5F",
            secondary="#4A90A4",
            accent="#F39C12",
            name="industrial",
        ),
        tags=["engineering", "technical", "professional", "industrial"],
    )


def _create_innovation_showcase() -> StylePreset:
    """Create an innovation/technology showcase style."""
    return StylePreset(
        name="innovation_showcase",
        description="Modern tech imagery highlighting innovation and progress",
        prompt_prefix="modern technology photography, innovation showcase style",
        prompt_suffix="futuristic aesthetic, clean lines, high-tech environment",
        negative_prompt="outdated, vintage, cluttered, dark, grainy",
        lighting="bright ambient lighting with tech-inspired highlights",
        camera_angle="dynamic angle showcasing technology",
        color_palette=ColorPalette(
            primary="#0077B5",  # LinkedIn blue
            secondary="#00A0DC",
            accent="#FFFFFF",
            background="#1B1B1B",
            name="tech_modern",
        ),
        tags=["innovation", "technology", "modern", "startup"],
    )


def _create_corporate_leadership() -> StylePreset:
    """Create a corporate leadership style."""
    return StylePreset(
        name="corporate_leadership",
        description="Professional corporate imagery for executive-level content",
        prompt_prefix="corporate photography, executive business style",
        prompt_suffix="polished professional environment, confident composition",
        negative_prompt="casual, informal, messy, low quality, unprofessional",
        lighting="professional studio lighting, well-balanced",
        camera_angle="slightly low angle for authority",
        color_palette=ColorPalette(
            primary="#2C3E50",
            secondary="#34495E",
            accent="#C0392B",
            background="#ECF0F1",
            name="corporate",
        ),
        tags=["leadership", "corporate", "executive", "professional"],
    )


def _create_sustainability_green() -> StylePreset:
    """Create a sustainability/green energy style."""
    return StylePreset(
        name="sustainability_green",
        description="Eco-friendly imagery for sustainability and green energy content",
        prompt_prefix="environmental photography, sustainability focus",
        prompt_suffix="natural elements, eco-conscious composition, hopeful atmosphere",
        negative_prompt="pollution, industrial waste, dark, gloomy, artificial",
        lighting="warm natural sunlight, golden hour feel",
        camera_angle="wide angle showcasing environment",
        color_palette=ColorPalette(
            primary="#27AE60",
            secondary="#2ECC71",
            accent="#F1C40F",
            background="#E8F6E9",
            name="green_energy",
        ),
        tags=["sustainability", "green", "renewable", "environment", "clean energy"],
    )


def _create_data_analytics() -> StylePreset:
    """Create a data/analytics visualization style."""
    return StylePreset(
        name="data_analytics",
        description="Data-driven imagery for analytics and insights content",
        prompt_prefix="data visualization, business intelligence style",
        prompt_suffix="clean infographic aesthetic, organized layout",
        negative_prompt="messy, chaotic, illegible, blurry, overcrowded",
        lighting="even lighting for clear visibility",
        camera_angle="straight-on perspective for clarity",
        color_palette=ColorPalette(
            primary="#3498DB",
            secondary="#9B59B6",
            accent="#E74C3C",
            background="#FFFFFF",
            name="analytics",
        ),
        tags=["data", "analytics", "visualization", "insights", "charts"],
    )


def _create_manufacturing() -> StylePreset:
    """Create a manufacturing/factory style."""
    return StylePreset(
        name="manufacturing",
        description="Industrial manufacturing imagery for production content",
        prompt_prefix="industrial manufacturing photography, factory floor style",
        prompt_suffix="precise machinery, operational excellence, quality focus",
        negative_prompt="unsafe conditions, mess, broken equipment, low quality",
        lighting="industrial lighting, high visibility",
        camera_angle="medium shot showing equipment context",
        color_palette=ColorPalette(
            primary="#7F8C8D",
            secondary="#F39C12",
            accent="#E74C3C",
            background="#BDC3C7",
            name="industrial_manufacturing",
        ),
        tags=["manufacturing", "factory", "production", "industrial", "machinery"],
    )


# =============================================================================
# Style Manager
# =============================================================================


class StyleManager:
    """
    Manage image styles for consistent brand imagery.
    
    Features:
    - Predefined style presets
    - Custom style creation
    - Style application to prompts
    - Performance tracking
    - Style recommendations
    """

    def __init__(self) -> None:
        """Initialize the style manager with default presets."""
        self._presets: dict[str, StylePreset] = {}
        self._usage_history: list[StyleUsage] = []
        self._max_history = 1000
        
        # Load default presets
        self._load_default_presets()

    def _load_default_presets(self) -> None:
        """Load all default style presets."""
        default_presets = [
            _create_technical_professional(),
            _create_innovation_showcase(),
            _create_corporate_leadership(),
            _create_sustainability_green(),
            _create_data_analytics(),
            _create_manufacturing(),
        ]
        
        for preset in default_presets:
            self._presets[preset.name] = preset

    def get_preset(self, name: str) -> StylePreset | None:
        """Get a style preset by name."""
        return self._presets.get(name)

    def list_presets(self) -> list[str]:
        """List all available preset names."""
        return list(self._presets.keys())

    def get_all_presets(self) -> dict[str, StylePreset]:
        """Get all presets."""
        return self._presets.copy()

    def add_preset(self, preset: StylePreset) -> None:
        """Add or update a style preset."""
        self._presets[preset.name] = preset

    def remove_preset(self, name: str) -> bool:
        """Remove a style preset."""
        if name in self._presets:
            del self._presets[name]
            return True
        return False

    def apply_style(
        self,
        base_prompt: str,
        style_preset: str | None = None,
        include_negative: bool = True,
    ) -> dict[str, str]:
        """
        Apply a style preset to a base prompt.
        
        Args:
            base_prompt: The original image generation prompt
            style_preset: Name of the preset to apply (default: technical_professional)
            include_negative: Whether to include negative prompt
            
        Returns:
            Dict with 'prompt' and optionally 'negative_prompt'
        """
        preset_name = style_preset or "technical_professional"
        preset = self._presets.get(preset_name)
        
        if not preset:
            # Return original prompt if preset not found
            return {"prompt": base_prompt}
        
        styled_prompt = preset.apply_to_prompt(base_prompt)
        
        # Track usage
        usage = StyleUsage(
            preset_name=preset_name,
            applied_at=datetime.now(timezone.utc),
            original_prompt=base_prompt,
            styled_prompt=styled_prompt,
        )
        self._add_usage(usage)
        
        # Update preset usage count
        preset.usage_count += 1
        
        result: dict[str, str] = {"prompt": styled_prompt}
        
        if include_negative and preset.negative_prompt:
            result["negative_prompt"] = preset.negative_prompt
            
        return result

    def _add_usage(self, usage: StyleUsage) -> None:
        """Add usage record, maintaining history limit."""
        self._usage_history.append(usage)
        if len(self._usage_history) > self._max_history:
            self._usage_history = self._usage_history[-self._max_history:]

    def record_engagement(
        self,
        preset_name: str,
        engagement_score: float,
    ) -> None:
        """
        Record engagement for a style preset.
        
        Args:
            preset_name: Name of the preset used
            engagement_score: Engagement score achieved
        """
        preset = self._presets.get(preset_name)
        if preset:
            preset.total_engagement += engagement_score

    def get_best_performing_style(self) -> str | None:
        """Get the best performing style based on average engagement."""
        best_name: str | None = None
        best_avg = 0.0
        
        for name, preset in self._presets.items():
            avg = preset.average_engagement
            if avg > best_avg:
                best_avg = avg
                best_name = name
                
        return best_name

    def recommend_style(
        self,
        content_keywords: list[str],
    ) -> str:
        """
        Recommend a style based on content keywords.
        
        Args:
            content_keywords: Keywords from the content
            
        Returns:
            Name of recommended style preset
        """
        if not content_keywords:
            return "technical_professional"
        
        # Score each preset based on tag matches
        scores: dict[str, int] = {}
        
        keywords_lower = [kw.lower() for kw in content_keywords]
        
        for name, preset in self._presets.items():
            score = 0
            for tag in preset.tags:
                for keyword in keywords_lower:
                    if tag in keyword or keyword in tag:
                        score += 1
            scores[name] = score
        
        # Return best match or default
        if scores:
            best = max(scores, key=lambda k: scores[k])
            if scores[best] > 0:
                return best
        
        return "technical_professional"

    def get_random_style(self, exclude: list[str] | None = None) -> str:
        """Get a random style preset name."""
        available = list(self._presets.keys())
        
        if exclude:
            available = [s for s in available if s not in exclude]
            
        if not available:
            return "technical_professional"
            
        return random.choice(available)

    def get_style_stats(self) -> dict[str, Any]:
        """Get statistics about style usage."""
        stats: dict[str, Any] = {
            "total_presets": len(self._presets),
            "total_usages": len(self._usage_history),
            "preset_stats": {},
        }
        
        for name, preset in self._presets.items():
            stats["preset_stats"][name] = {
                "usage_count": preset.usage_count,
                "total_engagement": preset.total_engagement,
                "average_engagement": preset.average_engagement,
            }
            
        return stats

    def format_style_report(self) -> str:
        """Format a human-readable style usage report."""
        lines = [
            "Style Usage Report",
            "=" * 40,
            f"Total Presets: {len(self._presets)}",
            f"Total Usages: {len(self._usage_history)}",
            "",
            "Preset Performance:",
        ]
        
        # Sort by usage count
        sorted_presets = sorted(
            self._presets.values(),
            key=lambda p: p.usage_count,
            reverse=True,
        )
        
        for preset in sorted_presets:
            lines.append(
                f"  {preset.name}: {preset.usage_count} uses, "
                f"avg engagement: {preset.average_engagement:.2f}"
            )
        
        return "\n".join(lines)


# =============================================================================
# Style A/B Testing Integration
# =============================================================================


@dataclass
class StyleVariant:
    """A style variant for A/B testing."""

    style_name: str
    weight: float = 1.0  # Relative probability


class StyleExperiment:
    """
    Run A/B tests on different image styles.
    
    Integrates with ab_testing.py for statistical analysis.
    """

    def __init__(
        self,
        name: str,
        variants: list[StyleVariant],
    ) -> None:
        """
        Initialize a style experiment.
        
        Args:
            name: Experiment name
            variants: List of style variants to test
        """
        self.name = name
        self.variants = variants
        self._results: dict[str, list[float]] = {v.style_name: [] for v in variants}
        self._created_at = datetime.now(timezone.utc)

    def get_variant(self, content_id: str) -> str:
        """
        Get a style variant for given content.
        
        Uses deterministic assignment for reproducibility.
        """
        if not self.variants:
            return "technical_professional"
        
        # Use hash of content_id for deterministic selection
        hash_val = hash(content_id + self.name)
        total_weight = sum(v.weight for v in self.variants)
        
        position = (abs(hash_val) % 1000) / 1000 * total_weight
        
        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.weight
            if position < cumulative:
                return variant.style_name
        
        return self.variants[-1].style_name

    def record_result(
        self,
        style_name: str,
        engagement_score: float,
    ) -> None:
        """Record an engagement result for a style."""
        if style_name in self._results:
            self._results[style_name].append(engagement_score)

    def get_summary(self) -> dict[str, Any]:
        """Get experiment summary."""
        summary: dict[str, Any] = {
            "name": self.name,
            "variants": {},
            "total_samples": sum(len(r) for r in self._results.values()),
        }
        
        for style_name, results in self._results.items():
            if results:
                summary["variants"][style_name] = {
                    "samples": len(results),
                    "mean": sum(results) / len(results),
                    "min": min(results),
                    "max": max(results),
                }
            else:
                summary["variants"][style_name] = {
                    "samples": 0,
                    "mean": 0.0,
                }
        
        return summary

    def get_winner(self, min_samples: int = 10) -> str | None:
        """
        Get the winning style if there's enough data.
        
        Args:
            min_samples: Minimum samples per variant
            
        Returns:
            Name of winning style or None if inconclusive
        """
        best_style: str | None = None
        best_mean = 0.0
        
        for style_name, results in self._results.items():
            if len(results) < min_samples:
                return None  # Not enough data
                
            mean = sum(results) / len(results)
            if mean > best_mean:
                best_mean = mean
                best_style = style_name
        
        return best_style


# =============================================================================
# Convenience Functions
# =============================================================================


def create_styled_prompt(
    base_prompt: str,
    style: str = "technical_professional",
) -> str:
    """
    Quick function to apply a style to a prompt.
    
    Args:
        base_prompt: Original prompt
        style: Style preset name
        
    Returns:
        Styled prompt string
    """
    manager = StyleManager()
    result = manager.apply_style(base_prompt, style_preset=style)
    return result["prompt"]


def list_available_styles() -> list[str]:
    """Get list of all available style preset names."""
    manager = StyleManager()
    return manager.list_presets()


def get_style_for_topic(topic: str) -> str:
    """
    Get recommended style for a topic.
    
    Args:
        topic: Topic description
        
    Returns:
        Recommended style name
    """
    manager = StyleManager()
    keywords = topic.lower().split()
    return manager.recommend_style(keywords)


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests() -> "TestSuite":
    """Create unit tests for this module."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_framework import TestSuite

    suite = TestSuite("Image Style Tests")

    def test_color_palette_description():
        palette = ColorPalette()
        desc = palette.to_prompt_description()
        assert "blue" in desc.lower()
        assert palette.primary in desc

    def test_style_preset_apply():
        preset = StylePreset(
            name="test",
            description="Test style",
            prompt_prefix="PREFIX",
            prompt_suffix="SUFFIX",
        )
        result = preset.apply_to_prompt("base prompt")
        assert "PREFIX" in result
        assert "SUFFIX" in result
        assert "base prompt" in result

    def test_style_preset_average_engagement():
        preset = StylePreset(name="test", description="Test")
        assert preset.average_engagement == 0.0
        
        preset.usage_count = 10
        preset.total_engagement = 50.0
        assert preset.average_engagement == 5.0

    def test_style_manager_init():
        manager = StyleManager()
        presets = manager.list_presets()
        assert len(presets) >= 5
        assert "technical_professional" in presets

    def test_style_manager_get_preset():
        manager = StyleManager()
        preset = manager.get_preset("technical_professional")
        assert preset is not None
        assert preset.name == "technical_professional"
        
        missing = manager.get_preset("nonexistent")
        assert missing is None

    def test_style_manager_apply():
        manager = StyleManager()
        result = manager.apply_style(
            "photo of a factory",
            style_preset="manufacturing",
        )
        assert "prompt" in result
        assert "factory" in result["prompt"]
        assert "negative_prompt" in result

    def test_style_manager_apply_missing():
        manager = StyleManager()
        result = manager.apply_style(
            "base prompt",
            style_preset="nonexistent",
        )
        assert result["prompt"] == "base prompt"

    def test_style_manager_recommend():
        manager = StyleManager()
        
        # Should recommend sustainability for green topics
        rec = manager.recommend_style(["solar", "renewable", "green"])
        assert rec == "sustainability_green"
        
        # Should recommend manufacturing for factory topics
        rec = manager.recommend_style(["factory", "production"])
        assert rec == "manufacturing"

    def test_style_manager_add_remove():
        manager = StyleManager()
        custom = StylePreset(name="custom", description="Custom style")
        manager.add_preset(custom)
        assert "custom" in manager.list_presets()
        
        manager.remove_preset("custom")
        assert "custom" not in manager.list_presets()

    def test_style_manager_record_engagement():
        manager = StyleManager()
        manager.apply_style("test", style_preset="manufacturing")
        manager.record_engagement("manufacturing", 10.0)
        
        preset = manager.get_preset("manufacturing")
        assert preset is not None
        assert preset.total_engagement >= 10.0

    def test_style_experiment_variant():
        experiment = StyleExperiment(
            name="test",
            variants=[
                StyleVariant("style_a", weight=1.0),
                StyleVariant("style_b", weight=1.0),
            ],
        )
        # Deterministic - same input should give same output
        v1 = experiment.get_variant("content_123")
        v2 = experiment.get_variant("content_123")
        assert v1 == v2

    def test_style_experiment_results():
        experiment = StyleExperiment(
            name="test",
            variants=[StyleVariant("style_a")],
        )
        experiment.record_result("style_a", 10.0)
        experiment.record_result("style_a", 20.0)
        
        summary = experiment.get_summary()
        assert summary["variants"]["style_a"]["samples"] == 2
        assert summary["variants"]["style_a"]["mean"] == 15.0

    def test_create_styled_prompt():
        prompt = create_styled_prompt("test image", "technical_professional")
        assert "test image" in prompt
        assert "professional" in prompt.lower()

    def test_list_available_styles():
        styles = list_available_styles()
        assert isinstance(styles, list)
        assert len(styles) >= 5

    def test_get_style_for_topic():
        style = get_style_for_topic("renewable energy solar panels")
        assert style == "sustainability_green"

    suite.add_test("Color palette description", test_color_palette_description)
    suite.add_test("Style preset apply", test_style_preset_apply)
    suite.add_test("Style preset average engagement", test_style_preset_average_engagement)
    suite.add_test("Style manager init", test_style_manager_init)
    suite.add_test("Style manager get preset", test_style_manager_get_preset)
    suite.add_test("Style manager apply", test_style_manager_apply)
    suite.add_test("Style manager apply missing", test_style_manager_apply_missing)
    suite.add_test("Style manager recommend", test_style_manager_recommend)
    suite.add_test("Style manager add/remove", test_style_manager_add_remove)
    suite.add_test("Style manager record engagement", test_style_manager_record_engagement)
    suite.add_test("Style experiment variant", test_style_experiment_variant)
    suite.add_test("Style experiment results", test_style_experiment_results)
    suite.add_test("Create styled prompt", test_create_styled_prompt)
    suite.add_test("List available styles", test_list_available_styles)
    suite.add_test("Get style for topic", test_get_style_for_topic)

    return suite


if __name__ == "__main__":
    suite = _create_module_tests()
    suite.run()
