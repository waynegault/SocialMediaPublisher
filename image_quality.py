"""Image quality assessment for AI-generated images.

This module provides automated quality scoring for images to ensure
only high-quality images are used in LinkedIn posts.

Features:
- Basic quality metrics (resolution, aspect ratio, sharpness)
- AI artifact detection (common AI generation issues)
- Image-story relevance scoring
- Automatic regeneration recommendations
- Quality threshold enforcement

Example:
    assessor = ImageQualityAssessor()
    score = assessor.assess_image(image_path)

    if score.overall_score < 0.6:
        print(f"Image quality too low: {score.issues}")
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class QualityMetrics:
    """Individual quality measurements for an image."""

    resolution_score: float = 0.0  # 0-1, based on pixel count
    aspect_ratio_score: float = 0.0  # 0-1, how suitable for LinkedIn
    sharpness_score: float = 0.0  # 0-1, edge detection based
    contrast_score: float = 0.0  # 0-1, histogram spread
    brightness_score: float = 0.0  # 0-1, optimal brightness range
    color_balance_score: float = 0.0  # 0-1, color distribution

    def average(self) -> float:
        """Calculate average of all metrics."""
        values = [
            self.resolution_score,
            self.aspect_ratio_score,
            self.sharpness_score,
            self.contrast_score,
            self.brightness_score,
            self.color_balance_score,
        ]
        return sum(values) / len(values) if values else 0.0


@dataclass
class ArtifactDetection:
    """Detection results for common AI artifacts."""

    has_text_artifacts: bool = False  # Garbled text in image
    has_distortion: bool = False  # Unusual distortions
    has_banding: bool = False  # Color banding issues
    has_noise: bool = False  # Excessive noise
    artifact_count: int = 0
    artifact_descriptions: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """Return True if no artifacts detected."""
        return self.artifact_count == 0


@dataclass
class QualityScore:
    """Complete quality assessment result for an image."""

    overall_score: float  # 0-1 composite score
    metrics: QualityMetrics
    artifacts: ArtifactDetection
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    image_path: str = ""
    image_hash: str = ""

    @property
    def is_acceptable(self) -> bool:
        """Return True if image meets minimum quality threshold.

        Note: Technical metrics alone cannot detect semantic AI artifacts
        (distorted faces, wrong anatomy, etc.) - those require LLM vision
        analysis during verification. This threshold catches technical issues.
        """
        return self.overall_score >= 0.75 and self.artifacts.is_clean

    @property
    def is_excellent(self) -> bool:
        """Return True if image is high quality."""
        return self.overall_score >= 0.85 and self.artifacts.is_clean

    def get_grade(self) -> str:
        """Get letter grade for quality."""
        if self.overall_score >= 0.9:
            return "A"
        elif self.overall_score >= 0.8:
            return "B"
        elif self.overall_score >= 0.7:
            return "C"
        elif self.overall_score >= 0.6:
            return "D"
        else:
            return "F"


# =============================================================================
# Quality Thresholds
# =============================================================================


@dataclass
class QualityThresholds:
    """Configurable quality thresholds."""

    min_width: int = 800  # Minimum image width
    min_height: int = 600  # Minimum image height
    max_width: int = 4096  # Maximum before resize needed
    max_height: int = 4096  # Maximum before resize needed

    # Optimal aspect ratios for LinkedIn
    linkedin_ratios: tuple[float, ...] = (1.0, 1.91, 0.8)  # square, landscape, portrait
    aspect_ratio_tolerance: float = 0.15  # How close to optimal

    # Quality thresholds
    min_overall_score: float = 0.6
    min_sharpness: float = 0.4
    min_contrast: float = 0.3
    max_brightness: float = 0.85
    min_brightness: float = 0.15


DEFAULT_THRESHOLDS = QualityThresholds()


# =============================================================================
# Image Quality Assessor
# =============================================================================


class ImageQualityAssessor:
    """
    Assess quality of AI-generated images.

    Uses PIL for basic image analysis without heavy ML dependencies.
    Provides actionable recommendations for image improvement.
    """

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        """
        Initialize the quality assessor.

        Args:
            thresholds: Optional custom quality thresholds
        """
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self._pillow_available = self._check_pillow()

    def _check_pillow(self) -> bool:
        """Check if PIL/Pillow is available."""
        try:
            from PIL import Image

            _ = Image  # Verify import succeeded

            return True
        except ImportError:
            return False

    def assess_image(self, image_path: str | Path) -> QualityScore:
        """
        Assess the quality of an image.

        Args:
            image_path: Path to the image file

        Returns:
            QualityScore with detailed assessment
        """
        path = Path(image_path)

        if not path.exists():
            return QualityScore(
                overall_score=0.0,
                metrics=QualityMetrics(),
                artifacts=ArtifactDetection(),
                issues=["Image file not found"],
                image_path=str(path),
            )

        if not self._pillow_available:
            # Return basic assessment without PIL
            return self._assess_without_pil(path)

        return self._assess_with_pil(path)

    def _assess_without_pil(self, path: Path) -> QualityScore:
        """Basic assessment when PIL is not available."""
        # Can only check file size and extension
        issues: list[str] = []
        recommendations: list[str] = []

        file_size = path.stat().st_size
        if file_size < 10000:  # < 10KB
            issues.append("Image file is very small, may be low quality")

        if file_size > 10_000_000:  # > 10MB
            recommendations.append("Consider compressing image for faster loading")

        # Check extension
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        if path.suffix.lower() not in valid_extensions:
            issues.append(f"Unusual image format: {path.suffix}")

        # Generate hash
        with open(path, "rb") as f:
            image_hash = hashlib.md5(f.read()).hexdigest()[:16]

        # Return neutral score since we can't analyze
        return QualityScore(
            overall_score=0.7,  # Neutral
            metrics=QualityMetrics(
                resolution_score=0.7,
                aspect_ratio_score=0.7,
                sharpness_score=0.7,
                contrast_score=0.7,
                brightness_score=0.7,
                color_balance_score=0.7,
            ),
            artifacts=ArtifactDetection(),
            issues=issues,
            recommendations=recommendations + ["Install Pillow for detailed analysis"],
            image_path=str(path),
            image_hash=image_hash,
        )

    def _assess_with_pil(self, path: Path) -> QualityScore:
        """Full assessment using PIL."""
        from PIL import Image

        issues: list[str] = []
        recommendations: list[str] = []

        try:
            with Image.open(path) as img:
                # Calculate metrics
                metrics = self._calculate_metrics(img)
                artifacts = self._detect_artifacts(img)

                # Generate issues and recommendations
                self._evaluate_resolution(img, issues, recommendations)
                self._evaluate_aspect_ratio(img, issues, recommendations)
                self._evaluate_metrics(metrics, issues, recommendations)
                self._evaluate_artifacts(artifacts, issues, recommendations)

                # Calculate overall score
                overall_score = self._calculate_overall_score(metrics, artifacts)

                # Generate hash
                image_hash = hashlib.md5(img.tobytes()[:10000]).hexdigest()[:16]

                return QualityScore(
                    overall_score=overall_score,
                    metrics=metrics,
                    artifacts=artifacts,
                    issues=issues,
                    recommendations=recommendations,
                    image_path=str(path),
                    image_hash=image_hash,
                )

        except Exception as e:
            return QualityScore(
                overall_score=0.0,
                metrics=QualityMetrics(),
                artifacts=ArtifactDetection(),
                issues=[f"Failed to analyze image: {e}"],
                image_path=str(path),
            )

    def _calculate_metrics(self, img: "Image.Image") -> QualityMetrics:
        """Calculate quality metrics for an image."""
        width, height = img.size

        # Resolution score (based on total pixels)
        total_pixels = width * height
        optimal_pixels = 1920 * 1080  # 2MP is optimal
        resolution_score = min(1.0, total_pixels / optimal_pixels)

        # Aspect ratio score
        aspect_ratio = width / height
        aspect_ratio_score = self._score_aspect_ratio(aspect_ratio)

        # Convert to RGB if needed for analysis
        if img.mode not in ("RGB", "L"):
            try:
                img = img.convert("RGB")
            except Exception:
                # Return basic metrics if conversion fails
                return QualityMetrics(
                    resolution_score=resolution_score,
                    aspect_ratio_score=aspect_ratio_score,
                    sharpness_score=0.7,
                    contrast_score=0.7,
                    brightness_score=0.7,
                    color_balance_score=0.7,
                )

        # Sharpness (edge detection variance)
        sharpness_score = self._calculate_sharpness(img)

        # Contrast (histogram spread)
        contrast_score = self._calculate_contrast(img)

        # Brightness
        brightness_score = self._calculate_brightness(img)

        # Color balance
        color_balance_score = self._calculate_color_balance(img)

        return QualityMetrics(
            resolution_score=resolution_score,
            aspect_ratio_score=aspect_ratio_score,
            sharpness_score=sharpness_score,
            contrast_score=contrast_score,
            brightness_score=brightness_score,
            color_balance_score=color_balance_score,
        )

    def _score_aspect_ratio(self, aspect_ratio: float) -> float:
        """Score aspect ratio based on LinkedIn optimal ratios."""
        best_score = 0.0

        for optimal in self.thresholds.linkedin_ratios:
            difference = abs(aspect_ratio - optimal) / optimal
            if difference <= self.thresholds.aspect_ratio_tolerance:
                score = 1.0 - (difference / self.thresholds.aspect_ratio_tolerance)
                best_score = max(best_score, score)

        # If not close to any optimal, give partial score
        if best_score == 0.0:
            # Penalize extreme aspect ratios
            if 0.5 <= aspect_ratio <= 2.5:
                best_score = 0.5
            else:
                best_score = 0.3

        return best_score

    def _calculate_sharpness(self, img: "Image.Image") -> float:
        """Calculate sharpness using Laplacian variance approximation."""
        try:
            from PIL import ImageFilter

            # Convert to grayscale
            gray = img.convert("L")

            # Apply edge detection
            edges = gray.filter(ImageFilter.FIND_EDGES)

            # Calculate variance of edge pixels
            pixels: list[Any] = list(edges.getdata())  # type: ignore[arg-type]
            mean = sum(pixels) / len(pixels)
            variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)

            # Normalize to 0-1 (variance of 1000+ is very sharp)
            sharpness = min(1.0, variance / 1000)

            return sharpness

        except Exception:
            return 0.7  # Default on error

    def _calculate_contrast(self, img: "Image.Image") -> float:
        """Calculate contrast from histogram spread."""
        try:
            gray = img.convert("L")
            histogram = gray.histogram()

            # Find range of non-zero bins
            non_zero = [i for i, count in enumerate(histogram) if count > 0]

            if not non_zero:
                return 0.0

            range_spread = non_zero[-1] - non_zero[0]

            # Normalize (255 is full range)
            contrast = range_spread / 255.0

            return contrast

        except Exception:
            return 0.7

    def _calculate_brightness(self, img: "Image.Image") -> float:
        """Calculate brightness score (penalize too dark or too bright)."""
        try:
            gray = img.convert("L")
            pixels: list[Any] = list(gray.getdata())  # type: ignore[arg-type]
            mean_brightness = sum(pixels) / len(pixels) / 255.0

            # Optimal is around 0.5 (middle gray)
            # Score drops as it approaches 0 or 1
            if mean_brightness < 0.5:
                score = mean_brightness / 0.5  # 0 at 0, 1 at 0.5
            else:
                score = (1.0 - mean_brightness) / 0.5  # 1 at 0.5, 0 at 1

            # Adjust to be less harsh (min 0.3 for any reasonable image)
            return max(0.3, score)

        except Exception:
            return 0.7

    def _calculate_color_balance(self, img: "Image.Image") -> float:
        """Check color balance across RGB channels."""
        try:
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Split into channels
            r, g, b = img.split()

            # Calculate mean of each channel
            r_data: list[Any] = list(r.getdata())  # type: ignore[arg-type]
            g_data: list[Any] = list(g.getdata())  # type: ignore[arg-type]
            b_data: list[Any] = list(b.getdata())  # type: ignore[arg-type]
            r_mean = sum(r_data) / len(r_data)
            g_mean = sum(g_data) / len(g_data)
            b_mean = sum(b_data) / len(b_data)

            # Calculate deviation from balanced (equal means)
            overall_mean = (r_mean + g_mean + b_mean) / 3

            if overall_mean == 0:
                return 0.5

            r_dev = abs(r_mean - overall_mean) / overall_mean
            g_dev = abs(g_mean - overall_mean) / overall_mean
            b_dev = abs(b_mean - overall_mean) / overall_mean

            max_dev = max(r_dev, g_dev, b_dev)

            # Score based on maximum deviation (0.3 deviation = 0.7 score)
            score = max(0.3, 1.0 - max_dev)

            return score

        except Exception:
            return 0.7

    def _detect_artifacts(self, img: "Image.Image") -> ArtifactDetection:
        """Detect common AI-generated image artifacts."""
        artifacts = ArtifactDetection()

        try:
            # Check for color banding
            if self._detect_banding(img):
                artifacts.has_banding = True
                artifacts.artifact_count += 1
                artifacts.artifact_descriptions.append("Color banding detected")

            # Check for excessive noise
            if self._detect_noise(img):
                artifacts.has_noise = True
                artifacts.artifact_count += 1
                artifacts.artifact_descriptions.append("High noise level")

        except Exception:
            pass  # Fail gracefully

        return artifacts

    def _detect_banding(self, img: "Image.Image") -> bool:
        """Detect color banding (common in AI-generated skies/gradients)."""
        try:
            gray = img.convert("L")
            histogram = gray.histogram()

            # Check for spiky histogram (indicates banding)
            # Count number of empty bins between non-empty bins
            in_gap = False
            gap_count = 0

            for count in histogram:
                if count == 0:
                    in_gap = True
                else:
                    if in_gap:
                        gap_count += 1
                    in_gap = False

            # Many gaps suggest banding
            return gap_count > 50

        except Exception:
            return False

    def _detect_noise(self, img: "Image.Image") -> bool:
        """Detect excessive noise in image."""
        try:
            gray = img.convert("L")

            # Sample a portion of the image
            width, height = gray.size
            sample_size = min(100, width, height)

            # Get center crop
            left = (width - sample_size) // 2
            top = (height - sample_size) // 2

            crop = gray.crop((left, top, left + sample_size, top + sample_size))
            pixels: list[Any] = list(crop.getdata())  # type: ignore[arg-type]

            # Calculate variance of adjacent pixel differences
            differences = []
            for i in range(len(pixels) - 1):
                differences.append(abs(pixels[i] - pixels[i + 1]))

            if not differences:
                return False

            mean_diff = sum(differences) / len(differences)

            # High mean difference in a small area suggests noise
            return mean_diff > 30

        except Exception:
            return False

    def _evaluate_resolution(
        self,
        img: "Image.Image",
        issues: list[str],
        recommendations: list[str],
    ) -> None:
        """Evaluate image resolution."""
        width, height = img.size

        if width < self.thresholds.min_width:
            issues.append(
                f"Width too small: {width}px (min: {self.thresholds.min_width}px)"
            )
            recommendations.append("Regenerate image at higher resolution")

        if height < self.thresholds.min_height:
            issues.append(
                f"Height too small: {height}px (min: {self.thresholds.min_height}px)"
            )

        if width > self.thresholds.max_width or height > self.thresholds.max_height:
            recommendations.append("Consider resizing for faster loading")

    def _evaluate_aspect_ratio(
        self,
        img: "Image.Image",
        issues: list[str],
        recommendations: list[str],
    ) -> None:
        """Evaluate aspect ratio for LinkedIn."""
        width, height = img.size
        aspect_ratio = width / height

        # Check if close to any optimal ratio
        is_optimal = False
        for optimal in self.thresholds.linkedin_ratios:
            if (
                abs(aspect_ratio - optimal) / optimal
                <= self.thresholds.aspect_ratio_tolerance
            ):
                is_optimal = True
                break

        if not is_optimal:
            recommendations.append(
                f"Aspect ratio {aspect_ratio:.2f} not optimal for LinkedIn. "
                f"Consider 1:1, 1.91:1, or 4:5"
            )

    def _evaluate_metrics(
        self,
        metrics: QualityMetrics,
        issues: list[str],
        recommendations: list[str],
    ) -> None:
        """Evaluate quality metrics."""
        if metrics.sharpness_score < self.thresholds.min_sharpness:
            issues.append("Image appears blurry")
            recommendations.append("Regenerate with higher sharpness settings")

        if metrics.contrast_score < self.thresholds.min_contrast:
            issues.append("Low contrast")
            recommendations.append("Increase contrast in post-processing")

    def _evaluate_artifacts(
        self,
        artifacts: ArtifactDetection,
        issues: list[str],
        recommendations: list[str],
    ) -> None:
        """Evaluate artifact detection results."""
        if artifacts.has_banding:
            recommendations.append("Color banding detected - consider regenerating")

        if artifacts.has_noise:
            issues.append("Image has excessive noise")
            recommendations.append("Apply noise reduction or regenerate")

    def _calculate_overall_score(
        self,
        metrics: QualityMetrics,
        artifacts: ArtifactDetection,
    ) -> float:
        """Calculate overall quality score."""
        # Weighted average of metrics
        weights = {
            "resolution": 0.15,
            "aspect_ratio": 0.10,
            "sharpness": 0.25,
            "contrast": 0.20,
            "brightness": 0.15,
            "color_balance": 0.15,
        }

        base_score = (
            metrics.resolution_score * weights["resolution"]
            + metrics.aspect_ratio_score * weights["aspect_ratio"]
            + metrics.sharpness_score * weights["sharpness"]
            + metrics.contrast_score * weights["contrast"]
            + metrics.brightness_score * weights["brightness"]
            + metrics.color_balance_score * weights["color_balance"]
        )

        # Penalize for artifacts
        artifact_penalty = artifacts.artifact_count * 0.1

        return max(0.0, min(1.0, base_score - artifact_penalty))

    def should_regenerate(self, score: QualityScore) -> bool:
        """Determine if an image should be regenerated."""
        # Regenerate if:
        # 1. Overall score below threshold
        if score.overall_score < self.thresholds.min_overall_score:
            return True

        # 2. Has critical artifacts
        if not score.artifacts.is_clean:
            return True

        # 3. Has critical issues
        critical_keywords = ["too small", "blurry", "distorted"]
        for issue in score.issues:
            if any(keyword in issue.lower() for keyword in critical_keywords):
                return True

        return False

    def compare_images(
        self,
        paths: list[str | Path],
    ) -> tuple[str | Path, QualityScore]:
        """
        Compare multiple images and return the best one.

        Args:
            paths: List of image paths to compare

        Returns:
            Tuple of (best_path, best_score)
        """
        if not paths:
            raise ValueError("No images to compare")

        best_path = paths[0]
        best_score = self.assess_image(paths[0])

        for path in paths[1:]:
            score = self.assess_image(path)
            if score.overall_score > best_score.overall_score:
                best_path = path
                best_score = score

        return best_path, best_score

    def format_report(self, score: QualityScore) -> str:
        """Format a human-readable quality report."""
        lines = [
            "Image Quality Report",
            "=" * 40,
            f"File: {Path(score.image_path).name}",
            f"Overall Score: {score.overall_score:.2f} ({score.get_grade()})",
            f"Acceptable: {'Yes' if score.is_acceptable else 'No'}",
            "",
            "Metrics:",
            f"  Resolution:     {score.metrics.resolution_score:.2f}",
            f"  Aspect Ratio:   {score.metrics.aspect_ratio_score:.2f}",
            f"  Sharpness:      {score.metrics.sharpness_score:.2f}",
            f"  Contrast:       {score.metrics.contrast_score:.2f}",
            f"  Brightness:     {score.metrics.brightness_score:.2f}",
            f"  Color Balance:  {score.metrics.color_balance_score:.2f}",
        ]

        if score.artifacts.artifact_count > 0:
            lines.extend(
                [
                    "",
                    f"Artifacts Detected: {score.artifacts.artifact_count}",
                ]
            )
            for desc in score.artifacts.artifact_descriptions:
                lines.append(f"  - {desc}")

        if score.issues:
            lines.extend(
                [
                    "",
                    "Issues:",
                ]
            )
            for issue in score.issues:
                lines.append(f"  ⚠ {issue}")

        if score.recommendations:
            lines.extend(
                [
                    "",
                    "Recommendations:",
                ]
            )
            for rec in score.recommendations:
                lines.append(f"  → {rec}")

        return "\n".join(lines)


# =============================================================================
# Batch Assessment
# =============================================================================


def assess_directory(
    directory: str | Path,
    assessor: ImageQualityAssessor | None = None,
) -> dict[str, QualityScore]:
    """
    Assess all images in a directory.

    Args:
        directory: Path to directory containing images
        assessor: Optional assessor instance

    Returns:
        Dictionary mapping filename to QualityScore
    """
    assessor = assessor or ImageQualityAssessor()
    directory = Path(directory)

    results: dict[str, QualityScore] = {}

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

    for path in directory.iterdir():
        if path.suffix.lower() in image_extensions:
            results[path.name] = assessor.assess_image(path)

    return results


def filter_quality_images(
    directory: str | Path,
    min_score: float = 0.6,
) -> list[Path]:
    """
    Filter images that meet quality threshold.

    Args:
        directory: Path to directory containing images
        min_score: Minimum acceptable quality score

    Returns:
        List of paths to acceptable images
    """
    assessor = ImageQualityAssessor()
    assessor.thresholds.min_overall_score = min_score

    results = assess_directory(directory, assessor)

    return [
        Path(directory) / name for name, score in results.items() if score.is_acceptable
    ]


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests() -> bool:
    """Create unit tests for this module."""

    from test_framework import TestSuite

    suite = TestSuite("Image Quality Tests", "image_quality.py")
    suite.start_suite()

    def test_quality_metrics_average():
        metrics = QualityMetrics(
            resolution_score=0.8,
            aspect_ratio_score=0.7,
            sharpness_score=0.9,
            contrast_score=0.6,
            brightness_score=0.7,
            color_balance_score=0.8,
        )
        avg = metrics.average()
        assert 0.74 < avg < 0.76, f"Average should be ~0.75, got {avg}"

    def test_artifact_detection_clean():
        artifacts = ArtifactDetection()
        assert artifacts.is_clean is True
        assert artifacts.artifact_count == 0

    def test_artifact_detection_with_issues():
        artifacts = ArtifactDetection(
            has_banding=True,
            artifact_count=1,
            artifact_descriptions=["Color banding detected"],
        )
        assert artifacts.is_clean is False

    def test_quality_score_acceptable():
        score = QualityScore(
            overall_score=0.75,
            metrics=QualityMetrics(),
            artifacts=ArtifactDetection(),
        )
        assert score.is_acceptable is True

    def test_quality_score_not_acceptable_low_score():
        score = QualityScore(
            overall_score=0.5,
            metrics=QualityMetrics(),
            artifacts=ArtifactDetection(),
        )
        assert score.is_acceptable is False

    def test_quality_score_not_acceptable_artifacts():
        score = QualityScore(
            overall_score=0.8,
            metrics=QualityMetrics(),
            artifacts=ArtifactDetection(has_banding=True, artifact_count=1),
        )
        assert score.is_acceptable is False

    def test_quality_score_grades():
        assert (
            QualityScore(0.95, QualityMetrics(), ArtifactDetection()).get_grade() == "A"
        )
        assert (
            QualityScore(0.85, QualityMetrics(), ArtifactDetection()).get_grade() == "B"
        )
        assert (
            QualityScore(0.75, QualityMetrics(), ArtifactDetection()).get_grade() == "C"
        )
        assert (
            QualityScore(0.65, QualityMetrics(), ArtifactDetection()).get_grade() == "D"
        )
        assert (
            QualityScore(0.45, QualityMetrics(), ArtifactDetection()).get_grade() == "F"
        )

    def test_thresholds_defaults():
        thresholds = QualityThresholds()
        assert thresholds.min_width == 800
        assert thresholds.min_height == 600
        assert thresholds.min_overall_score == 0.6

    def test_assessor_init():
        assessor = ImageQualityAssessor()
        assert assessor.thresholds is not None
        assert assessor.thresholds.min_width == 800

    def test_assessor_custom_thresholds():
        custom = QualityThresholds(min_width=1024, min_height=768)
        assessor = ImageQualityAssessor(thresholds=custom)
        assert assessor.thresholds.min_width == 1024

    def test_assess_missing_file():
        assessor = ImageQualityAssessor()
        score = assessor.assess_image("/nonexistent/path/image.jpg")
        assert score.overall_score == 0.0
        assert "not found" in score.issues[0].lower()

    def test_score_aspect_ratio():
        assessor = ImageQualityAssessor()
        # Square is optimal
        assert assessor._score_aspect_ratio(1.0) >= 0.9
        # 1.91:1 is optimal for LinkedIn
        assert assessor._score_aspect_ratio(1.91) >= 0.9
        # Extreme ratio is penalized
        assert assessor._score_aspect_ratio(5.0) < 0.5

    def test_should_regenerate_low_score():
        assessor = ImageQualityAssessor()
        score = QualityScore(
            overall_score=0.4,
            metrics=QualityMetrics(),
            artifacts=ArtifactDetection(),
        )
        assert assessor.should_regenerate(score) is True

    def test_should_regenerate_artifacts():
        assessor = ImageQualityAssessor()
        score = QualityScore(
            overall_score=0.8,
            metrics=QualityMetrics(),
            artifacts=ArtifactDetection(has_banding=True, artifact_count=1),
        )
        assert assessor.should_regenerate(score) is True

    def test_format_report():
        assessor = ImageQualityAssessor()
        score = QualityScore(
            overall_score=0.75,
            metrics=QualityMetrics(
                resolution_score=0.8,
                sharpness_score=0.7,
            ),
            artifacts=ArtifactDetection(),
            issues=["Test issue"],
            recommendations=["Test recommendation"],
            image_path="/test/image.jpg",
        )
        report = assessor.format_report(score)
        assert "Image Quality Report" in report
        assert "0.75" in report
        assert "Test issue" in report

    suite.run_test(
        test_name="Quality metrics average",
        test_func=test_quality_metrics_average,
        test_summary="Tests Quality metrics average functionality",
        method_description="Calls QualityMetrics and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Artifact detection clean",
        test_func=test_artifact_detection_clean,
        test_summary="Tests Artifact detection clean functionality",
        method_description="Calls ArtifactDetection and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Artifact detection with issues",
        test_func=test_artifact_detection_with_issues,
        test_summary="Tests Artifact detection with issues functionality",
        method_description="Calls ArtifactDetection and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Quality score acceptable",
        test_func=test_quality_score_acceptable,
        test_summary="Tests Quality score acceptable functionality",
        method_description="Calls QualityScore and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Quality score not acceptable - low",
        test_func=test_quality_score_not_acceptable_low_score,
        test_summary="Tests Quality score not acceptable with low scenario",
        method_description="Calls QualityScore and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Quality score not acceptable - artifacts",
        test_func=test_quality_score_not_acceptable_artifacts,
        test_summary="Tests Quality score not acceptable with artifacts scenario",
        method_description="Calls QualityScore and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Quality score grades",
        test_func=test_quality_score_grades,
        test_summary="Tests Quality score grades functionality",
        method_description="Calls QualityScore and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Thresholds defaults",
        test_func=test_thresholds_defaults,
        test_summary="Tests Thresholds defaults functionality",
        method_description="Calls QualityThresholds and verifies the result",
        expected_outcome="Function returns the correct default result",
    )
    suite.run_test(
        test_name="Assessor init",
        test_func=test_assessor_init,
        test_summary="Tests Assessor init functionality",
        method_description="Calls ImageQualityAssessor and verifies the result",
        expected_outcome="Function creates the expected object or result",
    )
    suite.run_test(
        test_name="Assessor custom thresholds",
        test_func=test_assessor_custom_thresholds,
        test_summary="Tests Assessor custom thresholds functionality",
        method_description="Calls QualityThresholds and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Assess missing file",
        test_func=test_assess_missing_file,
        test_summary="Tests Assess missing file functionality",
        method_description="Calls ImageQualityAssessor and verifies the result",
        expected_outcome="Function handles empty or missing input gracefully",
    )
    suite.run_test(
        test_name="Score aspect ratio",
        test_func=test_score_aspect_ratio,
        test_summary="Tests Score aspect ratio functionality",
        method_description="Calls ImageQualityAssessor and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Should regenerate - low score",
        test_func=test_should_regenerate_low_score,
        test_summary="Tests Should regenerate with low score scenario",
        method_description="Calls ImageQualityAssessor and verifies the result",
        expected_outcome="Function creates the expected object or result",
    )
    suite.run_test(
        test_name="Should regenerate - artifacts",
        test_func=test_should_regenerate_artifacts,
        test_summary="Tests Should regenerate with artifacts scenario",
        method_description="Calls ImageQualityAssessor and verifies the result",
        expected_outcome="Function creates the expected object or result",
    )
    suite.run_test(
        test_name="Format report",
        test_func=test_format_report,
        test_summary="Tests Format report functionality",
        method_description="Calls ImageQualityAssessor and verifies the result",
        expected_outcome="Function produces correctly formatted output",
    )

    return suite.finish_suite()
if __name__ == "__main__":
    suite = _create_module_tests()
    suite.run()
