"""Unified Web Server for Social Media Publisher.

This module provides a single Flask-based web server that combines:
- Story validation/review interface (from validation_server.py)
- Analytics dashboard (from dashboard.py)

Using Flask Blueprints for clean separation of concerns while sharing:
- Common infrastructure (port management, browser opening)
- Shared static assets and CSS theming
- Single Flask app instance
- Unified shutdown handling

Routes:
- / - Landing page with links to both views
- /review - Story validation/review interface
- /dashboard - Analytics and monitoring dashboard
- /api/stories/* - Story CRUD operations
- /api/dashboard/* - Dashboard metrics and stats
"""

import json
import logging
import socket
import subprocess
import threading
import webbrowser
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import (
    Blueprint,
    Flask,
    jsonify,
    render_template_string,
    request,
    send_from_directory,
)

try:
    from flask_cors import CORS
except ImportError:
    CORS = None  # type: ignore

from config import Config
from dashboard import DashboardMetrics, PipelineStatus
from database import Database

logger = logging.getLogger(__name__)


# =============================================================================
# Shared CSS Theme
# =============================================================================

SHARED_CSS = """
/* Common theme variables and base styles */
:root {
    --bg-primary: #1a1a2e;
    --bg-secondary: #16213e;
    --bg-card: rgba(255,255,255,0.05);
    --text-primary: #e0e0e0;
    --text-secondary: #888;
    --accent-blue: #00d4ff;
    --accent-purple: #8b5cf6;
    --accent-green: #00c853;
    --accent-yellow: #ffc107;
    --accent-red: #ff5252;
    --border-color: rgba(255,255,255,0.1);
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    color: var(--text-primary);
    min-height: 100vh;
}

.btn {
    padding: 8px 16px;
    border: none;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none;
    display: inline-block;
}

.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

.btn-primary {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    color: white;
}

.btn-success {
    background: linear-gradient(135deg, var(--accent-green), #00a845);
    color: white;
}

.btn-danger {
    background: linear-gradient(135deg, var(--accent-red), #d32f2f);
    color: white;
}

.btn-warning {
    background: linear-gradient(135deg, var(--accent-yellow), #ff9800);
    color: var(--bg-primary);
}

.card {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid var(--border-color);
}
"""


# =============================================================================
# Landing Page Template
# =============================================================================

LANDING_TEMPLATE = (
    """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Social Media Publisher</title>
    <style>
        """
    + SHARED_CSS
    + """

        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            text-align: center;
        }

        h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }

        .subtitle {
            color: var(--text-secondary);
            margin-bottom: 40px;
        }

        .nav-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }

        .nav-card {
            padding: 30px;
            text-align: left;
        }

        .nav-card h2 {
            color: var(--accent-blue);
            margin-bottom: 10px;
            font-size: 1.4rem;
        }

        .nav-card p {
            color: var(--text-secondary);
            margin-bottom: 20px;
            line-height: 1.6;
        }

        .nav-card .btn {
            width: 100%;
            text-align: center;
        }

        .stats {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin: 40px 0;
            flex-wrap: wrap;
        }

        .stat {
            text-align: center;
        }

        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--accent-blue);
        }

        .stat-label {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Social Media Publisher</h1>
        <p class="subtitle">{{ discipline_title }} Content Automation Platform</p>

        <div class="stats">
            <div class="stat">
                <div class="stat-value">{{ stats.pending }}</div>
                <div class="stat-label">Pending Review</div>
            </div>
            <div class="stat">
                <div class="stat-value">{{ stats.approved }}</div>
                <div class="stat-label">Approved</div>
            </div>
            <div class="stat">
                <div class="stat-value">{{ stats.published }}</div>
                <div class="stat-label">Published</div>
            </div>
        </div>

        <div class="nav-cards">
            <div class="card nav-card">
                <h2>📝 Story Review</h2>
                <p>Review pending stories, edit content, approve or reject for publication.
                   Preview how posts will appear on LinkedIn.</p>
                <a href="/review" class="btn btn-success">Open Review Interface</a>
            </div>

            <div class="card nav-card">
                <h2>📊 Dashboard</h2>
                <p>Monitor pipeline status, view analytics, track enrichment metrics,
                   and export people data.</p>
                <a href="/dashboard" class="btn btn-primary">Open Dashboard</a>
            </div>
        </div>
    </div>

    <script>
        // Auto-refresh stats every 30 seconds
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
"""
)


# =============================================================================
# Blueprint Factory Functions
# =============================================================================
# Each factory creates a NEW blueprint instance with routes attached.
# This avoids the Flask error about adding routes after registration.
# =============================================================================


def create_review_blueprint(db: Database) -> Blueprint:
    """Create and configure the review blueprint."""
    bp = Blueprint("review", __name__, url_prefix="/review")

    # Import the HTML template from validation_server
    try:
        from validation_server import HTML_TEMPLATE as REVIEW_TEMPLATE
    except ImportError:
        REVIEW_TEMPLATE = "<h1>Review interface not available</h1>"

    author_name = Config.LINKEDIN_AUTHOR_NAME or "Author"
    author_initial = author_name[0].upper() if author_name else "A"

    @bp.route("/")
    def index():
        html = REVIEW_TEMPLATE.replace("{{ author_name }}", author_name)
        html = html.replace("{{ author_initial }}", author_initial)
        html = html.replace("{{ discipline_title }}", Config.DISCIPLINE.title())
        return render_template_string(html)

    return bp


def create_dashboard_blueprint(
    db: Database, pipeline_status: PipelineStatus
) -> Blueprint:
    """Create and configure the dashboard blueprint."""
    bp = Blueprint("dashboard", __name__, url_prefix="/dashboard")

    try:
        from dashboard import DASHBOARD_TEMPLATE
    except ImportError:
        DASHBOARD_TEMPLATE = "<h1>Dashboard not available</h1>"

    @bp.route("/")
    def index():
        return render_template_string(DASHBOARD_TEMPLATE)

    return bp


def create_stories_api_blueprint(db: Database) -> Blueprint:
    """Create and configure the stories API blueprint."""
    bp = Blueprint("stories_api", __name__, url_prefix="/api/stories")

    def _row_to_dict(row) -> dict:
        """Convert a database row to a dictionary."""

        def parse_json_field(value):
            if value is None:
                return []
            if isinstance(value, list):
                return value
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return []

        def safe_get(row, key, default=None):
            try:
                return row[key] if row[key] is not None else default
            except (KeyError, IndexError):
                return default

        return {
            "id": row["id"],
            "title": row["title"],
            "summary": row["summary"],
            "source_links": parse_json_field(safe_get(row, "source_links")),
            "acquire_date": safe_get(row, "acquire_date"),
            "quality_score": safe_get(row, "quality_score", 0),
            "quality_justification": safe_get(row, "quality_justification", ""),
            "category": safe_get(row, "category", "Other"),
            "hashtags": parse_json_field(safe_get(row, "hashtags")),
            "image_path": safe_get(row, "image_path"),
            "image_alt_text": safe_get(row, "image_alt_text"),
            "verification_status": safe_get(row, "verification_status", "pending"),
            "verification_reason": safe_get(row, "verification_reason"),
            "scheduled_time": safe_get(row, "scheduled_time"),
            "publish_status": safe_get(row, "publish_status", "unpublished"),
            "direct_people": parse_json_field(safe_get(row, "direct_people")),
            "indirect_people": parse_json_field(safe_get(row, "indirect_people")),
            "promotion": safe_get(row, "promotion"),
        }

    @bp.route("/")
    def get_stories():
        """Get all stories for review."""
        try:
            with db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM stories
                    ORDER BY
                        CASE WHEN verification_status = 'pending' THEN 0 ELSE 1 END,
                        acquire_date DESC
                """)
                rows = cursor.fetchall()
            return jsonify([_row_to_dict(row) for row in rows])
        except Exception as e:
            logger.exception("Failed to get stories")
            return jsonify({"error": str(e)}), 500

    @bp.route("/<int:story_id>", methods=["PUT"])
    def update_story(story_id: int):
        """Update story fields."""
        try:
            data = request.get_json()
            story = db.get_story(story_id)
            if not story:
                return jsonify({"error": "Story not found"}), 404

            if "title" in data:
                story.title = data["title"]
            if "summary" in data:
                story.summary = data["summary"]
            if "hashtags" in data:
                story.hashtags = data["hashtags"]
            if "promotion" in data:
                story.promotion = data["promotion"]
            if "scheduled_time" in data:
                if data["scheduled_time"]:
                    story.scheduled_time = datetime.fromisoformat(
                        data["scheduled_time"]
                    )
                else:
                    story.scheduled_time = None

            db.update_story(story)
            return jsonify(story.to_dict())
        except Exception as e:
            logger.exception(f"Failed to update story {story_id}")
            return jsonify({"error": str(e)}), 500

    @bp.route("/<int:story_id>/accept", methods=["POST"])
    def accept_story(story_id: int):
        """Accept a story for publication (human approval)."""
        try:
            story = db.get_story(story_id)
            if not story:
                return jsonify({"error": "Story not found"}), 404
            story.verification_status = "approved"
            story.verification_reason = "Manually approved via human validation"
            story.human_approved = True
            story.human_approved_at = datetime.now()
            db.update_story(story)
            return jsonify(story.to_dict())
        except Exception as e:
            logger.exception(f"Failed to accept story {story_id}")
            return jsonify({"error": str(e)}), 500

    @bp.route("/<int:story_id>/reject", methods=["POST"])
    def reject_story(story_id: int):
        """Reject a story (human rejection)."""
        try:
            story = db.get_story(story_id)
            if not story:
                return jsonify({"error": "Story not found"}), 404
            story.verification_status = "rejected"
            story.verification_reason = "Manually rejected via human validation"
            story.human_approved = False
            story.human_approved_at = None
            db.update_story(story)
            return jsonify(story.to_dict())
        except Exception as e:
            logger.exception(f"Failed to reject story {story_id}")
            return jsonify({"error": str(e)}), 500

    @bp.route("/<int:story_id>/validate-publish", methods=["POST"])
    def validate_publish(story_id: int):
        """Validate a story before publishing (pre-flight check)."""
        logger.info(f"Validate-publish request for story {story_id}")
        try:
            story = db.get_story(story_id)
            if not story:
                logger.warning(f"Story {story_id} not found")
                return jsonify({"error": "Story not found"}), 404

            from linkedin_publisher import LinkedInPublisher

            logger.info("Creating LinkedInPublisher...")
            publisher = LinkedInPublisher(db)
            logger.info("Running validate_before_publish...")
            validation = publisher.validate_before_publish(story)
            logger.info(f"Validation complete: is_valid={validation.is_valid}")

            return jsonify(validation.to_dict())

        except Exception as e:
            logger.exception(f"Failed to validate story {story_id}")
            return jsonify({"error": str(e)}), 500

    @bp.route("/<int:story_id>/publish", methods=["POST"])
    def publish_story(story_id: int):
        """Publish or schedule a story to LinkedIn.

        If the story has a scheduled_time in the future, it will be scheduled.
        Otherwise, it will be published immediately.
        """
        try:
            story = db.get_story(story_id)
            if not story:
                return jsonify({"error": "Story not found"}), 404
            # Require human approval, not just AI approval
            if not story.human_approved:
                return jsonify(
                    {"error": "Story must be human-approved before publishing"}
                ), 400
            if story.publish_status == "published":
                return jsonify({"error": "Story is already published"}), 400

            from datetime import datetime

            from linkedin_publisher import LinkedInPublisher

            publisher = LinkedInPublisher(db)

            # Check if this should be scheduled or published immediately
            now = datetime.now()
            if story.scheduled_time and story.scheduled_time > now:
                # Schedule for later - just mark as scheduled
                story.publish_status = "scheduled"
                db.update_story(story)
                result = story.to_dict()
                result["success"] = True
                result["scheduled"] = True
                result["scheduled_time"] = story.scheduled_time.isoformat()
                return jsonify(result)

            # Publish immediately
            post_id = publisher.publish_immediately(story)

            if post_id:
                story = db.get_story(story_id)
                if story:
                    result = story.to_dict()
                    result["success"] = True
                    result["scheduled"] = False
                    return jsonify(result)
                return jsonify(
                    {"success": True, "scheduled": False, "linkedin_post_id": post_id}
                )
            else:
                return jsonify({"error": "Failed to publish to LinkedIn"}), 500

        except Exception as e:
            logger.exception(f"Failed to publish story {story_id}")
            return jsonify({"error": str(e)}), 500

    return bp


def create_dashboard_api_blueprint(
    db: Database, pipeline_status: PipelineStatus, lock: threading.Lock
) -> Blueprint:
    """Create and configure the dashboard API blueprint."""
    bp = Blueprint("dashboard_api", __name__, url_prefix="/api/dashboard")

    def _calculate_metrics() -> DashboardMetrics:
        """Calculate dashboard metrics from database."""
        try:
            all_stories = db.get_stories_needing_verification()
            published = db.get_published_stories()
            scheduled = db.get_scheduled_stories()
            stories = all_stories + published + scheduled

            total = len(stories)
            approved_count = sum(
                1 for s in stories if s.verification_status == "approved"
            )
            published_count = len(published)

            today = datetime.now().date()
            stories_today = sum(
                1 for s in stories if s.acquire_date and s.acquire_date.date() == today
            )

            quality_scores = [s.quality_score for s in stories if s.quality_score]
            avg_quality = (
                sum(quality_scores) / len(quality_scores) if quality_scores else 0
            )

            categories: dict[str, int] = {}
            for s in stories:
                cat = s.category or "Uncategorized"
                categories[cat] = categories.get(cat, 0) + 1

            statuses: dict[str, int] = {
                "pending": sum(
                    1 for s in stories if s.verification_status == "pending"
                ),
                "approved": approved_count,
                "rejected": sum(
                    1 for s in stories if s.verification_status == "rejected"
                ),
                "published": published_count,
                "scheduled": len(scheduled),
            }

            try:
                enrichment_stats = db.get_enrichment_dashboard_stats()
            except Exception:
                enrichment_stats = {}

            return DashboardMetrics(
                total_stories=total,
                stories_today=stories_today,
                avg_quality_score=avg_quality,
                publish_rate=published_count / approved_count
                if approved_count > 0
                else 0,
                approval_rate=approved_count / total if total > 0 else 0,
                stories_by_category=categories,
                stories_by_status=statuses,
                enrichment_stats=enrichment_stats,
            )
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return DashboardMetrics()

    @bp.route("/metrics")
    def get_metrics():
        return jsonify(asdict(_calculate_metrics()))

    @bp.route("/status")
    def get_status():
        with lock:
            return jsonify(asdict(pipeline_status))

    @bp.route("/status", methods=["POST"])
    def update_status():
        data = request.get_json()
        with lock:
            if "status" in data:
                pipeline_status.status = data["status"]
            if "current_stage" in data:
                pipeline_status.current_stage = data["current_stage"]
            if "last_run" in data:
                pipeline_status.last_run = data["last_run"]
            if "next_run" in data:
                pipeline_status.next_run = data["next_run"]
        return jsonify({"success": True})

    @bp.route("/queue")
    def get_queue():
        try:
            stories = db.get_stories_needing_verification()[:20]
            return jsonify(
                [
                    {
                        "id": s.id,
                        "title": s.title or "Untitled",
                        "status": s.verification_status or "pending",
                        "quality_score": s.quality_score,
                        "category": s.category or "Uncategorized",
                    }
                    for s in stories
                ]
            )
        except Exception as e:
            logger.error(f"Error getting story queue: {e}")
            return jsonify([])

    @bp.route("/enrichment")
    def get_enrichment_stats():
        stats = db.get_enrichment_dashboard_stats()
        return jsonify(stats)

    @bp.route("/export/<int:story_id>")
    def export_story(story_id: int):
        from company_mention_enricher import export_direct_people

        format = request.args.get("format", "json")
        story = db.get_story(story_id)
        if not story:
            return jsonify({"error": "Story not found"}), 404

        data = export_direct_people(story, format)

        if format == "csv":
            return data, 200, {"Content-Type": "text/csv"}
        elif format == "markdown":
            return data, 200, {"Content-Type": "text/markdown"}
        else:
            return jsonify(json.loads(data))

    return bp


# =============================================================================
# Unified Web Server
# =============================================================================


class UnifiedWebServer:
    """
    Unified Flask-based web server combining validation and dashboard.

    Features:
    - Single Flask app with blueprints for modular routing
    - Shared CSS theming and infrastructure
    - Auto port detection (finds available port)
    - Auto browser opening (Edge app mode preferred)
    - Clean shutdown handling
    """

    def __init__(self, database: Database, port: int = 5000) -> None:
        """Initialize the unified web server.

        Args:
            database: Database instance for story data
            port: Starting port number (will find available port)
        """
        self.db = database
        self.port = port
        self.app = Flask(__name__)

        if CORS is not None:
            CORS(self.app)

        # Pipeline status tracking
        self._pipeline_status = PipelineStatus()
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

        # Set up all routes
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Set up Flask routes and register blueprints."""

        # Landing page
        @self.app.route("/")
        def index():
            stats = self._get_quick_stats()
            return render_template_string(
                LANDING_TEMPLATE,
                discipline_title=Config.DISCIPLINE.title(),
                stats=stats,
            )

        # Favicon
        @self.app.route("/favicon.ico")
        def favicon():
            return "", 204

        # Image serving
        @self.app.route("/image/<path:image_path>")
        def serve_image(image_path: str):
            try:
                normalized_path = image_path.replace("\\", "/")
                image_file = Path(normalized_path)
                if not image_file.is_absolute():
                    image_file = Path(__file__).parent / normalized_path

                if image_file.exists():
                    return send_from_directory(
                        str(image_file.parent), image_file.name, mimetype="image/png"
                    )
                return "", 404
            except Exception:
                return "", 404

        # Shutdown endpoint
        @self.app.route("/api/shutdown", methods=["POST"])
        def shutdown():
            self._shutdown_event.set()
            return jsonify({"status": "shutting down"})

        # Create and register blueprints
        # Each factory creates a fresh blueprint with routes attached
        review_bp = create_review_blueprint(self.db)
        dashboard_bp = create_dashboard_blueprint(self.db, self._pipeline_status)
        stories_api_bp = create_stories_api_blueprint(self.db)
        dashboard_api_bp = create_dashboard_api_blueprint(
            self.db, self._pipeline_status, self._lock
        )

        self.app.register_blueprint(review_bp)
        self.app.register_blueprint(dashboard_bp)
        self.app.register_blueprint(stories_api_bp)
        self.app.register_blueprint(dashboard_api_bp)

    def _get_quick_stats(self) -> dict:
        """Get quick stats for landing page."""
        try:
            pending = len(self.db.get_stories_needing_verification())
            approved = len(self.db.get_approved_unpublished_stories(limit=1000))
            published = len(self.db.get_published_stories())
            return {"pending": pending, "approved": approved, "published": published}
        except Exception:
            return {"pending": 0, "approved": 0, "published": 0}

    def _find_available_port(self) -> bool:
        """Find an available port."""
        original_port = self.port
        for attempt in range(5):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("localhost", self.port))
                sock.close()
                if self.port != original_port:
                    print(f"   Using port {self.port} instead of {original_port}")
                return True
            except OSError:
                if attempt == 0:
                    print(f"   Port {self.port} is in use, trying another...")
                self.port += 1
        return False

    def _open_browser(self, path: str = "") -> None:
        """Open the server in a browser window."""
        url = f"http://localhost:{self.port}{path}"
        try:
            edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
            if not Path(edge_path).exists():
                edge_path = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
            if Path(edge_path).exists():
                subprocess.Popen(
                    [edge_path, f"--app={url}", "--start-maximized"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                webbrowser.open(url)
        except Exception as e:
            logger.warning(f"Could not open Edge: {e}")
            webbrowser.open(url)

    def run(
        self,
        view: str = "",
        open_browser: bool = True,
        debug: bool = False,
    ) -> None:
        """Run the unified web server.

        Args:
            view: Initial view to open ("", "review", or "dashboard")
            open_browser: Whether to open browser automatically
            debug: Enable Flask debug mode
        """
        import werkzeug.serving

        if not self._find_available_port():
            print("   Could not find an available port. Try again later.")
            return

        print(f"\n🌐 Starting web server at http://localhost:{self.port}")
        if open_browser:
            print("   Opening browser...")
        print("   Press Ctrl+C to stop and return to menu.\n")

        if open_browser:
            path = f"/{view}" if view else ""
            self._open_browser(path)

        server = werkzeug.serving.make_server(
            "localhost", self.port, self.app, threaded=True
        )
        server.timeout = 1

        try:
            while not self._shutdown_event.is_set():
                server.handle_request()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            print("   Web server stopped.\n")

    def run_in_thread(self) -> threading.Thread:
        """Run server in background thread."""
        thread = threading.Thread(
            target=lambda: self.run(open_browser=False), daemon=True
        )
        thread.start()
        logger.info(f"Web server started in background on port {self.port}")
        return thread

    def update_pipeline_status(
        self,
        status: Optional[str] = None,
        current_stage: Optional[str] = None,
        last_run: Optional[str] = None,
        next_run: Optional[str] = None,
    ) -> None:
        """Update pipeline status."""
        with self._lock:
            if status is not None:
                self._pipeline_status.status = status
            if current_stage is not None:
                self._pipeline_status.current_stage = current_stage
            if last_run is not None:
                self._pipeline_status.last_run = last_run
            if next_run is not None:
                self._pipeline_status.next_run = next_run


# =============================================================================
# Convenience Functions
# =============================================================================

# Singleton instance
_server: Optional[UnifiedWebServer] = None


def get_web_server(database: Optional[Database] = None) -> UnifiedWebServer:
    """Get or create the unified web server instance."""
    global _server
    if _server is None:
        if database is None:
            raise ValueError("Database required for first initialization")
        _server = UnifiedWebServer(database)
    return _server


def run_validation(database: Database, port: int = 5000) -> None:
    """Run the web server and open validation view."""
    server = UnifiedWebServer(database, port)
    server.run(view="review")


def run_dashboard(database: Database, port: int = 5000) -> None:
    """Run the web server and open dashboard view."""
    server = UnifiedWebServer(database, port)
    server.run(view="dashboard")


# =============================================================================
# Unit Tests
# =============================================================================


def _web_server_tests() -> bool:
    """Run comprehensive tests for the web server module."""
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite("Unified Web Server", "web_server.py")
        suite.start_suite()

        def test_shared_css_contains_theme_variables():
            assert ":root" in SHARED_CSS
            assert "--bg-primary" in SHARED_CSS
            assert "--bg-secondary" in SHARED_CSS

        def test_landing_template_has_navigation_links():
            assert "/review" in LANDING_TEMPLATE
            assert "/dashboard" in LANDING_TEMPLATE
            assert "Social Media Publisher" in LANDING_TEMPLATE

        def test_unified_web_server_init():
            import tempfile
            tmp = tempfile.mkdtemp()
            db = Database(f"{tmp}/test.db")
            server = UnifiedWebServer(db, port=0)
            assert server.db is db
            assert server.port == 0
            assert server.app is not None

        def test_flask_app_has_blueprints():
            import tempfile
            tmp = tempfile.mkdtemp()
            db = Database(f"{tmp}/test.db")
            server = UnifiedWebServer(db, port=0)
            blueprint_names = list(server.app.blueprints.keys())
            assert "review" in blueprint_names
            assert "dashboard" in blueprint_names
            assert "stories_api" in blueprint_names

        def test_landing_page_route():
            import tempfile
            tmp = tempfile.mkdtemp()
            db = Database(f"{tmp}/test.db")
            server = UnifiedWebServer(db, port=0)
            with server.app.test_client() as client:
                resp = client.get("/")
                assert resp.status_code == 200
                html = resp.data.decode()
                assert "Social Media Publisher" in html

        def test_review_route():
            import tempfile
            tmp = tempfile.mkdtemp()
            db = Database(f"{tmp}/test.db")
            server = UnifiedWebServer(db, port=0)
            with server.app.test_client() as client:
                resp = client.get("/review/")
                assert resp.status_code == 200

        def test_dashboard_route():
            import tempfile
            tmp = tempfile.mkdtemp()
            db = Database(f"{tmp}/test.db")
            server = UnifiedWebServer(db, port=0)
            with server.app.test_client() as client:
                resp = client.get("/dashboard/")
                assert resp.status_code == 200

        def test_stories_api_list_empty():
            import tempfile
            tmp = tempfile.mkdtemp()
            db = Database(f"{tmp}/test.db")
            server = UnifiedWebServer(db, port=0)
            with server.app.test_client() as client:
                resp = client.get("/api/stories/")
                assert resp.status_code == 200
                data = resp.get_json()
                assert isinstance(data, list)

        def test_update_pipeline_status():
            import tempfile
            tmp = tempfile.mkdtemp()
            db = Database(f"{tmp}/test.db")
            server = UnifiedWebServer(db, port=0)
            server.update_pipeline_status(status="running", current_stage="search")
            assert server._pipeline_status.status == "running"
            assert server._pipeline_status.current_stage == "search"

        def test_generated_images_route():
            import tempfile
            tmp = tempfile.mkdtemp()
            db = Database(f"{tmp}/test.db")
            server = UnifiedWebServer(db, port=0)
            with server.app.test_client() as client:
                resp = client.get("/generated_images/nonexistent.png")
                assert resp.status_code == 404

        suite.run_test(
            test_name="Shared CSS theme variables",
            test_func=test_shared_css_contains_theme_variables,
            test_summary="Verify SHARED_CSS contains CSS custom properties for theming",
            functions_tested="SHARED_CSS constant",
            expected_outcome="CSS contains :root with --bg-primary and --bg-secondary",
        )
        suite.run_test(
            test_name="Landing template navigation",
            test_func=test_landing_template_has_navigation_links,
            test_summary="Verify landing page template has links to review and dashboard",
            functions_tested="LANDING_TEMPLATE constant",
            expected_outcome="Template contains /review and /dashboard links",
        )
        suite.run_test(
            test_name="UnifiedWebServer initialization",
            test_func=test_unified_web_server_init,
            test_summary="Verify server initializes with database and Flask app",
            functions_tested="UnifiedWebServer.__init__",
            expected_outcome="Server has db reference, port, and Flask app",
        )
        suite.run_test(
            test_name="Flask blueprints registered",
            test_func=test_flask_app_has_blueprints,
            test_summary="Verify all required Flask blueprints are registered",
            functions_tested="UnifiedWebServer.__init__",
            expected_outcome="review, dashboard, and stories_api blueprints exist",
        )
        suite.run_test(
            test_name="Landing page returns 200",
            test_func=test_landing_page_route,
            test_summary="Verify GET / returns 200 with landing page HTML",
            functions_tested="Flask landing route",
            expected_outcome="Response status 200 with 'Social Media Publisher' in body",
        )
        suite.run_test(
            test_name="Review route returns 200",
            test_func=test_review_route,
            test_summary="Verify GET /review/ returns 200",
            functions_tested="create_review_blueprint / route",
            expected_outcome="Response status 200",
        )
        suite.run_test(
            test_name="Dashboard route returns 200",
            test_func=test_dashboard_route,
            test_summary="Verify GET /dashboard/ returns 200",
            functions_tested="create_dashboard_blueprint / route",
            expected_outcome="Response status 200",
        )
        suite.run_test(
            test_name="Stories API returns empty list",
            test_func=test_stories_api_list_empty,
            test_summary="Verify GET /api/stories/ returns JSON list from empty DB",
            functions_tested="create_stories_api_blueprint / list route",
            expected_outcome="Response is 200 with empty JSON list",
        )
        suite.run_test(
            test_name="Update pipeline status",
            test_func=test_update_pipeline_status,
            test_summary="Verify update_pipeline_status changes internal state",
            functions_tested="UnifiedWebServer.update_pipeline_status",
            expected_outcome="Pipeline status and stage are updated",
        )
        suite.run_test(
            test_name="Missing image returns 404",
            test_func=test_generated_images_route,
            test_summary="Verify request for non-existent image returns 404",
            functions_tested="generated_images static file route",
            expected_outcome="Response status 404",
        )

        return suite.finish_suite()


def _create_module_tests() -> bool:
    return _web_server_tests()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _web_server_tests
)