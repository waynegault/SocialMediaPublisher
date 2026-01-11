"""Web Dashboard for Social Media Publisher.

This module provides a real-time web dashboard for monitoring
and managing the Social Media Publisher pipeline.

TASK 8.4: Web Dashboard

Features:
- Real-time pipeline status
- Analytics visualization
- Story queue management
- Historical performance charts
"""

import logging
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Optional

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS  # type: ignore

from database import Database

logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PipelineStatus:
    """Current status of the content pipeline."""

    status: str = "idle"  # idle, running, paused, error
    current_stage: str = ""
    stories_pending: int = 0
    stories_approved: int = 0
    stories_rejected: int = 0
    stories_published: int = 0
    stories_scheduled: int = 0
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    errors: list[str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


@dataclass
class DashboardMetrics:
    """Dashboard metrics for visualization."""

    total_stories: int = 0
    stories_today: int = 0
    avg_quality_score: float = 0.0
    publish_rate: float = 0.0  # Published/Approved ratio
    approval_rate: float = 0.0  # Approved/Total ratio
    stories_by_category: dict[str, int] = None  # type: ignore
    stories_by_status: dict[str, int] = None  # type: ignore
    daily_activity: list[dict[str, Any]] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.stories_by_category is None:
            self.stories_by_category = {}
        if self.stories_by_status is None:
            self.stories_by_status = {}
        if self.daily_activity is None:
            self.daily_activity = []


# =============================================================================
# Dashboard Server
# =============================================================================

# HTML Dashboard Template
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Social Media Publisher Dashboard</title>
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect x='10' y='15' width='80' height='70' rx='8' fill='%231a1a2e' stroke='%2300d4ff' stroke-width='3'/><rect x='15' y='25' width='30' height='20' rx='4' fill='%2300d4ff'/><rect x='55' y='25' width='30' height='8' rx='2' fill='%238b5cf6'/><rect x='55' y='37' width='20' height='8' rx='2' fill='%238b5cf6' opacity='0.6'/><rect x='15' y='55' width='70' height='25' rx='4' fill='%2300c853' opacity='0.3'/><path d='M25 67.5 L40 67.5 L50 60 L60 72 L70 65' stroke='%2300c853' stroke-width='3' fill='none'/></svg>">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }

        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        header h1 {
            font-size: 1.8rem;
            background: linear-gradient(135deg, #00d4ff, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 16px;
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        .status-dot.idle { background: #888; }
        .status-dot.running { background: #00c853; }
        .status-dot.paused { background: #ffc107; }
        .status-dot.error { background: #ff5252; }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .metric-card {
            background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.08);
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-4px);
            border-color: rgba(0,212,255,0.3);
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        }

        .metric-card .label {
            font-size: 0.85rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }

        .metric-card .value {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00d4ff, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .metric-card .trend {
            font-size: 0.8rem;
            margin-top: 8px;
            color: #00c853;
        }

        .metric-card .trend.negative { color: #ff5252; }

        .panels-row {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }

        @media (max-width: 968px) {
            .panels-row {
                grid-template-columns: 1fr;
            }
        }

        .panel {
            background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .panel h2 {
            font-size: 1.1rem;
            color: #00d4ff;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .story-queue {
            max-height: 400px;
            overflow-y: auto;
        }

        .story-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            margin-bottom: 10px;
            transition: all 0.2s ease;
        }

        .story-item:hover {
            background: rgba(0,0,0,0.3);
        }

        .story-item .title {
            font-weight: 500;
            flex: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-right: 10px;
        }

        .story-item .status {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .status-pending { background: #e6a800; color: #1a1a2e; }
        .status-approved { background: #00a844; color: white; }
        .status-rejected { background: #ff5252; color: white; }
        .status-scheduled { background: #2196f3; color: white; }
        .status-published { background: #9c27b0; color: white; }

        .chart-container {
            height: 200px;
            position: relative;
        }

        .chart-placeholder {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #666;
            font-style: italic;
        }

        .category-bars {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .category-bar {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .category-bar .name {
            width: 120px;
            font-size: 0.85rem;
            color: #888;
        }

        .category-bar .bar {
            flex: 1;
            height: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            overflow: hidden;
        }

        .category-bar .bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #8b5cf6);
            border-radius: 10px;
            transition: width 0.5s ease;
        }

        .category-bar .count {
            width: 40px;
            text-align: right;
            font-weight: 600;
            color: #00d4ff;
        }

        .refresh-btn {
            padding: 8px 16px;
            background: linear-gradient(135deg, #00d4ff, #0099cc);
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .refresh-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,212,255,0.3);
        }

        .last-updated {
            font-size: 0.8rem;
            color: #666;
        }

        .activity-log {
            max-height: 300px;
            overflow-y: auto;
        }

        .activity-item {
            display: flex;
            gap: 10px;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }

        .activity-item .time {
            font-size: 0.8rem;
            color: #666;
            width: 60px;
        }

        .activity-item .message {
            font-size: 0.9rem;
            flex: 1;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <header>
            <h1>📊 Social Media Publisher Dashboard</h1>
            <div style="display: flex; align-items: center; gap: 20px;">
                <div class="status-indicator">
                    <div class="status-dot" id="statusDot"></div>
                    <span id="statusText">Loading...</span>
                </div>
                <button class="refresh-btn" onclick="refreshData()">↻ Refresh</button>
            </div>
        </header>

        <div class="metrics-grid" id="metricsGrid">
            <div class="metric-card">
                <div class="label">Total Stories</div>
                <div class="value" id="totalStories">-</div>
            </div>
            <div class="metric-card">
                <div class="label">Pending Review</div>
                <div class="value" id="pendingStories">-</div>
            </div>
            <div class="metric-card">
                <div class="label">Published</div>
                <div class="value" id="publishedStories">-</div>
            </div>
            <div class="metric-card">
                <div class="label">Approval Rate</div>
                <div class="value" id="approvalRate">-</div>
            </div>
            <div class="metric-card">
                <div class="label">Avg Quality Score</div>
                <div class="value" id="avgQuality">-</div>
            </div>
            <div class="metric-card">
                <div class="label">Stories Today</div>
                <div class="value" id="storiesToday">-</div>
            </div>
        </div>

        <div class="panels-row">
            <div class="panel">
                <h2>📋 Story Queue</h2>
                <div class="story-queue" id="storyQueue">
                    <div class="chart-placeholder">Loading stories...</div>
                </div>
            </div>

            <div class="panel">
                <h2>📊 Stories by Category</h2>
                <div class="category-bars" id="categoryBars">
                    <div class="chart-placeholder">Loading categories...</div>
                </div>
            </div>
        </div>

        <div class="panels-row">
            <div class="panel">
                <h2>📈 Activity Log</h2>
                <div class="activity-log" id="activityLog">
                    <div class="chart-placeholder">Loading activity...</div>
                </div>
            </div>

            <div class="panel">
                <h2>⏱️ Pipeline Status</h2>
                <div id="pipelineInfo">
                    <p><strong>Current Stage:</strong> <span id="currentStage">-</span></p>
                    <p><strong>Last Run:</strong> <span id="lastRun">-</span></p>
                    <p><strong>Next Run:</strong> <span id="nextRun">-</span></p>
                </div>
            </div>
        </div>

        <footer style="text-align: center; padding: 20px; color: #666; font-size: 0.8rem;">
            <span class="last-updated">Last updated: <span id="lastUpdated">-</span></span>
        </footer>
    </div>

    <script>
        let refreshInterval;

        async function refreshData() {
            try {
                // Fetch metrics
                const metricsRes = await fetch('/api/dashboard/metrics');
                const metrics = await metricsRes.json();

                // Fetch pipeline status
                const statusRes = await fetch('/api/dashboard/status');
                const status = await statusRes.json();

                // Fetch story queue
                const queueRes = await fetch('/api/dashboard/queue');
                const queue = await queueRes.json();

                updateMetrics(metrics);
                updateStatus(status);
                updateQueue(queue);
                updateCategories(metrics.stories_by_category || {});

                document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();
            } catch (error) {
                console.error('Failed to refresh data:', error);
            }
        }

        function updateMetrics(metrics) {
            document.getElementById('totalStories').textContent = metrics.total_stories || 0;
            document.getElementById('pendingStories').textContent = metrics.stories_by_status?.pending || 0;
            document.getElementById('publishedStories').textContent = metrics.stories_by_status?.published || 0;
            document.getElementById('approvalRate').textContent = (metrics.approval_rate * 100).toFixed(0) + '%';
            document.getElementById('avgQuality').textContent = metrics.avg_quality_score.toFixed(1);
            document.getElementById('storiesToday').textContent = metrics.stories_today || 0;
        }

        function updateStatus(status) {
            const dot = document.getElementById('statusDot');
            const text = document.getElementById('statusText');

            dot.className = 'status-dot ' + (status.status || 'idle');
            text.textContent = (status.status || 'idle').charAt(0).toUpperCase() + (status.status || 'idle').slice(1);

            document.getElementById('currentStage').textContent = status.current_stage || 'Idle';
            document.getElementById('lastRun').textContent = status.last_run || 'Never';
            document.getElementById('nextRun').textContent = status.next_run || 'Not scheduled';
        }

        function updateQueue(queue) {
            const container = document.getElementById('storyQueue');

            if (!queue || queue.length === 0) {
                container.innerHTML = '<div class="chart-placeholder">No pending stories</div>';
                return;
            }

            container.innerHTML = queue.map(story => `
                <div class="story-item">
                    <span class="title" title="${escapeHtml(story.title)}">${escapeHtml(story.title)}</span>
                    <span class="status status-${story.status}">${story.status}</span>
                </div>
            `).join('');
        }

        function updateCategories(categories) {
            const container = document.getElementById('categoryBars');

            if (!categories || Object.keys(categories).length === 0) {
                container.innerHTML = '<div class="chart-placeholder">No category data</div>';
                return;
            }

            const maxCount = Math.max(...Object.values(categories));

            container.innerHTML = Object.entries(categories)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 6)
                .map(([name, count]) => `
                    <div class="category-bar">
                        <span class="name">${escapeHtml(name)}</span>
                        <div class="bar">
                            <div class="bar-fill" style="width: ${(count / maxCount) * 100}%"></div>
                        </div>
                        <span class="count">${count}</span>
                    </div>
                `).join('');
        }

        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Auto-refresh every 30 seconds
        refreshInterval = setInterval(refreshData, 30000);

        // Initial load
        refreshData();
    </script>
</body>
</html>
"""


class DashboardServer:
    """Flask-based dashboard server for real-time pipeline monitoring."""

    def __init__(self, database: Database, port: int = 5001) -> None:
        """Initialize the dashboard server.

        Args:
            database: Database instance for story data
            port: Port number to run on (default: 5001)
        """
        self.db = database
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for API access

        self._setup_routes()

        # Pipeline status tracking
        self._pipeline_status = PipelineStatus()
        self._lock = threading.Lock()

    def _setup_routes(self) -> None:
        """Set up Flask routes."""

        @self.app.route("/")
        def index() -> str:
            return render_template_string(DASHBOARD_TEMPLATE)

        @self.app.route("/api/dashboard/metrics")
        def get_metrics() -> Any:
            metrics = self._calculate_metrics()
            return jsonify(asdict(metrics))

        @self.app.route("/api/dashboard/status")
        def get_status() -> Any:
            with self._lock:
                return jsonify(asdict(self._pipeline_status))

        @self.app.route("/api/dashboard/queue")
        def get_queue() -> Any:
            stories = self._get_story_queue()
            return jsonify(stories)

        @self.app.route("/api/dashboard/status", methods=["POST"])
        def update_status() -> Any:
            data = request.get_json()
            with self._lock:
                if "status" in data:
                    self._pipeline_status.status = data["status"]
                if "current_stage" in data:
                    self._pipeline_status.current_stage = data["current_stage"]
                if "last_run" in data:
                    self._pipeline_status.last_run = data["last_run"]
                if "next_run" in data:
                    self._pipeline_status.next_run = data["next_run"]
            return jsonify({"success": True})

    def _calculate_metrics(self) -> DashboardMetrics:
        """Calculate dashboard metrics from database."""
        try:
            # Get all stories
            all_stories = self.db.get_stories_needing_verification()
            published = self.db.get_published_stories()
            scheduled = self.db.get_scheduled_stories()

            # Combine all stories for calculations
            stories = all_stories + published + scheduled

            total = len(stories)
            approved_count = sum(
                1 for s in stories if s.verification_status == "approved"
            )
            published_count = len(published)

            # Calculate today's stories
            today = datetime.now().date()
            stories_today = sum(
                1 for s in stories if s.acquire_date and s.acquire_date.date() == today
            )

            # Average quality score
            quality_scores = [s.quality_score for s in stories if s.quality_score]
            avg_quality = (
                sum(quality_scores) / len(quality_scores) if quality_scores else 0
            )

            # Stories by category
            categories: dict[str, int] = {}
            for s in stories:
                cat = s.category or "Uncategorized"
                categories[cat] = categories.get(cat, 0) + 1

            # Stories by status
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
            )

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return DashboardMetrics()

    def _get_story_queue(self) -> list[dict[str, Any]]:
        """Get pending stories for the queue display."""
        try:
            stories = self.db.get_stories_needing_verification()[:20]  # Limit to 20

            return [
                {
                    "id": s.id,
                    "title": s.title or "Untitled",
                    "status": s.verification_status or "pending",
                    "quality_score": s.quality_score,
                    "category": s.category or "Uncategorized",
                }
                for s in stories
            ]
        except Exception as e:
            logger.error(f"Error getting story queue: {e}")
            return []

    def update_pipeline_status(
        self,
        status: Optional[str] = None,
        current_stage: Optional[str] = None,
        last_run: Optional[str] = None,
        next_run: Optional[str] = None,
    ) -> None:
        """Update the pipeline status from external code.

        Args:
            status: Pipeline status (idle, running, paused, error)
            current_stage: Current processing stage
            last_run: Last run timestamp
            next_run: Next scheduled run
        """
        with self._lock:
            if status is not None:
                self._pipeline_status.status = status
            if current_stage is not None:
                self._pipeline_status.current_stage = current_stage
            if last_run is not None:
                self._pipeline_status.last_run = last_run
            if next_run is not None:
                self._pipeline_status.next_run = next_run

    def run(self, debug: bool = False, threaded: bool = True) -> None:
        """Run the dashboard server.

        Args:
            debug: Enable Flask debug mode
            threaded: Enable threaded mode for concurrent requests
        """
        logger.info(f"Starting dashboard server on http://localhost:{self.port}")
        self.app.run(
            host="0.0.0.0",
            port=self.port,
            debug=debug,
            threaded=threaded,
            use_reloader=False,
        )

    def run_in_thread(self) -> threading.Thread:
        """Run the dashboard server in a background thread.

        Returns:
            The thread running the server
        """
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        logger.info(f"Dashboard server started in background on port {self.port}")
        return thread


# Singleton instance
_dashboard_server: Optional[DashboardServer] = None


def get_dashboard_server(database: Optional[Database] = None) -> DashboardServer:
    """Get or create the dashboard server instance.

    Args:
        database: Database instance (required on first call)

    Returns:
        DashboardServer instance
    """
    global _dashboard_server
    if _dashboard_server is None:
        if database is None:
            raise ValueError("Database required for first initialization")
        _dashboard_server = DashboardServer(database)
    return _dashboard_server


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for dashboard module."""
    from test_framework import TestSuite

    suite = TestSuite("Web Dashboard Tests")

    def test_pipeline_status_dataclass():
        """Test PipelineStatus defaults."""
        status = PipelineStatus()
        assert status.status == "idle"
        assert status.errors == []

    def test_dashboard_metrics_dataclass():
        """Test DashboardMetrics defaults."""
        metrics = DashboardMetrics()
        assert metrics.total_stories == 0
        assert metrics.stories_by_category == {}

    suite.add_test("PipelineStatus dataclass", test_pipeline_status_dataclass)
    suite.add_test("DashboardMetrics dataclass", test_dashboard_metrics_dataclass)

    return suite
