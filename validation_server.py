"""Human validation web GUI for reviewing and approving stories."""

import logging
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Flask, render_template_string, request, jsonify, send_from_directory

from config import Config
from database import Database

logger = logging.getLogger(__name__)

# HTML Template with embedded CSS and JavaScript
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Story Validation - Social Media Publisher</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 25px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            margin-bottom: 20px;
        }

        header h1 {
            font-size: 1.5rem;
            color: #00d4ff;
        }

        .story-counter {
            color: #888;
            font-size: 0.9rem;
        }

        .top-buttons {
            display: flex;
            gap: 10px;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }

        .btn-accept {
            background: linear-gradient(135deg, #00c853, #00a845);
            color: white;
        }

        .btn-reject {
            background: linear-gradient(135deg, #ff5252, #d32f2f);
            color: white;
        }

        .btn-edit {
            background: linear-gradient(135deg, #ffc107, #ff9800);
            color: #1a1a2e;
        }

        .btn-close {
            background: linear-gradient(135deg, #607d8b, #455a64);
            color: white;
        }

        .btn-nav {
            background: rgba(255,255,255,0.1);
            color: #00d4ff;
            border: 1px solid #00d4ff;
            padding: 10px 20px;
        }

        .btn-nav:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }

        .btn-save {
            background: linear-gradient(135deg, #2196f3, #1976d2);
            color: white;
        }

        .btn-cancel {
            background: rgba(255,255,255,0.1);
            color: #e0e0e0;
            border: 1px solid #666;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 20px;
        }

        .preview-panel {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 25px;
        }

        .linkedin-preview {
            background: white;
            color: #000;
            border-radius: 8px;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .linkedin-header {
            display: flex;
            gap: 12px;
            margin-bottom: 15px;
        }

        .linkedin-avatar {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, #0077b5, #00a0dc);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 1.2rem;
        }

        .linkedin-author-info h3 {
            font-size: 0.95rem;
            color: #000;
            margin-bottom: 2px;
        }

        .linkedin-author-info p {
            font-size: 0.75rem;
            color: #666;
        }

        .linkedin-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #000;
            margin-bottom: 12px;
        }

        .linkedin-summary {
            font-size: 0.9rem;
            line-height: 1.5;
            color: #333;
            margin-bottom: 15px;
            white-space: pre-wrap;
        }

        .linkedin-image {
            width: 100%;
            border-radius: 8px;
            margin-bottom: 15px;
        }

        .linkedin-mentions {
            font-size: 0.85rem;
            color: #0077b5;
            margin-bottom: 10px;
        }

        .linkedin-hashtags {
            font-size: 0.85rem;
            color: #0077b5;
            margin-bottom: 10px;
        }

        .linkedin-promotion {
            font-size: 0.85rem;
            color: #666;
            font-style: italic;
            padding-top: 10px;
            border-top: 1px solid #eee;
            margin-top: 10px;
        }

        .linkedin-footer {
            display: flex;
            justify-content: space-around;
            padding-top: 15px;
            border-top: 1px solid #eee;
            color: #666;
            font-size: 0.85rem;
        }

        .linkedin-footer span {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .edit-panel {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 25px;
            display: none;
        }

        .edit-panel.visible {
            display: block;
        }

        .edit-panel h2 {
            font-size: 1.2rem;
            margin-bottom: 20px;
            color: #00d4ff;
        }

        .edit-group {
            margin-bottom: 20px;
        }

        .edit-group label {
            display: block;
            font-size: 0.85rem;
            color: #888;
            margin-bottom: 6px;
        }

        .edit-group input,
        .edit-group textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #444;
            border-radius: 6px;
            background: rgba(255,255,255,0.08);
            color: #e0e0e0;
            font-size: 0.9rem;
        }

        .edit-group textarea {
            min-height: 100px;
            resize: vertical;
        }

        .edit-group input:focus,
        .edit-group textarea:focus {
            outline: none;
            border-color: #00d4ff;
        }

        .edit-buttons {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }

        .navigation {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
            padding: 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
        }

        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .status-pending {
            background: #ffc107;
            color: #1a1a2e;
        }

        .status-approved {
            background: #00c853;
            color: white;
        }

        .status-rejected {
            background: #ff5252;
            color: white;
        }

        .status-scheduled {
            background: #2196f3;
            color: white;
        }

        .status-published {
            background: #9c27b0;
            color: white;
        }

        .meta-info {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 15px;
            font-size: 0.85rem;
            color: #888;
        }

        .meta-info span {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .toast {
            position: fixed;
            bottom: 30px;
            right: 30px;
            padding: 15px 25px;
            border-radius: 8px;
            font-weight: 500;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.3s ease;
            z-index: 1000;
        }

        .toast.show {
            opacity: 1;
            transform: translateY(0);
        }

        .toast.success {
            background: #00c853;
            color: white;
        }

        .toast.error {
            background: #ff5252;
            color: white;
        }

        .no-stories {
            text-align: center;
            padding: 60px;
            color: #888;
        }

        .no-stories h2 {
            margin-bottom: 15px;
            color: #00d4ff;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>📋 Story Validation</h1>
                <span class="story-counter" id="storyCounter">Loading...</span>
            </div>
            <div class="top-buttons" id="topButtons">
                <button class="btn btn-accept" onclick="acceptStory()">✓ Accept</button>
                <button class="btn btn-reject" onclick="rejectStory()">✗ Reject</button>
                <button class="btn btn-edit" onclick="toggleEdit()">✎ Edit</button>
                <button class="btn btn-close" onclick="closeValidator()">Close</button>
            </div>
        </header>

        <div class="main-content">
            <div class="preview-panel">
                <div class="meta-info" id="metaInfo"></div>
                <div class="linkedin-preview" id="linkedinPreview">
                    <div class="no-stories" id="noStories" style="display:none;">
                        <h2>No Stories to Review</h2>
                        <p>All stories have been processed or none are available.</p>
                    </div>
                </div>
            </div>

            <div class="edit-panel" id="editPanel">
                <h2>Edit Story</h2>

                <div class="edit-group">
                    <label>Scheduled Time</label>
                    <input type="datetime-local" id="editScheduledTime">
                </div>

                <div class="edit-group">
                    <label>Title</label>
                    <input type="text" id="editTitle">
                </div>

                <div class="edit-group">
                    <label>Summary</label>
                    <textarea id="editSummary" rows="6"></textarea>
                </div>

                <div class="edit-group">
                    <label>Hashtags (comma-separated)</label>
                    <input type="text" id="editHashtags">
                </div>

                <div class="edit-group">
                    <label>LinkedIn Mentions (comma-separated handles)</label>
                    <input type="text" id="editMentions">
                </div>

                <div class="edit-group">
                    <label>Promotion Message</label>
                    <textarea id="editPromotion" rows="3"></textarea>
                </div>

                <div class="edit-buttons">
                    <button class="btn btn-save" onclick="saveEdits()">Save Changes</button>
                    <button class="btn btn-cancel" onclick="cancelEdit()">Cancel</button>
                </div>
            </div>
        </div>

        <div class="navigation">
            <button class="btn btn-nav" id="prevBtn" onclick="navigate(-1)">← Previous</button>
            <span id="navInfo" style="color: #888; align-self: center;">- / -</span>
            <button class="btn btn-nav" id="nextBtn" onclick="navigate(1)">Next →</button>
        </div>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        let stories = [];
        let currentIndex = 0;
        let isEditMode = false;

        async function loadStories() {
            try {
                const response = await fetch('/api/stories');
                stories = await response.json();

                if (stories.length === 0) {
                    document.getElementById('noStories').style.display = 'block';
                    document.getElementById('topButtons').style.display = 'none';
                    document.getElementById('storyCounter').textContent = 'No stories available';
                    return;
                }

                // Find first unpublished story
                const unpublishedIndex = stories.findIndex(s =>
                    s.publish_status === 'unpublished' && s.verification_status === 'pending'
                );
                currentIndex = unpublishedIndex >= 0 ? unpublishedIndex : 0;

                renderStory();
                updateNavigation();
            } catch (error) {
                showToast('Failed to load stories: ' + error.message, 'error');
            }
        }

        function renderStory() {
            if (stories.length === 0) return;

            const story = stories[currentIndex];
            const preview = document.getElementById('linkedinPreview');

            // Status badge
            let statusClass = 'status-pending';
            let statusText = story.verification_status;
            if (story.publish_status === 'published') {
                statusClass = 'status-published';
                statusText = 'published';
            } else if (story.publish_status === 'scheduled') {
                statusClass = 'status-scheduled';
                statusText = 'scheduled';
            } else if (story.verification_status === 'approved') {
                statusClass = 'status-approved';
            } else if (story.verification_status === 'rejected') {
                statusClass = 'status-rejected';
            }

            // Meta info
            const metaInfo = document.getElementById('metaInfo');
            metaInfo.innerHTML = `
                <span><span class="status-badge ${statusClass}">${statusText}</span></span>
                <span>📅 ${story.acquire_date ? new Date(story.acquire_date).toLocaleDateString() : 'N/A'}</span>
                <span>⭐ Score: ${story.quality_score}/10</span>
                <span>🏷️ ${story.category || 'Uncategorized'}</span>
                ${story.scheduled_time ? `<span>🕐 Scheduled: ${new Date(story.scheduled_time).toLocaleString()}</span>` : ''}
            `;

            // Build mentions string
            const mentions = (story.linkedin_handles || [])
                .map(h => '@' + (h.handle || h.name))
                .join(' ');

            // Build hashtags string
            const hashtags = (story.hashtags || [])
                .map(t => t.startsWith('#') ? t : '#' + t)
                .join(' ');

            // Image HTML
            let imageHtml = '';
            if (story.image_path) {
                imageHtml = `<img class="linkedin-image" src="/image/${encodeURIComponent(story.image_path)}" alt="Story image">`;
            }

            preview.innerHTML = `
                <div class="linkedin-header">
                    <div class="linkedin-avatar">{{ author_initial }}</div>
                    <div class="linkedin-author-info">
                        <h3>{{ author_name }}</h3>
                        <p>Chemical Engineer • Just now</p>
                    </div>
                </div>

                <div class="linkedin-title">${escapeHtml(story.title)}</div>
                <div class="linkedin-summary">${escapeHtml(story.summary)}</div>

                ${imageHtml}

                ${mentions ? `<div class="linkedin-mentions">${escapeHtml(mentions)}</div>` : ''}
                ${hashtags ? `<div class="linkedin-hashtags">${escapeHtml(hashtags)}</div>` : ''}
                ${story.promotion ? `<div class="linkedin-promotion">${escapeHtml(story.promotion)}</div>` : ''}

                <div class="linkedin-footer">
                    <span>👍 Like</span>
                    <span>💬 Comment</span>
                    <span>🔄 Repost</span>
                    <span>📤 Send</span>
                </div>
            `;

            // Update counter
            document.getElementById('storyCounter').textContent =
                `Story ${currentIndex + 1} of ${stories.length} (ID: ${story.id})`;

            // Populate edit fields
            populateEditFields(story);
        }

        function populateEditFields(story) {
            document.getElementById('editTitle').value = story.title || '';
            document.getElementById('editSummary').value = story.summary || '';
            document.getElementById('editHashtags').value = (story.hashtags || []).join(', ');
            document.getElementById('editMentions').value = (story.linkedin_handles || [])
                .map(h => h.handle || h.name).join(', ');
            document.getElementById('editPromotion').value = story.promotion || '';

            if (story.scheduled_time) {
                const dt = new Date(story.scheduled_time);
                document.getElementById('editScheduledTime').value =
                    dt.toISOString().slice(0, 16);
            } else {
                document.getElementById('editScheduledTime').value = '';
            }
        }

        function updateNavigation() {
            document.getElementById('prevBtn').disabled = currentIndex <= 0;
            document.getElementById('nextBtn').disabled = currentIndex >= stories.length - 1;
            document.getElementById('navInfo').textContent =
                `${currentIndex + 1} / ${stories.length}`;
        }

        function navigate(direction) {
            const newIndex = currentIndex + direction;
            if (newIndex >= 0 && newIndex < stories.length) {
                currentIndex = newIndex;
                renderStory();
                updateNavigation();
                if (isEditMode) cancelEdit();
            }
        }

        function toggleEdit() {
            isEditMode = !isEditMode;
            document.getElementById('editPanel').classList.toggle('visible', isEditMode);
        }

        function cancelEdit() {
            isEditMode = false;
            document.getElementById('editPanel').classList.remove('visible');
            populateEditFields(stories[currentIndex]);
        }

        async function saveEdits() {
            const story = stories[currentIndex];

            const updates = {
                title: document.getElementById('editTitle').value,
                summary: document.getElementById('editSummary').value,
                hashtags: document.getElementById('editHashtags').value
                    .split(',').map(t => t.trim()).filter(t => t),
                linkedin_handles: document.getElementById('editMentions').value
                    .split(',').map(h => ({ handle: h.trim(), name: h.trim() })).filter(h => h.handle),
                promotion: document.getElementById('editPromotion').value,
                scheduled_time: document.getElementById('editScheduledTime').value || null
            };

            try {
                const response = await fetch(`/api/stories/${story.id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(updates)
                });

                if (response.ok) {
                    const updatedStory = await response.json();
                    stories[currentIndex] = updatedStory;
                    renderStory();
                    showToast('Changes saved successfully!', 'success');
                } else {
                    throw new Error('Failed to save changes');
                }
            } catch (error) {
                showToast('Error saving changes: ' + error.message, 'error');
            }
        }

        async function acceptStory() {
            const story = stories[currentIndex];

            try {
                const response = await fetch(`/api/stories/${story.id}/accept`, {
                    method: 'POST'
                });

                if (response.ok) {
                    const updatedStory = await response.json();
                    stories[currentIndex] = updatedStory;
                    renderStory();
                    showToast('Story approved for publication!', 'success');

                    // Auto-advance to next pending story
                    setTimeout(() => {
                        const nextPending = stories.findIndex((s, i) =>
                            i > currentIndex && s.verification_status === 'pending'
                        );
                        if (nextPending >= 0) {
                            currentIndex = nextPending;
                            renderStory();
                            updateNavigation();
                        }
                    }, 500);
                } else {
                    throw new Error('Failed to accept story');
                }
            } catch (error) {
                showToast('Error accepting story: ' + error.message, 'error');
            }
        }

        async function rejectStory() {
            const story = stories[currentIndex];

            if (!confirm('Are you sure you want to reject this story?')) {
                return;
            }

            try {
                const response = await fetch(`/api/stories/${story.id}/reject`, {
                    method: 'POST'
                });

                if (response.ok) {
                    const updatedStory = await response.json();
                    stories[currentIndex] = updatedStory;
                    renderStory();
                    showToast('Story rejected and archived.', 'success');

                    // Auto-advance to next pending story
                    setTimeout(() => {
                        const nextPending = stories.findIndex((s, i) =>
                            i > currentIndex && s.verification_status === 'pending'
                        );
                        if (nextPending >= 0) {
                            currentIndex = nextPending;
                            renderStory();
                            updateNavigation();
                        }
                    }, 500);
                } else {
                    throw new Error('Failed to reject story');
                }
            } catch (error) {
                showToast('Error rejecting story: ' + error.message, 'error');
            }
        }

        async function closeValidator() {
            try {
                await fetch('/api/shutdown', { method: 'POST' });
            } catch (e) {
                // Expected - server shuts down
            }
            window.close();
            // Fallback message if window.close() is blocked
            document.body.innerHTML = `
                <div style="display:flex;align-items:center;justify-content:center;height:100vh;flex-direction:column;">
                    <h1 style="color:#00d4ff;margin-bottom:20px;">✓ Validator Closed</h1>
                    <p style="color:#888;">You can close this browser tab and return to the terminal.</p>
                </div>
            `;
        }

        function showToast(message, type) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = `toast ${type} show`;
            setTimeout(() => toast.classList.remove('show'), 3000);
        }

        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Initialize
        loadStories();
    </script>
</body>
</html>
"""


class ValidationServer:
    """Flask-based validation server for human review of stories."""

    def __init__(self, database: Database, port: int = 5000):
        self.db = database
        self.port = port
        self.app = Flask(__name__)
        self.server_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Get author info from config
        self.author_name = Config.LINKEDIN_AUTHOR_NAME or "Author"
        self.author_initial = self.author_name[0].upper() if self.author_name else "A"

        self._setup_routes()

    def _setup_routes(self):
        """Set up Flask routes."""

        @self.app.route("/")
        def index():
            """Serve the main validation page."""
            html = HTML_TEMPLATE.replace("{{ author_name }}", self.author_name)
            html = html.replace("{{ author_initial }}", self.author_initial)
            return render_template_string(html)

        @self.app.route("/api/stories")
        def get_stories():
            """Get all stories for review, ordered by date."""
            try:
                with self.db._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT * FROM stories
                        ORDER BY
                            CASE WHEN verification_status = 'pending' THEN 0 ELSE 1 END,
                            acquire_date DESC
                    """)
                    rows = cursor.fetchall()

                stories = []
                for row in rows:
                    story = self._row_to_dict(row)
                    stories.append(story)

                return jsonify(stories)
            except Exception as e:
                logger.exception("Failed to get stories")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/stories/<int:story_id>", methods=["PUT"])
        def update_story(story_id: int):
            """Update story fields."""
            try:
                data = request.get_json()

                story = self.db.get_story(story_id)
                if not story:
                    return jsonify({"error": "Story not found"}), 404

                # Update fields
                if "title" in data:
                    story.title = data["title"]
                if "summary" in data:
                    story.summary = data["summary"]
                if "hashtags" in data:
                    story.hashtags = data["hashtags"]
                if "linkedin_handles" in data:
                    story.linkedin_handles = data["linkedin_handles"]
                if "promotion" in data:
                    story.promotion = data["promotion"]
                if "scheduled_time" in data:
                    if data["scheduled_time"]:
                        story.scheduled_time = datetime.fromisoformat(
                            data["scheduled_time"]
                        )
                    else:
                        story.scheduled_time = None

                self.db.update_story(story)

                # Return updated story
                return jsonify(story.to_dict())
            except Exception as e:
                logger.exception(f"Failed to update story {story_id}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/stories/<int:story_id>/accept", methods=["POST"])
        def accept_story(story_id: int):
            """Accept a story for publication."""
            try:
                story = self.db.get_story(story_id)
                if not story:
                    return jsonify({"error": "Story not found"}), 404

                story.verification_status = "approved"
                story.verification_reason = "Manually approved via human validation"
                self.db.update_story(story)

                return jsonify(story.to_dict())
            except Exception as e:
                logger.exception(f"Failed to accept story {story_id}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/stories/<int:story_id>/reject", methods=["POST"])
        def reject_story(story_id: int):
            """Reject a story."""
            try:
                story = self.db.get_story(story_id)
                if not story:
                    return jsonify({"error": "Story not found"}), 404

                story.verification_status = "rejected"
                story.verification_reason = "Manually rejected via human validation"
                self.db.update_story(story)

                return jsonify(story.to_dict())
            except Exception as e:
                logger.exception(f"Failed to reject story {story_id}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/image/<path:image_path>")
        def serve_image(image_path: str):
            """Serve story images."""
            try:
                image_file = Path(image_path)
                if image_file.exists():
                    return send_from_directory(
                        image_file.parent, image_file.name, mimetype="image/png"
                    )
                return "", 404
            except Exception as e:
                logger.warning(f"Failed to serve image {image_path}: {e}")
                return "", 404

        @self.app.route("/api/shutdown", methods=["POST"])
        def shutdown():
            """Shutdown the server."""
            self._shutdown_event.set()
            return jsonify({"status": "shutting down"})

    def _row_to_dict(self, row) -> dict:
        """Convert a database row to a dictionary."""
        import json

        def parse_json_field(value):
            if value is None:
                return []
            if isinstance(value, list):
                return value
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return []

        return {
            "id": row["id"],
            "title": row["title"],
            "summary": row["summary"],
            "source_links": parse_json_field(row.get("source_links")),
            "acquire_date": row.get("acquire_date"),
            "quality_score": row.get("quality_score", 0),
            "category": row.get("category", "Other"),
            "hashtags": parse_json_field(row.get("hashtags")),
            "image_path": row.get("image_path"),
            "verification_status": row.get("verification_status", "pending"),
            "verification_reason": row.get("verification_reason"),
            "scheduled_time": row.get("scheduled_time"),
            "publish_status": row.get("publish_status", "unpublished"),
            "linkedin_handles": parse_json_field(row.get("linkedin_handles")),
            "promotion": row.get("promotion"),
        }

    def start(self):
        """Start the validation server and open browser."""
        import werkzeug.serving

        print(f"\n🌐 Starting validation server at http://localhost:{self.port}")
        print("   Opening browser...")
        print("   Close the browser or click 'Close' to return to menu.\n")

        # Open browser
        webbrowser.open(f"http://localhost:{self.port}")

        # Run server in blocking mode but check for shutdown
        # Use werkzeug's make_server for cleaner shutdown
        server = werkzeug.serving.make_server(
            "localhost", self.port, self.app, threaded=True
        )

        def serve():
            while not self._shutdown_event.is_set():
                server.handle_request()

        # Run server
        try:
            while not self._shutdown_event.is_set():
                server.handle_request()
        except KeyboardInterrupt:
            pass

        print("   Validation server stopped.\n")


def run_human_validation(database: Database, port: int = 5000) -> None:
    """Run the human validation web interface."""
    server = ValidationServer(database, port)
    server.start()
