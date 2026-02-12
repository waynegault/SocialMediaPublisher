"""Human validation web GUI for reviewing and approving stories."""

import logging
import subprocess
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
    <meta name="description" content="Human Story Review Interface for Social Media Publisher">
    <title>Human Story Review - Social Media Publisher</title>
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect x='15' y='10' width='70' height='85' rx='8' fill='%231a1a2e' stroke='%2300d4ff' stroke-width='4'/><rect x='30' y='5' width='40' height='15' rx='4' fill='%2300d4ff'/><path d='M30 45 L45 60 L70 35' stroke='%2300c853' stroke-width='8' fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>">
    <style>
        /* Skip to main content link for keyboard users */
        .skip-link {
            position: absolute;
            top: -40px;
            left: 0;
            background: #00d4ff;
            color: #1a1a2e;
            padding: 8px 16px;
            z-index: 100;
            font-weight: 600;
            text-decoration: none;
            border-radius: 0 0 8px 0;
        }
        .skip-link:focus {
            top: 0;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            height: 100vh;
            max-height: 100vh;
            padding: 8px;
            overflow: hidden;
            transition: padding 0.3s ease;
        }

        body.edit-mode-active {
            padding: 6px;
        }

        .container {
            max-width: 100%;
            width: 100%;
            height: 100%;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        body.edit-mode-active .container {
            max-width: 100%;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 12px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            flex-shrink: 0;
        }

        header h1 {
            font-size: 1.2rem;
            color: #00d4ff;
        }

        .story-counter {
            color: #888;
            font-size: 0.85rem;
        }

        .top-buttons {
            display: flex;
            gap: 6px;
        }

        .btn {
            padding: 6px 14px;
            border: none;
            border-radius: 6px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }

        .btn:focus {
            outline: 3px solid #00d4ff;
            outline-offset: 2px;
        }

        .btn:focus:not(:focus-visible) {
            outline: none;
        }

        .btn:focus-visible {
            outline: 3px solid #00d4ff;
            outline-offset: 2px;
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

        /* Tooltip styles for button hover explanations */
        .btn[title] {
            position: relative;
        }

        .btn[title]:hover::after {
            content: attr(title);
            position: absolute;
            bottom: calc(100% + 8px);
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: #fff;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: normal;
            white-space: nowrap;
            max-width: 300px;
            white-space: normal;
            text-align: center;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            pointer-events: none;
        }

        .btn[title]:hover::before {
            content: '';
            position: absolute;
            bottom: calc(100% + 2px);
            left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent;
            border-top-color: #333;
            z-index: 1000;
            pointer-events: none;
        }

        .main-content {
            display: flex;
            flex-direction: column;
            gap: 6px;
            flex: 1;
            min-height: 0;
            overflow: hidden;
        }

        .story-details-section {
            background: linear-gradient(135deg, rgba(0,212,255,0.08) 0%, rgba(139,92,246,0.08) 100%);
            border-radius: 8px;
            padding: 8px 12px;
            width: 100%;
            border: 1px solid rgba(0,212,255,0.2);
            box-shadow: 0 2px 12px rgba(0,0,0,0.15);
            flex-shrink: 0;
        }

        .preview-section {
            display: flex;
            gap: 10px;
            flex: 1;
            min-height: 0;
            overflow: hidden;
        }

        .preview-panel {
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 10px;
            flex: 1;
            transition: flex 0.3s ease;
            overflow-y: auto;
            min-height: 0;
        }

        /* When edit mode is active, preview panel is 45% and edit panel is 55% */
        .preview-section.edit-mode .preview-panel {
            flex: 45;
        }

        .linkedin-preview {
            background: white;
            color: #000;
            border-radius: 8px;
            padding: 12px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 1rem;
            display: flex;
            flex-direction: column;
            min-height: 0;
            overflow-y: auto;
            border: 3px solid transparent;
            transition: border-color 0.3s ease;
            position: relative;
        }

        .linkedin-preview.story-accepted {
            border-color: #4caf50;
        }

        .linkedin-preview.story-rejected {
            border-color: #f44336;
        }

        .linkedin-preview.story-accepted::before,
        .linkedin-preview.story-rejected::before {
            position: absolute;
            top: 10px;
            right: 10px;
            font-size: 1.5rem;
            font-weight: bold;
            padding: 4px 12px;
            border-radius: 4px;
            z-index: 10;
        }

        .linkedin-preview.story-accepted::before {
            content: '✓ HUMAN ACCEPTED';
            background: #4caf50;
            color: white;
        }

        .linkedin-preview.story-rejected::before {
            content: '✗ HUMAN REJECTED';
            background: #f44336;
            color: white;
        }

        .linkedin-header {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            flex-shrink: 0;
        }

        .linkedin-avatar {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #0077b5, #00a0dc);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 1rem;
        }

        .linkedin-author-info h3 {
            font-size: 1rem;
            color: #000;
            margin-bottom: 2px;
        }

        .linkedin-author-info p {
            font-size: 0.9rem;
            color: #666;
        }

        .linkedin-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #000;
            margin-bottom: 8px;
            flex-shrink: 0;
        }

        .linkedin-summary {
            font-size: 1rem;
            line-height: 1.4;
            color: #333;
            margin-bottom: 10px;
            white-space: pre-wrap;
            flex-shrink: 0;
        }

        .linkedin-image {
            width: 100%;
            border-radius: 6px;
            margin-bottom: 10px;
            object-fit: contain;
            object-position: center top;
            flex: 0 1 auto;
            min-height: 80px;
            max-height: 25vh;
        }

        /* Show full image height in edit mode */
        .preview-section.edit-mode .linkedin-image {
            max-height: 50vh;
        }

        .linkedin-mentions {
            font-size: 1rem;
            line-height: 1.4;
            color: #0077b5;
            margin-bottom: 4px;
            flex-shrink: 0;
        }

        .linkedin-hashtags {
            font-size: 1rem;
            line-height: 1.4;
            color: #0077b5;
            margin-bottom: 4px;
            flex-shrink: 0;
        }

        .linkedin-promotion {
            font-size: 1rem;
            line-height: 1.4;
            color: #333;
            white-space: pre-wrap;
            margin-top: 10px;
            flex-shrink: 0;
        }

        .linkedin-sources {
            font-size: 1rem;
            line-height: 1.4;
            color: #333;
            margin-top: 10px;
            flex-shrink: 0;
        }

        .linkedin-sources a {
            color: #0077b5;
            text-decoration: none;
            word-break: break-all;
        }

        .linkedin-sources a:hover {
            text-decoration: underline;
        }

        .linkedin-spacer {
            height: 1.5em;
            flex-shrink: 0;
        }

        .linkedin-footer {
            display: flex;
            justify-content: space-around;
            padding-top: 8px;
            border-top: 1px solid #eee;
            color: #666;
            font-size: 1rem;
            flex-shrink: 0;
        }

        .linkedin-footer span {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .edit-panel {
            background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
            border-radius: 10px;
            padding: 16px;
            display: none;
            flex: 0;
            transition: flex 0.3s ease;
            border: 1px solid rgba(0,212,255,0.2);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            overflow-y: auto;
            min-height: 0;
        }

        .edit-panel.visible {
            display: flex;
            flex-direction: column;
            flex: 55;
        }

        .edit-panel h2 {
            font-size: 1.1rem;
            margin-bottom: 10px;
            color: #00d4ff;
        }

        .edit-group {
            margin-bottom: 10px;
        }

        .edit-row {
            display: flex;
            gap: 12px;
            margin-bottom: 10px;
        }

        .edit-row .edit-group {
            margin-bottom: 0;
        }

        .edit-row .edit-group.compact {
            flex: 0 0 auto;
        }

        .edit-row .edit-group.flex-grow {
            flex: 1;
        }

        .edit-group label {
            display: block;
            font-size: 0.85rem;
            color: #00d4ff;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 500;
        }

        .edit-group input[type="datetime-local"] {
            width: auto;
            min-width: 180px;
        }

        .edit-group input,
        .edit-group textarea {
            width: 100%;
            padding: 8px 10px;
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 6px;
            background: rgba(0,0,0,0.3);
            color: #e0e0e0;
            font-size: 1rem;
            line-height: 1.4;
            transition: all 0.2s ease;
            font-family: inherit;
        }

        .edit-group textarea {
            min-height: 38px;
            resize: none;
            overflow-y: auto;
            box-sizing: border-box;
        }

        .edit-group input:hover,
        .edit-group textarea:hover {
            border-color: rgba(0,212,255,0.4);
            background: rgba(0,0,0,0.4);
        }

        .edit-group input:focus,
        .edit-group textarea:focus {
            outline: none;
            border-color: #00d4ff;
            background: rgba(0,0,0,0.5);
            box-shadow: 0 0 0 3px rgba(0,212,255,0.15);
        }

        .edit-buttons {
            display: flex;
            gap: 8px;
            margin-top: auto;
            padding-top: 8px;
            flex-shrink: 0;
        }

        .navigation {
            display: flex;
            justify-content: center;
            gap: 12px;
            padding: 6px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            flex-shrink: 0;
        }

        .status-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        /* WCAG AA compliant contrast ratios */
        .status-pending {
            background: #e6a800;
            color: #1a1a2e;
        }

        .status-approved {
            background: #00a844;
            color: #ffffff;
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
            display: none;
        }

        .meta-info span {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .story-details-panel {
            background: transparent;
            border: none;
            padding: 0;
            margin-bottom: 0;
        }

        .story-details-panel h3 {
            display: none;
        }

        .details-grid {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .details-row {
            display: grid;
            grid-template-columns: 280px 1fr;
            gap: 16px;
            align-items: center;
            min-height: 36px;
        }

        .detail-item.fixed-width {
            width: 280px;
            flex-shrink: 0;
            display: flex;
            align-items: center;
        }

        .detail-item {
            display: inline-flex;
            flex-direction: row;
            align-items: baseline;
            gap: 8px;
            background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid rgba(255,255,255,0.08);
            transition: all 0.2s ease;
        }

        .detail-item:hover {
            background: linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.06));
            border-color: rgba(0,212,255,0.3);
        }

        .detail-item.full-width {
            flex-basis: 100%;
        }

        .detail-item.flex-grow {
            flex: 1;
        }

        .detail-label {
            font-size: 0.7rem;
            color: #00d4ff;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
            white-space: nowrap;
        }

        .detail-label::after {
            content: ':';
        }

        .detail-value {
            font-size: 0.85rem;
            color: #e0e0e0;
            line-height: 1.3;
        }

        .detail-value.score {
            font-size: 1rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00d4ff, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .detail-value.justification,
        .detail-value.reason {
            font-size: 0.8rem;
            color: #bbb;
            line-height: 1.4;
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

        /* ========================================
           Publish Button Styles
           ======================================== */
        .btn-publish {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
        }

        .btn-publish:hover {
            background: linear-gradient(135deg, #059669, #047857);
            transform: translateY(-1px);
        }

        .btn-publish:disabled {
            background: linear-gradient(135deg, #6b7280, #4b5563);
            cursor: not-allowed;
            opacity: 0.6;
        }

        /* Keyboard shortcuts help */
        .shortcuts-help {
            position: fixed;
            bottom: 80px;
            right: 30px;
            background: rgba(26,26,46,0.95);
            border: 1px solid rgba(0,212,255,0.3);
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 0.8rem;
            color: #888;
            display: none;
            z-index: 50;
        }

        .shortcuts-help.visible {
            display: block;
        }

        .shortcuts-help h4 {
            color: #00d4ff;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }

        .shortcuts-help kbd {
            background: rgba(255,255,255,0.1);
            padding: 2px 6px;
            border-radius: 4px;
            margin-right: 4px;
            font-family: monospace;
            color: #e0e0e0;
        }

        .shortcuts-help ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .shortcuts-help li {
            margin-bottom: 4px;
        }

        /* ========================================
           TASK 8.1: Mobile-Responsive Design
           ======================================== */
        @media (max-width: 768px) {
            body {
                padding: 4px;
            }

            header {
                flex-direction: column;
                gap: 8px;
                padding: 8px;
            }

            header h1 {
                font-size: 1rem;
            }

            .top-buttons {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 4px;
            }

            .btn {
                padding: 8px 12px;
                font-size: 0.85rem;
            }

            .preview-section {
                flex-direction: column;
            }

            .preview-section.edit-mode .preview-panel {
                flex: 1;
                max-height: 40vh;
            }

            .edit-panel.visible {
                flex: 1;
                min-height: 50vh;
            }

            .details-row {
                flex-direction: column;
                gap: 6px;
            }

            .detail-item.fixed-width {
                width: 100%;
            }

            .navigation {
                flex-wrap: wrap;
                padding: 8px;
            }

            .btn-nav {
                padding: 8px 16px;
                flex: 1;
                min-width: 100px;
            }

            .linkedin-preview {
                font-size: 0.95rem;
            }

            .linkedin-image {
                max-height: 20vh;
            }

            .toast {
                bottom: 10px;
                right: 10px;
                left: 10px;
                text-align: center;
            }

            .shortcuts-help {
                display: none !important;
            }

            /* Swipe hint for mobile */
            .swipe-hint {
                display: block;
                text-align: center;
                color: #666;
                font-size: 0.75rem;
                padding: 4px;
            }
        }

        @media (max-width: 480px) {
            header h1 {
                font-size: 0.9rem;
            }

            .story-counter {
                font-size: 0.75rem;
            }

            .btn {
                padding: 6px 10px;
                font-size: 0.8rem;
            }

            .story-details-section {
                padding: 6px 8px;
            }

            .detail-label {
                font-size: 0.65rem;
            }

            .detail-value {
                font-size: 0.8rem;
            }

            .edit-group label {
                font-size: 0.75rem;
            }

            .edit-group input,
            .edit-group textarea {
                font-size: 0.9rem;
                padding: 6px 8px;
            }
        }

        /* Touch-friendly targets for mobile */
        @media (hover: none) and (pointer: coarse) {
            .btn {
                min-height: 44px;
                min-width: 44px;
            }

            .btn-nav {
                min-height: 48px;
            }

            .story-checkbox {
                width: 32px;
                height: 32px;
            }
        }
    </style>
</head>
<body>
    <a href="#main-content" class="skip-link">Skip to main content</a>
    <div class="container" role="application" aria-label="Human Story Review Application">
        <header role="banner">
            <div>
                <h1>📋 Human Story Review</h1>
                <span class="story-counter" id="storyCounter" aria-live="polite">Loading...</span>
            </div>
            <div class="top-buttons" id="topButtons" role="toolbar" aria-label="Story actions">
                <button class="btn btn-accept" onclick="acceptStory()" aria-label="Accept this story for publication (A)" title="Mark this story as APPROVED for publication. Use after reviewing the content, image, and promotion message. Keyboard: A">✓ Accept</button>
                <button class="btn btn-reject" onclick="rejectStory()" aria-label="Reject this story (R)" title="Mark this story as REJECTED. It will not be published. Use for low-quality or irrelevant content. Keyboard: R">✗ Reject</button>
                <button class="btn btn-edit" onclick="toggleEdit()" aria-label="Edit story details (E)" aria-expanded="false" id="editToggleBtn" title="Open the edit panel to modify the title, summary, hashtags, or promotion message. Keyboard: E">✎ Edit</button>
                <button class="btn btn-publish" onclick="publishStory()" aria-label="Publish this story to LinkedIn (P)" id="publishBtn" title="Publish this approved story to LinkedIn immediately. Story must be approved first. Keyboard: P">🚀 Publish</button>
                <button class="btn btn-close" onclick="closeValidator()" aria-label="Close validator and return to menu" title="Close this review interface and return to the main menu. All changes are saved automatically.">Close</button>
            </div>
        </header>



        <main id="main-content" class="main-content" role="main">
            <!-- Story Details Section (above preview) -->
            <div class="story-details-section" id="storyDetailsSection">
                <div class="meta-info" id="metaInfo"></div>
                <div id="storyDetailsPanel"></div>
            </div>

            <!-- Preview Section (LinkedIn preview + Edit panel) -->
            <div class="preview-section">
                <div class="preview-panel">
                    <div class="linkedin-preview" id="linkedinPreview">
                        <div class="no-stories" id="noStories" style="display:none;">
                            <h2>No Stories to Review</h2>
                            <p>All stories have been processed or none are available.</p>
                        </div>
                    </div>
                </div>

                <div class="edit-panel" id="editPanel" role="region" aria-label="Edit story form">
                <h2>Edit Story</h2>

                <div class="edit-row">
                    <div class="edit-group compact">
                        <label for="editScheduledTime">Scheduled</label>
                        <input type="datetime-local" id="editScheduledTime" aria-describedby="scheduleHelp">
                    </div>

                    <div class="edit-group flex-grow">
                        <label for="editTitle">Title</label>
                        <input type="text" id="editTitle" aria-required="true">
                    </div>
                </div>

                <div class="edit-group">
                    <label for="editSummary">Summary</label>
                    <textarea id="editSummary" aria-required="true"></textarea>
                </div>

                <div class="edit-group">
                    <label for="editHashtags">Hashtags (comma-separated)</label>
                    <input type="text" id="editHashtags" placeholder="e.g., #Engineering, #Innovation">
                </div>

                <div class="edit-group-row" style="display: flex; gap: 10px; align-items: stretch;">
                    <div class="edit-group" style="flex: 1; display: flex; flex-direction: column;">
                        <label for="editDirectPeople">Direct People @mentions</label>
                        <textarea id="editDirectPeople" readonly disabled placeholder="People mentioned in story" style="flex: 1; min-height: 60px; resize: none;"></textarea>
                    </div>
                    <div class="edit-group" style="flex: 1; display: flex; flex-direction: column;">
                        <label for="editIndirectPeople">Indirect People @mentions</label>
                        <textarea id="editIndirectPeople" readonly disabled placeholder="Org leadership or key contacts" style="flex: 1; min-height: 60px; resize: none;"></textarea>
                    </div>
                </div>

                <div class="edit-group">
                    <label for="editPromotion">Promotion Message</label>
                    <textarea id="editPromotion"></textarea>
                </div>

                <div class="edit-buttons">
                    <button class="btn btn-save" onclick="saveEdits()" aria-label="Save all changes" title="Save all your edits to the database. The story will remain in 'pending' status until you Accept or Reject it.">Save Changes</button>
                    <button class="btn btn-cancel" onclick="cancelEdit()" aria-label="Cancel editing and discard changes" title="Discard all changes and close the edit panel without saving.">Cancel</button>
                </div>
            </div>
            </div>
        </main>

        <nav class="navigation" role="navigation" aria-label="Story navigation">
            <button class="btn btn-nav" id="prevBtn" onclick="navigate(-1)" aria-label="Go to previous story" title="Go to the previous story in the queue. Keyboard: Left Arrow">← Previous</button>
            <span id="navInfo" role="status" aria-live="polite" style="color: #b0b0b0; align-self: center;">- / -</span>
            <button class="btn btn-nav" id="nextBtn" onclick="navigate(1)" aria-label="Go to next story" title="Go to the next story in the queue. Keyboard: Right Arrow">Next →</button>
        </nav>
    </div>

    <div class="toast" id="toast" role="alert" aria-live="assertive"></div>

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

            // Apply visual feedback class based on HUMAN approval status
            // Use actual human_approved field, not heuristic check
            preview.classList.remove('story-accepted', 'story-rejected');
            if (story.human_approved === true) {
                preview.classList.add('story-accepted');
            } else if (story.human_approved === false && story.human_approved_at) {
                // human_approved is false AND human_approved_at is set means explicitly rejected
                preview.classList.add('story-rejected');
            }

            // Status badge - show "pending human approval" unless human has made decision
            let statusClass = 'status-pending';
            let statusText = 'pending human approval';
            if (story.publish_status === 'published') {
                statusClass = 'status-published';
                statusText = 'published';
            } else if (story.publish_status === 'scheduled') {
                statusClass = 'status-scheduled';
                statusText = 'scheduled';
            } else if (story.human_approved === true) {
                statusClass = 'status-approved';
                statusText = 'human approved';
            } else if (story.human_approved === false && story.human_approved_at) {
                statusClass = 'status-rejected';
                statusText = 'human rejected';
            }

            // Build status displays - distinguish between AI and human verification
            const aiVerificationClass = story.verification_status === 'approved' ? 'status-approved' :
                story.verification_status === 'rejected' ? 'status-rejected' : 'status-pending';
            // Show AI recommendation with note if human has already reviewed
            const humanHasReviewed = story.human_approved_at != null;
            const aiVerificationText = humanHasReviewed ?
                (story.verification_status === 'approved' ? 'approved (AI rec.)' :
                 story.verification_status === 'rejected' ? 'rejected (AI rec.)' : 'pending') :
                (story.verification_status || 'pending');
            const publishBadgeClass = story.publish_status === 'published' ? 'status-published' :
                story.publish_status === 'scheduled' ? 'status-scheduled' : 'status-pending';

            // Meta info (compact top bar)
            const metaInfo = document.getElementById('metaInfo');
            metaInfo.innerHTML = `
                <span><span class="status-badge ${statusClass}">${statusText}</span></span>
                <span>📅 ${story.acquire_date ? new Date(story.acquire_date).toLocaleDateString() : 'N/A'}</span>
                <span>🏷️ ${story.category || 'Uncategorized'}</span>
                <span>ID: ${story.id}</span>
            `;

            // Story details panel (detailed info) - goes in separate section above
            const detailsPanel = document.getElementById('storyDetailsPanel');
            detailsPanel.innerHTML = `
                <div class="story-details-panel">
                    <h3>📊 Story Details</h3>
                    <div class="details-grid">
                        <div class="details-row">
                            <div class="detail-item fixed-width">
                                <span class="detail-label">Quality Score</span>
                                <span class="detail-value score">${story.quality_score}/10</span>
                            </div>
                            <div class="detail-item flex-grow">
                                <span class="detail-label">Quality Justification</span>
                                <span class="detail-value justification">${story.quality_justification ? escapeHtml(story.quality_justification) : 'No justification provided'}</span>
                            </div>
                        </div>
                        <div class="details-row">
                            <div class="detail-item fixed-width">
                                <span class="detail-label">AI Recommendation</span>
                                <span class="detail-value"><span class="status-badge ${aiVerificationClass}">${aiVerificationText}</span></span>
                            </div>
                            <div class="detail-item flex-grow">
                                ${story.verification_reason && !humanHasReviewed ? `
                                <span class="detail-label">AI Reason</span>
                                <span class="detail-value reason">${escapeHtml(story.verification_reason)}</span>
                                ` : ''}
                            </div>
                        </div>
                        <div class="details-row">
                            <div class="detail-item fixed-width">
                                <span class="detail-label">Publish Status</span>
                                <span class="detail-value"><span class="status-badge ${publishBadgeClass}">${story.publish_status || 'unpublished'}</span></span>
                            </div>
                            <div class="detail-item flex-grow">
                                <span class="detail-label">Scheduled Time</span>
                                <span class="detail-value">${story.scheduled_time ? new Date(story.scheduled_time).toLocaleString() : 'Not scheduled'}</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Build mentions string from direct_people (directly mentioned) with URNs
            const directPeopleMentions = (story.direct_people || [])
                .filter(p => p.linkedin_urn || p.linkedin_profile)
                .map(p => '@' + p.name)
                .join(' ');

            // Build mentions string from indirect_people (institution leaders) - show all for reference
            const indirectPeopleMentions = (story.indirect_people || [])
                .map(p => {
                    const hasLinkedIn = p.linkedin_urn || p.linkedin_profile;
                    return hasLinkedIn ? '@' + p.name : p.name;
                })
                .join(' ');

            // Build hashtags string
            const hashtags = (story.hashtags || [])
                .map(t => t.startsWith('#') ? t : '#' + t)
                .join(' ');

            // Image HTML - use the raw path and alt text
            let imageHtml = '';
            if (story.image_path) {
                // Replace backslashes with forward slashes for URL using String.fromCharCode to avoid escape issues
                const backslash = String.fromCharCode(92);
                const imagePath = story.image_path.split(backslash).join('/');
                // Use the generated alt text or fallback to story title
                const altText = story.image_alt_text || ('Illustration for: ' + story.title);
                // Escape quotes in alt text for HTML attribute
                const escapedAltText = altText.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
                imageHtml = '<img class="linkedin-image" src="/image/' + imagePath + '" alt="' + escapedAltText + '" onerror="this.style.display=' + "'none'" + '">';
            }

            // Build sources HTML
            let sourcesHtml = '';
            if (story.source_links && story.source_links.length > 0) {
                const sourceLinks = story.source_links
                    .map(url => `<a href="${escapeHtml(url)}" target="_blank">${escapeHtml(url)}</a>`)
                    .join('<br>');
                sourcesHtml = `<div class="linkedin-sources"><strong>Source:</strong><br>${sourceLinks}</div>`;
            }

            // Reordered: Author -> Image -> Title -> Summary -> Promotion -> Sources -> (spacing) -> Hashtags -> Mentions
            preview.innerHTML = `
                <div class="linkedin-header">
                    <div class="linkedin-avatar">{{ author_initial }}</div>
                    <div class="linkedin-author-info">
                        <h3>{{ author_name }}</h3>
                        <p>{{ discipline_title }} • Just now</p>
                    </div>
                </div>

                ${imageHtml}

                <div class="linkedin-title">${escapeHtml(story.title)}</div>
                <div class="linkedin-summary">${escapeHtml(story.summary)}</div>

                ${story.promotion ? `<div class="linkedin-promotion">${escapeHtml(story.promotion)}</div>` : ''}
                ${sourcesHtml}

                ${(hashtags || directPeopleMentions || indirectPeopleMentions) ? '<div class="linkedin-spacer"></div>' : ''}
                ${hashtags ? `<div class="linkedin-hashtags">${escapeHtml(hashtags)}</div>` : ''}
                ${directPeopleMentions ? `<div class="linkedin-mentions">${escapeHtml(directPeopleMentions)}</div>` : ''}
                ${(directPeopleMentions && indirectPeopleMentions) ? '<div class="linkedin-spacer"></div>' : ''}
                ${indirectPeopleMentions ? `<div class="linkedin-mentions linkedin-org-leaders">${escapeHtml(indirectPeopleMentions)}</div>` : ''}

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

            // Update publish button state
            updatePublishButton();

            // Populate edit fields
            populateEditFields(story);
        }

        function populateEditFields(story) {
            document.getElementById('editTitle').value = story.title || '';
            document.getElementById('editSummary').value = story.summary || '';
            document.getElementById('editHashtags').value = (story.hashtags || []).join(', ');
            // Populate direct_people mentions (people directly mentioned in story)
            const directPeopleMentions = (story.direct_people || [])
                .filter(p => p.linkedin_urn || p.linkedin_profile)
                .map(p => '@' + p.name);
            document.getElementById('editDirectPeople').value = directPeopleMentions.join(', ');
            // Populate indirect_people mentions (institution leaders) - show all, mark those without LinkedIn
            const indirectPeopleMentions = (story.indirect_people || [])
                .map(p => {
                    const hasLinkedIn = p.linkedin_urn || p.linkedin_profile;
                    return hasLinkedIn ? '@' + p.name : p.name + ' (no LinkedIn)';
                });
            document.getElementById('editIndirectPeople').value = indirectPeopleMentions.join(', ');
            document.getElementById('editPromotion').value = story.promotion || '';

            if (story.scheduled_time) {
                const dt = new Date(story.scheduled_time);
                document.getElementById('editScheduledTime').value =
                    dt.toISOString().slice(0, 16);
            } else {
                document.getElementById('editScheduledTime').value = '';
            }

            // Auto-resize all textareas after populating
            setTimeout(autoResizeAllTextareas, 10);
        }

        function autoResizeTextarea(el) {
            // Force reflow by temporarily setting height to 0
            el.style.height = '0px';
            el.style.overflowY = 'hidden';
            // Get scrollHeight which is the full content height
            const scrollHeight = el.scrollHeight;
            // Set height with min 60px, max 400px
            const newHeight = Math.min(Math.max(scrollHeight, 60), 400);
            el.style.height = newHeight + 'px';
            // Show scrollbar only if content exceeds max
            el.style.overflowY = scrollHeight > 400 ? 'scroll' : 'hidden';
        }

        function autoResizeAllTextareas() {
            const textareas = document.querySelectorAll('.edit-group textarea');
            textareas.forEach(el => {
                autoResizeTextarea(el);
            });
        }

        // Use ResizeObserver to resize textareas when edit panel becomes visible
        const editPanel = document.getElementById('editPanel');
        if (editPanel) {
            const resizeObserver = new ResizeObserver(() => {
                if (editPanel.classList.contains('visible')) {
                    autoResizeAllTextareas();
                }
            });
            resizeObserver.observe(editPanel);
        }

        // Add input event listeners for auto-resize while typing
        document.addEventListener('DOMContentLoaded', function() {
            document.querySelectorAll('.edit-group textarea').forEach(textarea => {
                textarea.addEventListener('input', function() {
                    autoResizeTextarea(this);
                });
            });
        });

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
            const editPanel = document.getElementById('editPanel');
            const editToggleBtn = document.getElementById('editToggleBtn');
            const previewSection = document.querySelector('.preview-section');

            editPanel.classList.toggle('visible', isEditMode);
            previewSection.classList.toggle('edit-mode', isEditMode);
            document.body.classList.toggle('edit-mode-active', isEditMode);
            editToggleBtn.setAttribute('aria-expanded', isEditMode);

            if (isEditMode) {
                // Focus first input when opening edit panel
                document.getElementById('editScheduledTime').focus();
                // Auto-resize textareas after panel is fully visible (multiple attempts)
                setTimeout(autoResizeAllTextareas, 50);
                setTimeout(autoResizeAllTextareas, 150);
            }
        }

        function cancelEdit() {
            isEditMode = false;
            const editPanel = document.getElementById('editPanel');
            const editToggleBtn = document.getElementById('editToggleBtn');
            const previewSection = document.querySelector('.preview-section');

            editPanel.classList.remove('visible');
            previewSection.classList.remove('edit-mode');
            document.body.classList.remove('edit-mode-active');
            editToggleBtn.setAttribute('aria-expanded', 'false');
            populateEditFields(stories[currentIndex]);
            editToggleBtn.focus(); // Return focus to edit button
        }

        async function saveEdits() {
            const story = stories[currentIndex];

            const updates = {
                title: document.getElementById('editTitle').value,
                summary: document.getElementById('editSummary').value,
                hashtags: document.getElementById('editHashtags').value
                    .split(',').map(t => t.trim()).filter(t => t),
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

        // Keyboard navigation support (WCAG 2.1.1)
        document.addEventListener('keydown', function(e) {
            // Only handle if not in edit mode or not in an input field
            const isInInput = ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName);

            if (!isEditMode && !isInInput) {
                switch(e.key) {
                    case 'ArrowLeft':
                        e.preventDefault();
                        navigate(-1);
                        break;
                    case 'ArrowRight':
                        e.preventDefault();
                        navigate(1);
                        break;
                    case 'a':
                    case 'A':
                        if (!e.ctrlKey && !e.metaKey) {
                            e.preventDefault();
                            acceptStory();
                        }
                        break;
                    case 'r':
                    case 'R':
                        if (!e.ctrlKey && !e.metaKey) {
                            e.preventDefault();
                            rejectStory();
                        }
                        break;
                    case 'e':
                    case 'E':
                        if (!e.ctrlKey && !e.metaKey) {
                            e.preventDefault();
                            toggleEdit();
                        }
                        break;
                    case 'p':
                    case 'P':
                        if (!e.ctrlKey && !e.metaKey) {
                            e.preventDefault();
                            publishStory();
                        }
                        break;
                    case '?':  // Show keyboard shortcuts help
                        e.preventDefault();
                        toggleShortcutsHelp();
                        break;
                    case 'Escape':
                        if (isEditMode) {
                            e.preventDefault();
                            cancelEdit();
                        }
                        break;
                }
            } else if (isEditMode && e.key === 'Escape') {
                e.preventDefault();
                cancelEdit();
            }
        });

        // ========================================
        // Publish Story Function
        // ========================================
        async function publishStory() {
            if (stories.length === 0) return;

            const story = stories[currentIndex];

            // Check if story is human-approved (not just AI-approved)
            if (!story.human_approved) {
                showToast('Story must be human-approved before publishing', 'error');
                return;
            }

            // Check if already published
            if (story.publish_status === 'published') {
                showToast('Story is already published', 'error');
                return;
            }

            const publishBtn = document.getElementById('publishBtn');
            publishBtn.disabled = true;
            publishBtn.textContent = '⏳ Validating...';

            try {
                // First, run pre-publish validation
                const validateResponse = await fetch(`/api/stories/${story.id}/validate-publish`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });

                // Check content type before parsing
                const valContentType = validateResponse.headers.get('content-type') || '';
                let validation;
                if (valContentType.includes('application/json')) {
                    validation = await validateResponse.json();
                } else {
                    const text = await validateResponse.text();
                    console.error('Validation returned non-JSON:', text.substring(0, 500));
                    throw new Error('Server returned an unexpected response during validation.');
                }

                if (!validateResponse.ok) {
                    showToast(validation.error || 'Validation failed', 'error');
                    publishBtn.disabled = false;
                    publishBtn.textContent = '🚀 Publish';
                    return;
                }

                // Build validation summary for confirmation
                let validationSummary = '';

                if (validation.author_verified) {
                    validationSummary += `✅ Publishing as: ${validation.author_name}\\n`;
                } else if (validation.author_name) {
                    validationSummary += `⚠️ Author: ${validation.author_name} (unverified)\\n`;
                }

                // Count valid mentions
                const validMentions = (validation.mention_validations || []).filter(m => m.urn_valid).length;
                const totalMentions = (validation.mention_validations || []).length;

                if (totalMentions > 0) {
                    validationSummary += `📋 @mentions: ${validMentions}/${totalMentions} have valid URNs\\n`;
                }

                // Show warnings if any
                if (validation.warnings && validation.warnings.length > 0) {
                    validationSummary += '\\n⚠️ Warnings:\\n';
                    validation.warnings.slice(0, 3).forEach(w => {
                        validationSummary += `  • ${w}\\n`;
                    });
                    if (validation.warnings.length > 3) {
                        validationSummary += `  ... and ${validation.warnings.length - 3} more\\n`;
                    }
                }

                // Check for critical errors
                if (!validation.is_valid) {
                    let errorMsg = 'Pre-publish validation failed:\\n\\n';
                    validation.errors.forEach(e => {
                        errorMsg += `❌ ${e}\\n`;
                    });
                    alert(errorMsg);
                    publishBtn.disabled = false;
                    publishBtn.textContent = '🚀 Publish';
                    return;
                }

                // Determine if this is immediate or scheduled publish
                const hasSchedule = story.scheduled_time && new Date(story.scheduled_time) > new Date();
                const scheduleInfo = hasSchedule ?
                    `\\n📅 Scheduled for: ${new Date(story.scheduled_time).toLocaleString()}` : '';

                // Confirm publish with validation summary
                const actionWord = hasSchedule ? 'Schedule' : 'Publish';
                const confirmMsg = `${actionWord} this story to LinkedIn?\\n\\n` +
                    `"${story.title}"${scheduleInfo}\\n\\n` +
                    `Validation Summary:\\n${validationSummary}`;

                if (!confirm(confirmMsg)) {
                    publishBtn.disabled = false;
                    publishBtn.textContent = '🚀 Publish';
                    return;
                }

                publishBtn.textContent = hasSchedule ? '⏳ Scheduling...' : '⏳ Publishing...';

                const response = await fetch(`/api/stories/${story.id}/publish`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });

                // Check content type before parsing - handle HTML error pages
                const contentType = response.headers.get('content-type') || '';
                let data;
                if (contentType.includes('application/json')) {
                    data = await response.json();
                } else {
                    // Server returned non-JSON (likely HTML error page)
                    const text = await response.text();
                    console.error('Non-JSON response:', text.substring(0, 500));
                    if (text.includes('expired') || text.includes('invalid') || text.includes('<!doctype')) {
                        throw new Error('LinkedIn access token may be expired. Please refresh your token in Settings.');
                    }
                    throw new Error('Server returned an unexpected response. Check the terminal for details.');
                }

                if (response.ok) {
                if (data.scheduled) {
                    // Story was scheduled, not immediately published
                    story.publish_status = 'scheduled';
                    renderStory();
                    showToast(`Story scheduled for ${new Date(data.scheduled_time).toLocaleString()}!`, 'success');
                } else {
                    // Story was published immediately
                    story.publish_status = 'published';
                    story.linkedin_post_url = data.linkedin_post_url;
                    story.linkedin_post_id = data.linkedin_post_id;

                    renderStory();
                    showToast('Story published to LinkedIn!', 'success');

                    if (data.linkedin_post_url) {
                        // Offer to open the post
                        if (confirm('Story published! Open the post on LinkedIn?')) {
                            window.open(data.linkedin_post_url, '_blank');
                        }
                    }
                }
                } else {
                    showToast(data.error || 'Failed to publish story', 'error');
                }
            } catch (error) {
                console.error('Publish error:', error);
                showToast('Failed to publish: ' + error.message, 'error');
            } finally {
                publishBtn.disabled = false;
                publishBtn.textContent = '🚀 Publish';
            }
        }

        // Update publish button state based on current story
        function updatePublishButton() {
            const publishBtn = document.getElementById('publishBtn');
            if (!publishBtn || stories.length === 0) return;

            const story = stories[currentIndex];
            // Use human_approved for publish permission, not AI's verification_status
            const canPublish = story.human_approved === true &&
                             story.publish_status !== 'published';

            publishBtn.disabled = !canPublish;

            if (story.publish_status === 'published') {
                publishBtn.textContent = '✓ Published';
            } else if (!story.human_approved) {
                publishBtn.textContent = '🚀 Publish';
                publishBtn.title = 'Story must be human-approved before publishing';
            } else {
                publishBtn.textContent = '🚀 Publish';
                publishBtn.title = 'Publish this approved story to LinkedIn immediately';
            }
        }

        // Keyboard shortcuts help
        let shortcutsVisible = false;

        function toggleShortcutsHelp() {
            shortcutsVisible = !shortcutsVisible;
            let helpEl = document.getElementById('shortcutsHelp');

            if (!helpEl) {
                helpEl = document.createElement('div');
                helpEl.id = 'shortcutsHelp';
                helpEl.className = 'shortcuts-help';
                helpEl.innerHTML = `
                    <h4>⌨️ Keyboard Shortcuts</h4>
                    <ul>
                        <li><kbd>A</kbd> Accept story</li>
                        <li><kbd>R</kbd> Reject story</li>
                        <li><kbd>P</kbd> Publish story</li>
                        <li><kbd>E</kbd> Edit story</li>
                        <li><kbd>←</kbd> <kbd>→</kbd> Navigate</li>
                        <li><kbd>Esc</kbd> Cancel edit</li>
                        <li><kbd>?</kbd> Toggle this help</li>
                    </ul>
                `;
                document.body.appendChild(helpEl);
            }

            helpEl.classList.toggle('visible', shortcutsVisible);
        }

        // Initialize
        loadStories();
    </script>
</body>
</html>
"""


class ValidationServer:
    """Flask-based validation server for human review of stories."""

    def __init__(self, database: Database, port: int = 5000) -> None:
        """Initialize the validation server.

        Args:
            database: Database instance for story storage.
            port: Port number to run the server on (default: 5000).
        """
        self.db = database
        self.port = port
        self.app = Flask(__name__)
        self.server_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Get author info from config
        self.author_name = Config.LINKEDIN_AUTHOR_NAME or "Author"
        self.author_initial = self.author_name[0].upper() if self.author_name else "A"

        self._setup_routes()
        self._setup_error_handlers()

    def _setup_error_handlers(self):
        """Set up global error handlers to ensure JSON responses."""

        @self.app.errorhandler(Exception)
        def handle_exception(e):
            """Return JSON for any unhandled exception."""
            logger.exception(f"Unhandled exception: {e}")
            return jsonify({"error": str(e)}), 500

        @self.app.errorhandler(500)
        def handle_500(e):
            """Return JSON for 500 errors."""
            return jsonify({"error": "Internal server error"}), 500

        @self.app.errorhandler(404)
        def handle_404(e):
            """Return JSON for 404 errors on API routes."""
            if request.path.startswith("/api/"):
                return jsonify({"error": "Not found"}), 404
            return e  # Let Flask handle non-API 404s normally

    def _setup_routes(self):
        """Set up Flask routes."""

        @self.app.route("/")
        def index():
            """Serve the main validation page."""
            html = HTML_TEMPLATE.replace("{{ author_name }}", self.author_name)
            html = html.replace("{{ author_initial }}", self.author_initial)
            html = html.replace("{{ discipline_title }}", Config.DISCIPLINE.title())
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
            """Accept a story for publication (human approval).

            This sets human_approved=True but does NOT overwrite the AI's
            verification_status recommendation. The original AI decision
            is preserved for analytics and auditing.
            """
            try:
                story = self.db.get_story(story_id)
                if not story:
                    return jsonify({"error": "Story not found"}), 404

                # Only set human approval - don't overwrite AI recommendation
                story.human_approved = True
                story.human_approved_at = datetime.now()
                self.db.update_story(story)

                return jsonify(story.to_dict())
            except Exception as e:
                logger.exception(f"Failed to accept story {story_id}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/stories/<int:story_id>/reject", methods=["POST"])
        def reject_story(story_id: int):
            """Reject a story (human rejection).

            This sets human_approved=False but does NOT overwrite the AI's
            verification_status recommendation. The original AI decision
            is preserved for analytics and auditing.
            """
            try:
                story = self.db.get_story(story_id)
                if not story:
                    return jsonify({"error": "Story not found"}), 404

                # Mark as human rejected - don't overwrite AI recommendation
                story.human_approved = False
                story.human_approved_at = datetime.now()  # Track when rejected too
                self.db.update_story(story)

                return jsonify(story.to_dict())
            except Exception as e:
                logger.exception(f"Failed to reject story {story_id}")
                return jsonify({"error": str(e)}), 500

        @self.app.route(
            "/api/stories/<int:story_id>/validate-publish", methods=["POST"]
        )
        def validate_publish(story_id: int):
            """Validate a story before publishing (pre-flight check)."""
            logger.info(f"Validate-publish request for story {story_id}")
            try:
                story = self.db.get_story(story_id)
                if not story:
                    logger.warning(f"Story {story_id} not found")
                    return jsonify({"error": "Story not found"}), 404

                # Import and use linkedin_publisher
                from linkedin_publisher import LinkedInPublisher

                logger.info("Creating LinkedInPublisher...")
                publisher = LinkedInPublisher(self.db)
                logger.info("Running validate_before_publish...")
                validation = publisher.validate_before_publish(story)
                logger.info(f"Validation complete: is_valid={validation.is_valid}")

                return jsonify(validation.to_dict())

            except Exception as e:
                logger.exception(f"Failed to validate story {story_id}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/stories/<int:story_id>/publish", methods=["POST"])
        def publish_story(story_id: int):
            """Publish or schedule a story to LinkedIn.

            If the story has a scheduled_time in the future, it will be scheduled.
            Otherwise, it will be published immediately.
            """
            logger.info(f"Publish request received for story {story_id}")
            try:
                story = self.db.get_story(story_id)
                if not story:
                    logger.warning(f"Story {story_id} not found")
                    return jsonify({"error": "Story not found"}), 404

                # Verify story is human-approved (not just AI-approved)
                if not story.human_approved:
                    return jsonify(
                        {"error": "Story must be human-approved before publishing"}
                    ), 400

                # Verify story is not already published
                if story.publish_status == "published":
                    return jsonify({"error": "Story is already published"}), 400

                # Import and use linkedin_publisher
                from linkedin_publisher import LinkedInPublisher

                publisher = LinkedInPublisher(self.db)

                # Run pre-publish validation first
                validation = publisher.validate_before_publish(story)
                if not validation.is_valid:
                    error_msg = "; ".join(validation.errors)
                    return jsonify(
                        {
                            "error": f"Pre-publish validation failed: {error_msg}",
                            "validation": validation.to_dict(),
                        }
                    ), 400

                # Check if this should be scheduled or published immediately
                now = datetime.now()
                if story.scheduled_time and story.scheduled_time > now:
                    # Schedule for later - just mark as scheduled
                    story.publish_status = "scheduled"
                    self.db.update_story(story)

                    # Return scheduled response
                    result = story.to_dict()
                    result["success"] = True
                    result["scheduled"] = True
                    result["scheduled_time"] = story.scheduled_time.isoformat()
                    result["validation"] = validation.to_dict()
                    logger.info(
                        f"Story {story_id} scheduled for {story.scheduled_time}"
                    )
                    return jsonify(result)

                # Publish immediately (skip_validation=True since we just validated)
                post_id = publisher.publish_immediately(story, skip_validation=True)

                if post_id:
                    # Refresh story from database to get updated fields
                    story = self.db.get_story(story_id)
                    if story:
                        result = story.to_dict()
                        result["success"] = True
                        result["scheduled"] = False
                        result["validation"] = validation.to_dict()
                        return jsonify(result)
                    else:
                        return jsonify(
                            {
                                "success": True,
                                "scheduled": False,
                                "linkedin_post_id": post_id,
                                "validation": validation.to_dict(),
                            }
                        )
                else:
                    return jsonify({"error": "Failed to publish to LinkedIn"}), 500

            except Exception as e:
                logger.exception(f"Failed to publish story {story_id}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/favicon.ico")
        def favicon():
            """Return empty favicon to prevent 404 errors."""
            return "", 204

        @self.app.route("/image/<path:image_path>")
        def serve_image(image_path: str):
            """Serve story images."""
            try:
                # Normalize backslashes to forward slashes first
                normalized_path = image_path.replace("\\", "/")

                # Handle both relative and absolute paths
                image_file = Path(normalized_path)
                if not image_file.is_absolute():
                    # Relative path - resolve from workspace root
                    image_file = Path(__file__).parent / normalized_path

                logger.debug(
                    f"Serving image: requested='{image_path}', resolved='{image_file}', exists={image_file.exists()}"
                )

                if image_file.exists():
                    return send_from_directory(
                        str(image_file.parent), image_file.name, mimetype="image/png"
                    )
                else:
                    logger.warning(f"Image not found: {image_file}")
                    # Extract story ID from filename and clear image_path in database
                    import re

                    match = re.search(r"story_(\d+)_", image_file.name)
                    if match:
                        story_id = int(match.group(1))
                        story = self.db.get_story(story_id)
                        if story and story.image_path:
                            logger.info(
                                f"Clearing missing image_path for story {story_id}"
                            )
                            story.image_path = None
                            story.image_alt_text = None
                            self.db.update_story(story)
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

        def safe_get(row, key, default=None):
            """Safely get a value from sqlite3.Row, returning default if key doesn't exist."""
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

    def start(self):
        """Start the validation server and open browser."""
        import werkzeug.serving
        import socket

        # Check if port is available, if not try to find an available one
        original_port = self.port
        for attempt in range(5):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("localhost", self.port))
                sock.close()
                break
            except OSError:
                if attempt == 0:
                    print(f"   Port {self.port} is in use, trying another...")
                self.port += 1
        else:
            print("   Could not find an available port. Try again later.")
            return

        if self.port != original_port:
            print(f"   Using port {self.port} instead.")

        print(f"\n🌐 Starting validation server at http://localhost:{self.port}")
        print("   Opening browser...")
        print("   Close the browser or click 'Close' to return to menu.")
        print("   Press Ctrl+C in terminal to force stop.\n")

        # Open Edge browser in app mode
        url = f"http://localhost:{self.port}"
        try:
            # Try to open Edge in app mode (creates a standalone window)
            edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
            if not Path(edge_path).exists():
                edge_path = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
            subprocess.Popen(
                [edge_path, f"--app={url}", "--start-maximized"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.warning(f"Could not open Edge in app mode: {e}")
            # Fallback to default browser
            webbrowser.open(url)

        # Run server in blocking mode but check for shutdown
        # Use werkzeug's make_server for cleaner shutdown
        server = werkzeug.serving.make_server(
            "localhost", self.port, self.app, threaded=True
        )
        server.timeout = 1  # Check shutdown event every second

        # Run server until shutdown is triggered
        try:
            while not self._shutdown_event.is_set():
                server.handle_request()
        except KeyboardInterrupt:
            pass

        server.server_close()
        print("   Validation server stopped.\n")


def run_human_validation(database: Database, port: int = 5000) -> None:
    """Run the human validation web interface."""
    server = ValidationServer(database, port)
    server.start()


# ============================================================================
# Unit Tests
# ============================================================================
def _validation_server_tests() -> bool:
    """Run comprehensive tests for the validation server module."""
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite("Validation Server", "validation_server.py")
        suite.start_suite()

        def test_html_template_structure():
            assert "<!DOCTYPE html>" in HTML_TEMPLATE
            assert "Human Story Review" in HTML_TEMPLATE
            assert len(HTML_TEMPLATE) > 1000

        def test_html_has_action_buttons():
            assert "Accept" in HTML_TEMPLATE
            assert "Reject" in HTML_TEMPLATE
            assert "Edit" in HTML_TEMPLATE

        def test_html_has_api_endpoints():
            assert "/api/stories" in HTML_TEMPLATE
            assert "/api/shutdown" in HTML_TEMPLATE
            assert "acceptStory()" in HTML_TEMPLATE
            assert "rejectStory()" in HTML_TEMPLATE

        def test_validation_server_init():
            import tempfile
            tmp = tempfile.mkdtemp()
            db = Database(f"{tmp}/test.db")
            server = ValidationServer(db, port=0)
            assert server.db is db
            assert server.port == 0
            assert server.app is not None

        def test_validation_server_flask_routes():
            import tempfile
            tmp = tempfile.mkdtemp()
            db = Database(f"{tmp}/test.db")
            server = ValidationServer(db, port=0)
            rules = [rule.rule for rule in server.app.url_map.iter_rules()]
            assert "/" in rules
            assert "/api/stories" in rules or any("/api/" in r for r in rules)

        def test_validation_server_index_route():
            import tempfile
            tmp = tempfile.mkdtemp()
            db = Database(f"{tmp}/test.db")
            server = ValidationServer(db, port=0)
            with server.app.test_client() as client:
                resp = client.get("/")
                assert resp.status_code == 200
                html = resp.data.decode()
                assert "Human Story Review" in html

        def test_validation_server_stories_api_empty():
            import tempfile
            tmp = tempfile.mkdtemp()
            db = Database(f"{tmp}/test.db")
            server = ValidationServer(db, port=0)
            with server.app.test_client() as client:
                resp = client.get("/api/stories")
                assert resp.status_code == 200
                data = resp.get_json()
                assert isinstance(data, list)
                assert len(data) == 0

        def test_row_to_dict_basic():
            import tempfile
            tmp = tempfile.mkdtemp()
            db = Database(f"{tmp}/test.db")
            server = ValidationServer(db, port=0)
            # Use a dict-like object simulating sqlite3.Row
            mock_row = {
                "id": 1,
                "title": "Test Title",
                "summary": "Test Summary",
                "source_links": '["http://example.com"]',
                "acquire_date": "2026-01-01",
                "quality_score": 85,
                "quality_justification": "Good",
                "category": "Tech",
                "hashtags": '["#test"]',
                "image_path": None,
                "image_alt_text": None,
                "verification_status": "pending",
                "verification_reason": None,
                "scheduled_time": None,
                "publish_status": "unpublished",
                "direct_people": "[]",
                "indirect_people": "[]",
                "promotion": None,
            }
            result = server._row_to_dict(mock_row)
            assert result["id"] == 1
            assert result["title"] == "Test Title"
            assert result["quality_score"] == 85
            assert result["source_links"] == ["http://example.com"]
            assert result["hashtags"] == ["#test"]
            assert result["category"] == "Tech"

        def test_row_to_dict_missing_fields():
            import tempfile
            tmp = tempfile.mkdtemp()
            db = Database(f"{tmp}/test.db")
            server = ValidationServer(db, port=0)
            # Row with minimal fields and missing optional ones
            mock_row = {
                "id": 2,
                "title": "Minimal",
                "summary": "Min",
                "source_links": None,
                "acquire_date": None,
                "quality_score": None,
                "quality_justification": None,
                "category": None,
                "hashtags": None,
                "image_path": None,
                "image_alt_text": None,
                "verification_status": None,
                "verification_reason": None,
                "scheduled_time": None,
                "publish_status": None,
                "direct_people": None,
                "indirect_people": None,
                "promotion": None,
            }
            result = server._row_to_dict(mock_row)
            assert result["id"] == 2
            assert result["source_links"] == []
            assert result["quality_score"] == 0
            assert result["category"] == "Other"
            assert result["verification_status"] == "pending"

        suite.run_test(
            test_name="HTML template structure",
            test_func=test_html_template_structure,
            test_summary="Verify HTML_TEMPLATE has valid HTML structure",
            functions_tested="HTML_TEMPLATE constant",
            expected_outcome="Template contains DOCTYPE, title, and is substantial",
        )
        suite.run_test(
            test_name="HTML action buttons",
            test_func=test_html_has_action_buttons,
            test_summary="Verify template has Accept/Reject/Edit buttons",
            functions_tested="HTML_TEMPLATE constant",
            expected_outcome="All required action buttons present in template",
        )
        suite.run_test(
            test_name="HTML API endpoints",
            test_func=test_html_has_api_endpoints,
            test_summary="Verify template references required API endpoints",
            functions_tested="HTML_TEMPLATE constant",
            expected_outcome="Template contains /api/stories, /api/shutdown, and JS functions",
        )
        suite.run_test(
            test_name="ValidationServer initialization",
            test_func=test_validation_server_init,
            test_summary="Verify ValidationServer creates a Flask app with database",
            functions_tested="ValidationServer.__init__",
            expected_outcome="Server has db, port, and Flask app references",
        )
        suite.run_test(
            test_name="Flask routes registered",
            test_func=test_validation_server_flask_routes,
            test_summary="Verify ValidationServer registers required URL routes",
            functions_tested="ValidationServer._setup_routes",
            expected_outcome="Flask URL map contains / and /api/ routes",
        )
        suite.run_test(
            test_name="Index route returns 200",
            test_func=test_validation_server_index_route,
            test_summary="Verify GET / returns 200 with review page HTML",
            functions_tested="ValidationServer index route handler",
            expected_outcome="Response 200 with 'Human Story Review' in body",
        )
        suite.run_test(
            test_name="Stories API returns empty list",
            test_func=test_validation_server_stories_api_empty,
            test_summary="Verify GET /api/stories returns empty JSON list from empty DB",
            functions_tested="ValidationServer stories API route",
            expected_outcome="Response 200 with empty JSON list",
        )
        suite.run_test(
            test_name="_row_to_dict with full data",
            test_func=test_row_to_dict_basic,
            test_summary="Verify _row_to_dict parses a complete row into a dict",
            functions_tested="ValidationServer._row_to_dict",
            expected_outcome="Dict has correct values, JSON strings parsed to lists",
        )
        suite.run_test(
            test_name="_row_to_dict with missing fields",
            test_func=test_row_to_dict_missing_fields,
            test_summary="Verify _row_to_dict handles None fields with proper defaults",
            functions_tested="ValidationServer._row_to_dict",
            expected_outcome="Missing fields use defaults: score=0, category=Other, status=pending",
        )

        return suite.finish_suite()


def _create_module_tests() -> bool:
    return _validation_server_tests()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _validation_server_tests
)