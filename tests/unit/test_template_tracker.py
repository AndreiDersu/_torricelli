"""Unit tests for the Normalized Cross-Correlation TemplateTracker."""

import math

import cv2
import numpy as np

from light_tracker.core.models import Point2D
from light_tracker.tracking.template_tracker import TemplateTracker


def test_template_tracker_exact_match() -> None:
    # 200x200 image with a square feature at (100, 100)
    img = np.ones((200, 200), dtype=np.uint8) * 200
    cv2.circle(img, (100, 100), 8, 20, -1)

    tracker = TemplateTracker(template_size=(24, 24), search_size=(64, 64), min_score=0.7)
    target = Point2D(100.0, 100.0)

    tracker.initialize(img, target)
    assert tracker.is_initialized

    detected, score = tracker.track_frame(img)
    assert detected is not None
    assert score > 0.95
    assert math.isclose(detected.x, 100.0, abs_tol=0.2)
    assert math.isclose(detected.y, 100.0, abs_tol=0.2)


def test_template_tracker_shifted() -> None:
    # Frame 1: circle at (80, 80)
    frame1 = np.ones((200, 200), dtype=np.uint8) * 240
    cv2.circle(frame1, (80, 80), 8, 30, -1)

    # Frame 2: circle shifted by dx = +6, dy = -4 -> at (86, 76)
    frame2 = np.ones((200, 200), dtype=np.uint8) * 240
    cv2.circle(frame2, (86, 76), 8, 30, -1)

    tracker = TemplateTracker(template_size=(24, 24), search_size=(64, 64), min_score=0.7)
    tracker.initialize(frame1, Point2D(80.0, 80.0))

    detected, score = tracker.track_frame(frame2, prev_pos=Point2D(80.0, 80.0))
    assert detected is not None
    assert score > 0.90
    assert math.isclose(detected.x, 86.0, abs_tol=0.3)
    assert math.isclose(detected.y, 76.0, abs_tol=0.3)


def test_template_tracker_subpixel_accuracy() -> None:
    # Subpixel translation test
    frame1 = np.ones((160, 160), dtype=np.uint8) * 230
    cv2.circle(frame1, (80, 80), 7, 20, -1, cv2.LINE_AA)

    tracker = TemplateTracker(template_size=(24, 24), search_size=(50, 50), min_score=0.8)
    tracker.initialize(frame1, Point2D(80.0, 80.0))

    detected, _score = tracker.track_frame(frame1)
    assert detected is not None
    assert abs(detected.x - 80.0) < 0.1
    assert abs(detected.y - 80.0) < 0.1
