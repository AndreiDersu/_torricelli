"""Tracking algorithms package."""

from __future__ import annotations

from light_tracker.tracking.base import TrackerProtocol
from light_tracker.tracking.color_tracker import ColorTracker
from light_tracker.tracking.manual_tracker import ManualTracker
from light_tracker.tracking.template_tracker import TemplateTracker

__all__ = ["ColorTracker", "ManualTracker", "TemplateTracker", "TrackerProtocol"]
