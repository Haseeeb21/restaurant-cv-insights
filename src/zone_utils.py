"""
Loads hand-labeled zones (configs/zones.json) and provides the geometry
logic that turns raw tracked detections into restaurant-operations metrics:
    - point-in-polygon membership (waiting areas, tables, exclusion zones)
    - line-crossing direction (entrance in/out)
    - per-track dwell-time accumulation

No models here — this is pure geometry on top of detector/tracker output.
"""
import json
import time
from collections import defaultdict

import numpy as np
from shapely.geometry import Polygon, box


def load_zones(path="configs/zones.json"):
    with open(path) as f:
        data = json.load(f)
    zones = data["zones"]
    for z in zones:
        if z["type"] == "entrance_line" and len(z["points"]) != 2:
            raise ValueError(
                f"Zone '{z['name']}' is type 'entrance_line' but has {len(z['points'])} points. "
                f"An entrance_line must have EXACTLY 2 points (start, end) — it's a line, not an "
                f"area. If you clicked 4 points thinking you were marking a box/area, re-label it "
                f"with just 2 clicks (the two ends of the doorway threshold)."
            )
    return zones


def point_in_polygon(point, polygon_points):
    """Ray-casting point-in-polygon test. polygon_points: list of (x, y)."""
    poly = np.array(polygon_points, dtype=np.float32)
    x, y = point
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def overlap_ratio(person_xyxy, polygon_points):
    """intersection_area / person_bbox_area — NOT classic symmetric IoU.
    Classic IoU (intersection / union) would be near-zero here regardless of
    true containment, since a zone polygon is usually far larger than one
    person's box and would dominate the union. This answers 'what fraction of
    THIS person is inside the zone', which is the question that matters for
    zone membership, and is far more tolerant of partial occlusion (e.g. a
    person standing behind a counter, where only their upper body is visible)
    than a single anchor-point test."""
    x1, y1, x2, y2 = person_xyxy
    person_box = box(x1, y1, x2, y2)
    zone_poly = Polygon(polygon_points)
    if not zone_poly.is_valid or person_box.area == 0:
        return 0.0
    inter = person_box.intersection(zone_poly).area
    return inter / person_box.area


def side_of_line(point, line_points):
    """Returns +1/-1 depending on which side of a 2-point line the point is on.
    Used for entrance in/out direction."""
    (x1, y1), (x2, y2) = line_points
    x, y = point
    d = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    return 1 if d > 0 else -1


class ZoneTracker:
    """
    Maintains per-track state across frames:
      - which zone(s) each track is currently in
      - dwell time accumulated per (track_id, zone_name)
      - entrance in/out counts based on line-side transitions

    Line-crossing is gated by `count_line` in update() so a caller can hold
    off counting for tracks that are too young to trust (see min_track_age
    in run_pipeline.py) — this filters out phantom crossings caused by
    ByteTrack re-spawning a new ID on the opposite side of the line after
    briefly losing a person (very common right at a doorway).
    """

    def __init__(self, zones):
        self.zones = zones
        self.line_zones = [z for z in zones if z["type"] == "entrance_line"]
        self.area_zones = [z for z in zones if z["type"] in ("waiting_area", "table", "exclusion")]

        self.track_last_side = {}         # track_id -> last side of each entrance line
        self.track_zone_entry_time = {}   # (track_id, zone_name) -> first-seen timestamp
        self.dwell_totals = defaultdict(float)  # (track_id, zone_name) -> accumulated seconds
        self.in_count = 0
        self.out_count = 0

    def update(self, track_id, centroid, now=None, count_line=True):
        """Call once per frame per tracked person centroid. Returns list of zone
        names the centroid currently falls in (excluding entrance_line zones).
        Set count_line=False for tracks too young to trust for crossing counts
        (area-zone membership/dwell is still processed either way)."""
        now = now or time.time()
        current_zones = []

        # Area zone membership + dwell accumulation
        for z in self.area_zones:
            in_zone = point_in_polygon(centroid, z["points"])
            key = (track_id, z["name"])
            if in_zone:
                current_zones.append(z["name"])
                if key not in self.track_zone_entry_time:
                    self.track_zone_entry_time[key] = now
            else:
                if key in self.track_zone_entry_time:
                    self.dwell_totals[key] += now - self.track_zone_entry_time[key]
                    del self.track_zone_entry_time[key]

        # Entrance line crossing (in/out) — only recorded/counted if count_line
        if count_line:
            for z in self.line_zones:
                side = side_of_line(centroid, z["points"])
                prev = self.track_last_side.get((track_id, z["name"]))
                if prev is not None and prev != side:
                    if side > 0:
                        self.in_count += 1
                    else:
                        self.out_count += 1
                self.track_last_side[(track_id, z["name"])] = side

        return current_zones

    def current_dwell(self, track_id, zone_name, now=None):
        """Live dwell time for an in-progress zone visit (for on-frame display)."""
        now = now or time.time()
        key = (track_id, zone_name)
        base = self.dwell_totals.get(key, 0.0)
        if key in self.track_zone_entry_time:
            base += now - self.track_zone_entry_time[key]
        return base
