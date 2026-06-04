"""
generate_events.py
------------------
Converts the CCTV tracker outputs into a single events.jsonl file.

Input files consumed (relative to the project root):
    outputs/persons_cam1.json
    outputs/persons_cam2.json
    outputs/zone_analytics_cam1.json
    outputs/zone_analytics_cam2.json

Output written:
    outputs/events.jsonl

Each line of the output is a self-contained JSON object with **only** the
fields that are present in the official sample_events.jsonl schema:

    camera_id               str   – "CAM1" or "CAM2"
    person_id               int   – tracker-assigned ID
    first_seen              str   – "HH:MM:SS.mmm"  (time offset in video)
    last_seen               str   – "HH:MM:SS.mmm"
    visible_duration_seconds float – total seconds the track was visible
    frame_count             int   – number of frames the track was detected in
    valid_person            bool  – True if the track passes the quality filter
    staff_candidate         bool  – True if the track is likely a staff member
    zones                   dict  – {zone_name: dwell_seconds, ...}
    zone_visits             dict  – {zone_name: visit_count, ...}

Usage (from the project root, with .venv active):
    python src/cctv_tracker/generate_events.py

Or via the installed entry-point (if wired up in pyproject.toml):
    generate-events
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]  # .../yolo/
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

PERSONS_FILES = [
    OUTPUTS_DIR / "persons_cam1.json",
    OUTPUTS_DIR / "persons_cam2.json",
]

ZONE_FILES = [
    OUTPUTS_DIR / "zone_analytics_cam1.json",
    OUTPUTS_DIR / "zone_analytics_cam2.json",
]

OUTPUT_FILE = OUTPUTS_DIR / "events.jsonl"

# ---------------------------------------------------------------------------
# Schema field names – the exact keys present in the official sample schema.
# No additional fields will be written.
# ---------------------------------------------------------------------------

EVENT_FIELDS = [
    "camera_id",
    "person_id",
    "first_seen",
    "last_seen",
    "visible_duration_seconds",
    "frame_count",
    "valid_person",
    "staff_candidate",
    "zones",
    "zone_visits",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: pathlib.Path) -> list[dict[str, Any]]:
    """Load a JSON file that contains a list of objects."""
    if not path.exists():
        print(f"[WARN] File not found, skipping: {path}", file=sys.stderr)
        return []
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        print(f"[WARN] Expected a JSON array in {path}, got {type(data).__name__}", file=sys.stderr)
        return []
    return data


def build_zone_index(zone_files: list[pathlib.Path]) -> dict[tuple[str, int], dict[str, Any]]:
    """
    Build a lookup dict: (camera_id, person_id) → zone record.

    The zone record has exactly two keys from the official schema:
        zones        – {zone_name: dwell_seconds}
        zone_visits  – {zone_name: visit_count}
    """
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for path in zone_files:
        for record in load_json(path):
            key = (record["camera_id"], record["person_id"])
            index[key] = {
                "zones": record.get("zones", {}),
                "zone_visits": record.get("zone_visits", {}),
            }
    return index


def build_event(person: dict[str, Any], zone_index: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    """
    Merge a person record with its zone data to produce a single event object.
    Only fields listed in EVENT_FIELDS are included.
    """
    camera_id = person["camera_id"]
    person_id = person["person_id"]

    zone_data = zone_index.get((camera_id, person_id), {"zones": {}, "zone_visits": {}})

    event: dict[str, Any] = {
        "camera_id":               camera_id,
        "person_id":               person_id,
        "first_seen":              person["first_seen"],
        "last_seen":               person["last_seen"],
        "visible_duration_seconds": person["visible_duration_seconds"],
        "frame_count":             person["frame_count"],
        "valid_person":            person["valid_person"],
        "staff_candidate":         person["staff_candidate"],
        "zones":                   zone_data["zones"],
        "zone_visits":             zone_data["zone_visits"],
    }

    # Safety: strip any key that isn't in the official schema
    return {k: event[k] for k in EVENT_FIELDS if k in event}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_events() -> None:
    print("[INFO] Reading person records ...")
    persons: list[dict[str, Any]] = []
    for path in PERSONS_FILES:
        records = load_json(path)
        persons.extend(records)
        print(f"       {len(records):>4} records  <- {path.name}")

    if not persons:
        print("[ERROR] No person records found. Aborting.", file=sys.stderr)
        sys.exit(1)

    print("\n[INFO] Building zone lookup index ...")
    zone_index = build_zone_index(ZONE_FILES)
    print(f"       {len(zone_index):>4} zone entries indexed")

    print("\n[INFO] Merging records into events ...")
    events = [build_event(p, zone_index) for p in persons]
    print(f"       {len(events):>4} events built")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Writing -> {OUTPUT_FILE}")

    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    print(f"[OK]   {OUTPUT_FILE.name} written ({OUTPUT_FILE.stat().st_size:,} bytes, {len(events)} lines)")


if __name__ == "__main__":
    generate_events()
