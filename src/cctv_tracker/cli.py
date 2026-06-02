from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


DEFAULT_INPUT = Path("./CCTV Footage/CAM 1.mp4")
DEFAULT_OUTPUT = Path("./outputs/annotated_cam1.mp4")
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_ALL_CAMERAS = [
    Path("./CCTV Footage/CAM 1.mp4"),
    Path("./CCTV Footage/CAM 2.mp4"),
]
PERSON_CLASS_ID = 0
MIN_VISIBLE_SECONDS = 15
MIN_FRAMES = 300
STRICT_MIN_VISIBLE_SECONDS = 30
STRICT_MIN_FRAMES = 600
CAM1_ZONES = {
    "skincare_wall": {
        "polygon": [(0, 120), (1320, 120), (1160, 760), (0, 900)],
        "color": (255, 180, 0),
    },
    "makeup_unit": {
        "polygon": [(760, 700), (1320, 650), (1490, 1040), (820, 1080), (700, 820)],
        "color": (180, 80, 255),
    },
    "mirror_area": {
        "polygon": [(1320, 470), (1780, 430), (1860, 900), (1500, 980), (1270, 720)],
        "color": (255, 120, 120),
    },
    "cash_counter": {
        "polygon": [(1600, 500), (1919, 485), (1919, 840), (1735, 900), (1560, 690)],
        "color": (0, 220, 255),
    },
    "front_display": {
        "polygon": [(520, 740), (1080, 760), (1170, 1080), (450, 1080)],
        "color": (80, 220, 120),
    },
}
CAM2_ZONES = {
    "skincare_wall": {
        "polygon": [(260, 180), (900, 140), (930, 560), (310, 670), (160, 470)],
        "color": (255, 180, 0),
    },
    "makeup_unit": {
        "polygon": [(820, 320), (1919, 260), (1919, 1080), (1160, 1080), (930, 700)],
        "color": (180, 80, 255),
    },
    "mirror_area": {
        "polygon": [(70, 320), (430, 300), (520, 760), (130, 820), (20, 520)],
        "color": (255, 120, 120),
    },
    "cash_counter": {
        "polygon": [(10, 620), (260, 590), (340, 1080), (20, 1080)],
        "color": (0, 220, 255),
    },
    "front_display": {
        "polygon": [(420, 640), (1080, 610), (1280, 1080), (470, 1080)],
        "color": (80, 220, 120),
    },
}
CAMERA_ZONES = {"CAM1": CAM1_ZONES, "CAM2": CAM2_ZONES}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track people in a CCTV video with YOLOv8 and ByteTrack."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input video path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output video path.")
    parser.add_argument(
        "--camera",
        choices=["CAM1", "CAM2", "ALL"],
        default=None,
        help="Select a camera by id, or ALL to process every configured camera.",
    )
    parser.add_argument(
        "--all-cameras",
        action="store_true",
        help="Process all configured camera videos.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="YOLOv8 model weights path or model name.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 'cpu', '0', or '0,1'.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Process every Nth frame. For example, 2 processes every second frame.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Generate an annotated output video.",
    )
    return parser


def ensure_input_video(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Input video not found: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")
    return path


def prepare_output_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def normalize_camera_name(path: Path) -> str:
    stem = path.stem.strip().lower()
    tokens = stem.replace("-", " ").replace("_", " ").split()
    compact = "".join(tokens)
    return compact or "camera"


def get_camera_id(path: Path) -> str:
    return path.stem.strip().replace(" ", "").replace("-", "").upper()


def get_camera_zones(camera_id: str) -> dict:
    zones = CAMERA_ZONES.get(camera_id)
    if zones is None:
        raise ValueError(f"No zone configuration defined for camera: {camera_id}")
    return zones


def build_input_paths(all_cameras: bool, input_path: Path, camera: str | None) -> list[Path]:
    if camera == "ALL" or all_cameras:
        paths = DEFAULT_ALL_CAMERAS
    elif camera == "CAM1":
        paths = [DEFAULT_ALL_CAMERAS[0]]
    elif camera == "CAM2":
        paths = [DEFAULT_ALL_CAMERAS[1]]
    else:
        paths = [input_path]
    return [ensure_input_video(path) for path in paths]


def build_output_paths(input_path: Path, output_path: Path | None) -> tuple[Path, Path, Path]:
    camera_name = normalize_camera_name(input_path)
    output_dir = Path("./outputs")

    if output_path is None:
        annotated_path = output_dir / f"annotated_{camera_name}.mp4"
    else:
        annotated_path = output_path
        output_dir = annotated_path.parent

    persons_path = output_dir / f"persons_{camera_name}.json"
    summary_path = output_dir / f"summary_{camera_name}.json"
    return annotated_path, persons_path, summary_path


def build_summary_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    camera_name = normalize_camera_name(input_path)
    raw_summary_path = output_dir / f"summary_raw_{camera_name}.json"
    filtered_summary_path = output_dir / f"summary_filtered_{camera_name}.json"
    return raw_summary_path, filtered_summary_path


def build_ranking_and_strict_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    camera_name = normalize_camera_name(input_path)
    ranking_path = output_dir / f"person_ranking_{camera_name}.json"
    strict_summary_path = output_dir / f"summary_strict_{camera_name}.json"
    return ranking_path, strict_summary_path


def build_zone_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    camera_name = normalize_camera_name(input_path)
    zone_analytics_path = output_dir / f"zone_analytics_{camera_name}.json"
    zone_summary_path = output_dir / f"zone_summary_{camera_name}.json"
    top_zones_path = output_dir / f"top_zones_{camera_name}.json"
    return zone_analytics_path, zone_summary_path, top_zones_path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def format_size_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.0f} MB"


def extract_first_frame(input_path: Path, output_path: Path) -> Path:
    capture = open_video(input_path)
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Failed to read first frame from: {input_path}")
    finally:
        capture.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Failed to write first frame image: {output_path}")
    return output_path


def open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return capture


def resolve_video_properties(capture: cv2.VideoCapture) -> tuple[float, int, int]:
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25.0
    if width <= 0 or height <= 0:
        raise RuntimeError("Failed to read input video dimensions.")

    return fps, width, height


def create_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {path}")
    return writer


def validate_skip_frames(skip_frames: int) -> int:
    if skip_frames < 1:
        raise ValueError("--skip-frames must be greater than or equal to 1.")
    return skip_frames


@dataclass
class PersonPresence:
    person_id: int
    first_seen_frame: int
    last_seen_frame: int
    visible_frame_count: int = 0

    def update(self, frame_number: int) -> None:
        if self.visible_frame_count == 0:
            self.first_seen_frame = frame_number
        self.last_seen_frame = frame_number
        self.visible_frame_count += 1


@dataclass
class PersonZonePresence:
    person_id: int
    zone_dwell_seconds: dict[str, float] = field(default_factory=dict)
    zone_visit_counts: dict[str, int] = field(default_factory=dict)
    current_zone: str | None = None
    current_zone_entry_frame: int | None = None
    last_seen_frame: int | None = None

    def add_zone_dwell(self, zone_name: str, dwell_seconds: float) -> None:
        self.zone_dwell_seconds[zone_name] = (
            self.zone_dwell_seconds.get(zone_name, 0.0) + dwell_seconds
        )

    def add_zone_visit(self, zone_name: str) -> None:
        self.zone_visit_counts[zone_name] = self.zone_visit_counts.get(zone_name, 0) + 1


def format_timestamp(frame_number: int, fps: float) -> str:
    total_seconds = max(frame_number - 1, 0) / fps if fps > 0 else 0.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def compute_visible_duration_seconds(person: PersonPresence, fps: float, skip_frames: int) -> float:
    if fps <= 0:
        return 0.0
    return (person.visible_frame_count * skip_frames) / fps


def compute_presence_span_seconds(person: PersonPresence, fps: float, skip_frames: int) -> float:
    if fps <= 0:
        return 0.0
    span_frames = max(person.last_seen_frame - person.first_seen_frame + skip_frames, skip_frames)
    return span_frames / fps


def is_staff_candidate(person: PersonPresence, fps: float, skip_frames: int, video_duration_seconds: float) -> bool:
    if video_duration_seconds <= 0:
        return False

    visible_duration_seconds = compute_visible_duration_seconds(person, fps, skip_frames)
    presence_span_seconds = compute_presence_span_seconds(person, fps, skip_frames)

    long_duration_threshold = max(60.0, video_duration_seconds * 0.30)
    broad_coverage_threshold = video_duration_seconds * 0.50

    return (
        visible_duration_seconds >= long_duration_threshold
        and presence_span_seconds >= broad_coverage_threshold
    )


def build_person_record(
    camera_id: str,
    person: PersonPresence,
    fps: float,
    skip_frames: int,
    video_duration_seconds: float,
) -> dict:
    visible_duration_seconds = round(
        compute_visible_duration_seconds(person, fps, skip_frames), 2
    )
    frame_count = person.visible_frame_count * skip_frames
    valid_person = (
        visible_duration_seconds >= MIN_VISIBLE_SECONDS
        and frame_count >= MIN_FRAMES
    )

    return {
        "camera_id": camera_id,
        "person_id": person.person_id,
        "first_seen": format_timestamp(person.first_seen_frame, fps),
        "last_seen": format_timestamp(person.last_seen_frame, fps),
        "visible_duration_seconds": visible_duration_seconds,
        "frame_count": frame_count,
        "valid_person": valid_person,
        "staff_candidate": is_staff_candidate(person, fps, skip_frames, video_duration_seconds),
    }


def is_strict_valid_person(person_record: dict) -> bool:
    return (
        person_record["visible_duration_seconds"] >= STRICT_MIN_VISIBLE_SECONDS
        and person_record["frame_count"] >= STRICT_MIN_FRAMES
    )


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def point_in_zone(point: tuple[int, int], polygon: list[tuple[int, int]]) -> bool:
    contour = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def resolve_zone_name(center_point: tuple[int, int], zones: dict) -> str | None:
    for zone_name, zone_config in zones.items():
        if point_in_zone(center_point, zone_config["polygon"]):
            return zone_name
    return None


def draw_zones(frame, zones: dict):
    annotated = frame.copy()
    for zone_name, zone_config in zones.items():
        polygon = zone_config["polygon"]
        color = zone_config["color"]
        contour = np.array(polygon, dtype=np.int32)
        cv2.polylines(annotated, [contour], isClosed=True, color=color, thickness=2)
        label_x, label_y = polygon[0]
        cv2.putText(
            annotated,
            zone_name,
            (label_x + 8, label_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def generate_zone_preview(input_path: Path, output_path: Path) -> Path:
    capture = open_video(input_path)
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Failed to read preview frame from: {input_path}")
    finally:
        capture.release()

    camera_id = get_camera_id(input_path)
    zones = get_camera_zones(camera_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview = draw_zones(frame, zones)
    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(f"Failed to write zone preview image: {output_path}")
    return output_path


def draw_summary_overlay(frame, customer_candidates: int, staff_candidates: int):
    overlay = frame.copy()
    lines = [
        f"Customers Detected: {customer_candidates}",
        f"Staff Candidates: {staff_candidates}",
    ]

    x = 20
    y = 35
    box_width = 320
    box_height = 70
    cv2.rectangle(overlay, (x - 10, y - 25), (x - 10 + box_width, y - 25 + box_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28

    return frame


def annotate_frame(
    frame,
    result,
    customer_candidates: int,
    staff_candidates: int,
    person_zone_labels: dict[int, str | None],
    zones: dict,
):
    boxes = result.boxes
    annotated = draw_zones(frame, zones)

    if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        classes = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        coordinates = boxes.xyxy.int().cpu().tolist()

        for track_id, class_id, confidence, (x1, y1, x2, y2) in zip(
            ids, classes, confidences, coordinates
        ):
            if class_id != PERSON_CLASS_ID:
                continue

            color = (0, 200, 0)
            label_id = track_id if track_id is not None else "NA"
            label = f"ID {label_id} | person {confidence:.2f}"
            zone_name = person_zone_labels.get(track_id)
            if zone_name:
                label = f"{label} | {zone_name}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            text_y = max(y1 - 10, text_height + 10)
            cv2.rectangle(
                annotated,
                (x1, text_y - text_height - baseline - 6),
                (x1 + text_width + 8, text_y + baseline - 6),
                color,
                thickness=-1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, text_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    return draw_summary_overlay(annotated, customer_candidates, staff_candidates)


def update_zone_presence(
    person_id: int,
    zone_name: str | None,
    zone_presence: dict[int, PersonZonePresence],
    frame_number: int,
    fps: float,
) -> None:
    analytics = zone_presence.get(person_id)
    if analytics is None:
        analytics = PersonZonePresence(person_id=person_id)
        zone_presence[person_id] = analytics

    previous_zone = analytics.current_zone
    if previous_zone != zone_name:
        if previous_zone is not None:
            print(f"Person {person_id} exited {previous_zone}")
            close_zone_interval(analytics, previous_zone, frame_number, fps)
        if zone_name is not None:
            print(f"Person {person_id} entered {zone_name}")
            analytics.add_zone_visit(zone_name)
            analytics.current_zone_entry_frame = frame_number
        analytics.current_zone = zone_name
        if zone_name is None:
            analytics.current_zone_entry_frame = None

    if previous_zone == zone_name and zone_name is not None and analytics.current_zone_entry_frame is None:
        analytics.current_zone_entry_frame = frame_number

    analytics.last_seen_frame = frame_number


def close_zone_interval(
    analytics: PersonZonePresence,
    zone_name: str,
    exit_frame: int,
    fps: float,
) -> None:
    entry_frame = analytics.current_zone_entry_frame
    if entry_frame is None or fps <= 0:
        return
    dwell_seconds = max(exit_frame - entry_frame, 0) / fps
    analytics.add_zone_dwell(zone_name, dwell_seconds)
    print(f"Person {analytics.person_id}: {zone_name} -> {dwell_seconds:.1f} sec")
    analytics.current_zone_entry_frame = None


def close_missing_tracks(
    zone_presence: dict[int, PersonZonePresence],
    active_person_ids: set[int],
    skip_frames: int,
    fps: float,
) -> None:
    for analytics in zone_presence.values():
        if (
            analytics.person_id not in active_person_ids
            and analytics.current_zone is not None
            and analytics.last_seen_frame is not None
        ):
            print(f"Person {analytics.person_id} exited {analytics.current_zone}")
            close_zone_interval(
                analytics,
                analytics.current_zone,
                analytics.last_seen_frame + skip_frames,
                fps,
            )
            analytics.current_zone = None


def flush_zone_exits(zone_presence: dict[int, PersonZonePresence], skip_frames: int, fps: float) -> None:
    for person_id, analytics in zone_presence.items():
        if analytics.current_zone is not None:
            print(f"Person {person_id} exited {analytics.current_zone}")
            exit_frame = (
                analytics.last_seen_frame + skip_frames
                if analytics.last_seen_frame is not None
                else 0
            )
            close_zone_interval(analytics, analytics.current_zone, exit_frame, fps)
            analytics.current_zone = None


def build_zone_analytics_records(
    camera_id: str, zone_presence: dict[int, PersonZonePresence], fps: float
) -> list[dict]:
    records = []
    for person_id, analytics in sorted(zone_presence.items()):
        zones = {
            zone_name: round(dwell_seconds, 2)
            for zone_name, dwell_seconds in analytics.zone_dwell_seconds.items()
            if dwell_seconds > 0
        }
        records.append(
            {
                "camera_id": camera_id,
                "person_id": person_id,
                "zones": zones,
                "zone_visits": dict(sorted(analytics.zone_visit_counts.items())),
            }
        )
    return records


def build_zone_summary(
    zone_records: list[dict],
    valid_person_ids: set[int],
    zones: dict,
) -> dict:
    summary = {
        zone_name: {"total_visits": 0, "total_dwell_seconds": 0.0}
        for zone_name in zones
    }

    for record in zone_records:
        if record["person_id"] not in valid_person_ids:
            continue
        for zone_name, dwell_seconds in record["zones"].items():
            visit_count = record.get("zone_visits", {}).get(zone_name, 0)
            if dwell_seconds <= 0 and visit_count <= 0:
                continue
            summary[zone_name]["total_visits"] += visit_count
            summary[zone_name]["total_dwell_seconds"] += dwell_seconds

    for zone_name in summary:
        summary[zone_name]["total_dwell_seconds"] = round(
            summary[zone_name]["total_dwell_seconds"], 2
        )

    return summary


def build_top_zones(zone_summary: dict) -> dict:
    ordered = sorted(
        zone_summary.items(),
        key=lambda item: (item[1]["total_visits"], item[1]["total_dwell_seconds"]),
    )
    if not ordered:
        return {"most_visited_zone": None, "least_visited_zone": None}
    return {
        "most_visited_zone": ordered[-1][0],
        "least_visited_zone": ordered[0][0],
    }


def sum_zone_dwell_seconds(zone_summary: dict) -> float:
    return round(
        sum(zone_data["total_dwell_seconds"] for zone_data in zone_summary.values()),
        2,
    )


def build_combined_summary(camera_results: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_zone_summary: dict[str, dict[str, float | int]] = {}
    camera_breakdown = {}

    total_people = 0
    total_staff_candidates = 0
    total_customer_candidates = 0
    total_dwell_seconds = 0.0

    for result in camera_results:
        camera_id = result["camera_id"]
        summary = result["summary"]
        zone_summary = result["zone_summary"]
        camera_breakdown[camera_id] = {
            "people": summary["unique_people"],
            "staff_candidates": summary["staff_candidates"],
            "customer_candidates": summary["customer_candidates"],
            "total_dwell_seconds": result["total_dwell_seconds"],
        }

        total_people += summary["unique_people"]
        total_staff_candidates += summary["staff_candidates"]
        total_customer_candidates += summary["customer_candidates"]
        total_dwell_seconds += result["total_dwell_seconds"]

        for zone_name, zone_data in zone_summary.items():
            aggregate = combined_zone_summary.setdefault(
                zone_name, {"total_visits": 0, "total_dwell_seconds": 0.0}
            )
            aggregate["total_visits"] += zone_data["total_visits"]
            aggregate["total_dwell_seconds"] += zone_data["total_dwell_seconds"]

    for zone_name in combined_zone_summary:
        combined_zone_summary[zone_name]["total_dwell_seconds"] = round(
            combined_zone_summary[zone_name]["total_dwell_seconds"], 2
        )

    payload = {
        "camera_count": len(camera_results),
        "total_people": total_people,
        "total_staff_candidates": total_staff_candidates,
        "total_customer_candidates": total_customer_candidates,
        "total_dwell_seconds": round(total_dwell_seconds, 2),
        "cam1_people": camera_breakdown.get("CAM1", {}).get("people", 0),
        "cam2_people": camera_breakdown.get("CAM2", {}).get("people", 0),
        "combined_people": total_people,
        "cam1_dwell_seconds": camera_breakdown.get("CAM1", {}).get("total_dwell_seconds", 0.0),
        "cam2_dwell_seconds": camera_breakdown.get("CAM2", {}).get("total_dwell_seconds", 0.0),
        "camera_breakdown": camera_breakdown,
        "zones": combined_zone_summary,
        "zone_totals": {
            zone_name: zone_data["total_dwell_seconds"]
            for zone_name, zone_data in combined_zone_summary.items()
        },
    }

    output_path = output_dir / "combined_summary.json"
    write_json(output_path, payload)
    return output_path


def process_video(
    input_path: Path,
    output_path: Path | None,
    model_name: str,
    confidence: float,
    image_size: int,
    device: str | None,
    skip_frames: int,
    save_video: bool,
) -> dict:
    from ultralytics import YOLO

    input_path = ensure_input_video(input_path)
    camera_id = get_camera_id(input_path)
    zones = get_camera_zones(camera_id)
    output_path, persons_output_path, summary_output_path = build_output_paths(
        input_path, output_path
    )
    if save_video:
        prepare_output_path(output_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_skip_frames(skip_frames)
    print(f"Video generation: {'ENABLED' if save_video else 'DISABLED'}")

    capture = open_video(input_path)
    fps, width, height = resolve_video_properties(capture)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    output_fps = fps / skip_frames
    if output_fps <= 0:
        output_fps = 1.0

    writer = None
    if save_video:
        print("Writing video:")
        print(display_path(output_path))
        writer = create_writer(output_path, output_fps, width, height)

    model = YOLO(model_name)
    processed_frames = 0
    started_at = time.perf_counter()
    people: dict[int, PersonPresence] = {}
    zone_presence: dict[int, PersonZonePresence] = {}
    video_duration_seconds = total_frames / fps if fps > 0 and total_frames > 0 else 0.0
    raw_summary_path, filtered_summary_path = build_summary_paths(input_path, output_path.parent)
    ranking_path, strict_summary_path = build_ranking_and_strict_paths(
        input_path, output_path.parent
    )
    zone_analytics_path, zone_summary_path, top_zones_path = build_zone_paths(
        input_path, output_path.parent
    )

    try:
        results = model.track(
            source=str(input_path),
            tracker="bytetrack.yaml",
            persist=True,
            stream=True,
            classes=[PERSON_CLASS_ID],
            conf=confidence,
            imgsz=image_size,
            device=device,
            vid_stride=skip_frames,
            verbose=False,
        )

        for result in results:
            processed_frames += 1
            source_frame_number = ((processed_frames - 1) * skip_frames) + 1
            tracked_ids: list[int] = []
            person_zone_labels: dict[int, str | None] = {}

            if result.boxes is not None and result.boxes.id is not None:
                tracked_ids = [int(track_id) for track_id in result.boxes.id.int().cpu().tolist()]

            for track_id in tracked_ids:
                person = people.get(track_id)
                if person is None:
                    person = PersonPresence(
                        person_id=track_id,
                        first_seen_frame=source_frame_number,
                        last_seen_frame=source_frame_number,
                    )
                    people[track_id] = person
                person.update(source_frame_number)

            if result.boxes is not None and result.boxes.xyxy is not None and result.boxes.id is not None:
                tracked_boxes = result.boxes.xyxy.int().cpu().tolist()
                for track_id, (x1, y1, x2, y2) in zip(tracked_ids, tracked_boxes):
                    center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                    zone_name = resolve_zone_name(center_point, zones)
                    person_zone_labels[track_id] = zone_name
                    update_zone_presence(
                        person_id=track_id,
                        zone_name=zone_name,
                        zone_presence=zone_presence,
                        frame_number=source_frame_number,
                        fps=fps,
                    )

            close_missing_tracks(zone_presence, set(tracked_ids), skip_frames, fps)

            provisional_video_seconds = source_frame_number / fps if fps > 0 else 0.0
            provisional_staff_count = sum(
                1
                for person in people.values()
                if is_staff_candidate(person, fps, skip_frames, provisional_video_seconds)
            )
            provisional_customer_count = max(len(people) - provisional_staff_count, 0)

            if save_video and writer is not None:
                frame = result.orig_img
                annotated_frame = annotate_frame(
                    frame,
                    result,
                    customer_candidates=provisional_customer_count,
                    staff_candidates=provisional_staff_count,
                    person_zone_labels=person_zone_labels,
                    zones=zones,
                )
                writer.write(annotated_frame)

            if processed_frames % 100 == 0:
                print(f"Processed {processed_frames} frames")
    finally:
        flush_zone_exits(zone_presence, skip_frames, fps)
        if writer is not None:
            writer.release()

    elapsed_seconds = time.perf_counter() - started_at
    average_fps = processed_frames / elapsed_seconds if elapsed_seconds > 0 else 0.0
    all_person_records = [
        build_person_record(camera_id, person, fps, skip_frames, video_duration_seconds)
        for person_id, person in sorted(people.items())
    ]
    valid_person_records = [person for person in all_person_records if person["valid_person"]]
    strict_valid_person_records = [
        person for person in all_person_records if is_strict_valid_person(person)
    ]
    valid_person_ids = {person["person_id"] for person in valid_person_records}
    staff_candidates = sum(
        1 for person in valid_person_records if person["staff_candidate"]
    )
    summary = {
        "unique_people": len(valid_person_records),
        "staff_candidates": staff_candidates,
        "customer_candidates": len(valid_person_records) - staff_candidates,
    }
    raw_summary = {
        "track_ids": [person["person_id"] for person in all_person_records],
        "raw_tracks": len(all_person_records),
    }
    filtered_summary = {
        "person_ids": [person["person_id"] for person in valid_person_records],
        "filtered_people": len(valid_person_records),
    }
    strict_summary = {
        "person_ids": [person["person_id"] for person in strict_valid_person_records],
        "filtered_people": len(strict_valid_person_records),
    }
    person_ranking = [
        {
            "camera_id": person["camera_id"],
            "person_id": person["person_id"],
            "visible_duration_seconds": person["visible_duration_seconds"],
            "frame_count": person["frame_count"],
        }
        for person in sorted(
            all_person_records,
            key=lambda item: item["visible_duration_seconds"],
            reverse=True,
        )
    ]
    zone_analytics_records = build_zone_analytics_records(camera_id, zone_presence, fps)
    zone_summary = build_zone_summary(zone_analytics_records, valid_person_ids, zones)
    top_zones = build_top_zones(zone_summary)
    total_zone_dwell_seconds = sum_zone_dwell_seconds(zone_summary)

    write_json(persons_output_path, all_person_records)
    write_json(summary_output_path, summary)
    write_json(raw_summary_path, raw_summary)
    write_json(filtered_summary_path, filtered_summary)
    write_json(ranking_path, person_ranking)
    write_json(strict_summary_path, strict_summary)
    write_json(zone_analytics_path, zone_analytics_records)
    write_json(zone_summary_path, zone_summary)
    write_json(top_zones_path, top_zones)

    print(f"Raw Tracks: {len(all_person_records)}")
    print(f"Filtered People: {len(valid_person_records)}")
    print(f"Filtered People (strict): {len(strict_valid_person_records)}")
    print("Zone Summary:")
    for zone_name, zone_data in zone_summary.items():
        print(f"- {zone_name}: {zone_data['total_dwell_seconds']:.2f} sec")
    print(f"Processing Time: {elapsed_seconds:.2f} seconds")
    print(f"Total frames processed: {processed_frames}")
    print(f"Average FPS: {average_fps:.2f}")
    print(f"Person analytics saved to: {persons_output_path}")
    print(f"Summary saved to: {summary_output_path}")
    print(f"Raw summary saved to: {raw_summary_path}")
    print(f"Filtered summary saved to: {filtered_summary_path}")
    print(f"Person ranking saved to: {ranking_path}")
    print(f"Strict summary saved to: {strict_summary_path}")
    print(f"Zone analytics saved to: {zone_analytics_path}")
    print(f"Zone summary saved to: {zone_summary_path}")
    print(f"Top zones saved to: {top_zones_path}")
    if save_video:
        if output_path.exists():
            print("Annotated video saved:")
            print(display_path(output_path))
            print(f"Size: {format_size_mb(output_path.stat().st_size)}")
        else:
            raise RuntimeError(f"Annotated video output was not created: {output_path}")

    return {
        "camera_id": camera_id,
        "summary": summary,
        "zone_summary": zone_summary,
        "total_dwell_seconds": total_zone_dwell_seconds,
        "output_dir": output_path.parent,
        "annotated_video_path": output_path if save_video else None,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        print(f"Save Video: {args.save_video}")
        input_paths = build_input_paths(args.all_cameras, args.input, args.camera)
        camera_results = []
        use_custom_output = args.output != DEFAULT_OUTPUT

        for index, input_path in enumerate(input_paths):
            camera_id = get_camera_id(input_path)
            print(f"Processing {camera_id}...")
            result = process_video(
                input_path=input_path,
                output_path=args.output if len(input_paths) == 1 and use_custom_output and index == 0 else None,
                model_name=args.model,
                confidence=args.conf,
                image_size=args.imgsz,
                device=args.device,
                skip_frames=args.skip_frames,
                save_video=args.save_video,
            )
            camera_results.append(result)

        combined_summary_path = build_combined_summary(camera_results, Path("./outputs"))
        total_people = sum(result["summary"]["unique_people"] for result in camera_results)
        total_staff_candidates = sum(
            result["summary"]["staff_candidates"] for result in camera_results
        )
        total_customer_candidates = sum(
            result["summary"]["customer_candidates"] for result in camera_results
        )
        total_dwell_seconds = round(
            sum(result["total_dwell_seconds"] for result in camera_results), 2
        )

        print("Combined Results:")
        print(f"- Total People: {total_people}")
        print(f"- Total Staff Candidates: {total_staff_candidates}")
        print(f"- Total Customer Candidates: {total_customer_candidates}")
        print(f"- Total Dwell Time: {total_dwell_seconds:.2f} seconds")
        print(f"Combined summary saved to: {combined_summary_path}")
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
