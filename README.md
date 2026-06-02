# CCTV Person Tracker

Simple Python project for processing CCTV footage with:

- YOLOv8 for person detection
- ByteTrack for stable track IDs
- OpenCV for video I/O and drawing

## What it does

- Detects only people
- Assigns track IDs
- Draws bounding boxes
- Draws track IDs
- Saves an annotated output video
- Builds person-presence analytics from tracking data

## Input / Output

- Inputs:
  `./CCTV Footage/CAM 1.mp4`
  `./CCTV Footage/CAM 2.mp4`
- Output video: `./outputs/annotated_cam1.mp4`
- Output video: `./outputs/annotated_cam2.mp4`
- Person analytics: `./outputs/persons_cam1.json`
- Person analytics: `./outputs/persons_cam2.json`
- Summary: `./outputs/summary_cam1.json`
- Summary: `./outputs/summary_cam2.json`
- Raw summary: `./outputs/summary_raw_cam1.json`
- Raw summary: `./outputs/summary_raw_cam2.json`
- Filtered summary: `./outputs/summary_filtered_cam1.json`
- Filtered summary: `./outputs/summary_filtered_cam2.json`
- Person ranking: `./outputs/person_ranking_cam1.json`
- Person ranking: `./outputs/person_ranking_cam2.json`
- Strict summary: `./outputs/summary_strict_cam1.json`
- Strict summary: `./outputs/summary_strict_cam2.json`
- Zone analytics: `./outputs/zone_analytics_cam1.json`
- Zone analytics: `./outputs/zone_analytics_cam2.json`
- Zone summary: `./outputs/zone_summary_cam1.json`
- Zone summary: `./outputs/zone_summary_cam2.json`
- Top zones: `./outputs/top_zones_cam1.json`
- Top zones: `./outputs/top_zones_cam2.json`
- Combined summary: `./outputs/combined_summary.json`

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Run

```bash
track-cctv
```

Or:

```bash
python -m cctv_tracker.cli
```

Process both cameras:

```bash
track-cctv --all-cameras
```

Select a single camera explicitly:

```bash
track-cctv --camera CAM1
track-cctv --camera CAM2
track-cctv --camera ALL
```

## Optional arguments

```bash
python -m cctv_tracker.cli ^
  --input ".\CCTV Footage\CAM 1.mp4" ^
  --output ".\outputs\annotated_cam1.mp4" ^
  --model "yolov8n.pt" ^
  --conf 0.35 ^
  --imgsz 640 ^
  --skip-frames 2 ^
  --save-video
```

Multi-camera with videos:

```bash
track-cctv --all-cameras --save-video
```

Equivalent explicit selector:

```bash
track-cctv --camera ALL --save-video
```

## Video Generation

Annotated video generation is disabled by default for faster analytics iteration.

Use:

```bash
track-cctv --save-video
```

to generate annotated videos.

If enabled, the video overlay also draws:

- Zone polygons
- The current zone name near each tracked person

## Notes

- The default YOLO class filter is `person` only.
- ByteTrack is enabled through Ultralytics tracking with `tracker="bytetrack.yaml"`.
- The default image size is `640` for faster processing.
- `--skip-frames N` uses video stride to process every Nth frame while keeping the saved output playable.
- Current filtering keeps only people with `visible_duration_seconds >= 15` and `frame_count >= 300`.
- Strict filtering reports an additional stricter estimate using `visible_duration_seconds >= 30` and `frame_count >= 600`.
- Zone analytics uses manual polygon zones for `CAM 1` and assigns each person by bounding-box center point.
- Zone analytics uses manual polygon zones for `CAM 1` and `CAM 2` and assigns each person by bounding-box center point.
- Zone dwell times are measured from explicit zone entry and exit timing, including zone switches, track disappearance, and end-of-video flushes.
- Track IDs remain isolated per camera, and person records include `camera_id`.
- Combined analytics include per-camera people counts, per-camera dwell totals, and aggregated `zone_totals`.
- Progress is logged every 100 processed frames, followed by total frames, total time, and average FPS.
- Each tracked person records first seen time, last seen time, visible duration, and a simple `staff_candidate` flag.
- Staff candidates are inferred using long visible duration plus broad coverage across the video timeline.
- Output filenames are derived from the input camera name, such as `cam1`, `cam2`, and so on.
- On first run, model weights such as `yolov8n.pt` may be downloaded automatically if they are not already available.

## Modified Files

File:
`src/cctv_tracker/cli.py`

Exact file path:
`C:/Users/vipee/Desktop/study/yolo/src/cctv_tracker/cli.py`

Functions:
`build_parser()`
`build_input_paths()`
`get_camera_id()`
`build_summary_paths()`
`build_zone_paths()`
`build_combined_summary()`
`build_person_record()`
`annotate_frame()`
`update_zone_presence()`
`process_video()`
`create_writer()`
`main()`

Lines:
approximately `15-110`, `180-260`, `300-470`, and `520-650`
