# Retail Store Analytics Platform

## Overview

This project combines:

1. CCTV-based customer analytics
2. Sales analytics

to provide store performance insights.

It is designed as a demo-ready analytics platform for retail environments, hackathons, portfolio projects, and business presentations.

## Features

### Customer Analytics

- YOLOv8 person detection
- ByteTrack tracking
- Multi-camera support
- Zone analytics
- Dwell time analytics
- Customer movement insights

### Sales Analytics

- Sales summaries
- Brand performance
- Category performance
- Salesperson rankings

### Dashboard

- CCTV analytics dashboard
- Sales analytics dashboard
- Combined business overview

## Project Structure

```text
yolo/
├── CCTV Footage/
│   ├── CAM 1.mp4
│   └── CAM 2.mp4
├── csv/
│   ├── Brigade_Bangalore_10_April_26 (1)bc6219c.csv
│   └── Brigade Road - Store layoutc5f5d56.xlsx
├── dashboard/
│   └── app.py
├── outputs/
│   ├── annotated_cam1.mp4
│   ├── annotated_cam2.mp4
│   ├── combined_summary.json
│   ├── persons_cam1.json
│   ├── persons_cam2.json
│   ├── zone_summary_cam1.json
│   ├── zone_summary_cam2.json
│   ├── sales_summary.json
│   ├── brand_summary.json
│   ├── category_summary.json
│   └── salesperson_summary.json
├── src/
│   └── cctv_tracker/
│       ├── __init__.py
│       ├── cli.py
│       └── sales_analytics.py
├── pyproject.toml
└── README.md
```

## Installation

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it:

```bash
.venv\Scripts\activate
```

3. Install the project and dashboard dependencies:

```bash
pip install -e .
```

## Running CCTV Analytics

Run the default single-camera pipeline:

```bash
track-cctv
```

Process a specific camera:

```bash
track-cctv --camera CAM1
track-cctv --camera CAM2
```

Process both cameras:

```bash
track-cctv --camera ALL
```

Process both cameras and save annotated videos:

```bash
track-cctv --camera ALL --save-video
```

You can also run the module directly:

```bash
python -m cctv_tracker.cli --camera ALL --save-video
```

## Running Sales Analytics

```bash
python -m cctv_tracker.sales_analytics
```

## Running Dashboard

```bash
streamlit run dashboard/app.py
```

## Outputs

### CCTV Outputs

- `persons_cam1.json`
- `persons_cam2.json`
- `summary_cam1.json`
- `summary_cam2.json`
- `summary_raw_cam1.json`
- `summary_raw_cam2.json`
- `summary_filtered_cam1.json`
- `summary_filtered_cam2.json`
- `summary_strict_cam1.json`
- `summary_strict_cam2.json`
- `person_ranking_cam1.json`
- `person_ranking_cam2.json`
- `zone_analytics_cam1.json`
- `zone_analytics_cam2.json`
- `zone_summary_cam1.json`
- `zone_summary_cam2.json`
- `top_zones_cam1.json`
- `top_zones_cam2.json`
- `combined_summary.json`
- `annotated_cam1.mp4`
- `annotated_cam2.mp4`

### Sales Outputs

- `sales_summary.json`
- `brand_summary.json`
- `category_summary.json`
- `salesperson_summary.json`

### Preview / Debug Outputs

- `cam1_first_frame.jpg`
- `cam2_first_frame.jpg`
- `cam1_zones_preview.png`
- `cam2_zones_preview.png`

## Dashboard Overview

The Streamlit dashboard includes three major views:

- `CCTV Analytics`
  Shows per-camera people counts, dwell time, zone distribution, and top engagement zones.
- `Sales Analytics`
  Shows order, quantity, GMV, NMV, average bill, and top performers across brands, categories, and salespeople.
- `Combined Overview`
  Merges CCTV and sales metrics into a high-level business summary.

The dashboard is built to handle missing JSON files gracefully and will show warnings instead of crashing when data is unavailable.

## Future Improvements

- Heatmaps
- Product analytics
- Hourly sales analytics
- Real-time CCTV monitoring
