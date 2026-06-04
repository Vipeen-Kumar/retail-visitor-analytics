# Retail Store Analytics Platform

## Overview

Retail Store Analytics Platform combines computer vision and sales analytics to help retailers understand customer behavior and business performance.

The system analyzes CCTV footage using YOLOv8 and ByteTrack to measure visitor movement, dwell time, and engagement across different store zones. It also processes transaction data to generate insights about sales performance, product categories, brands, and salesperson effectiveness.

All analytics are presented through an interactive Streamlit dashboard.

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

## Business Value

The platform helps retailers:

- Understand customer movement patterns
- Identify high-engagement store zones
- Measure customer dwell time
- Optimize store layouts
- Evaluate promotional areas
- Analyze brand performance
- Compare category performance
- Track salesperson effectiveness

## Business Value

The platform helps retailers:

- Understand customer movement patterns
- Identify high-engagement store zones
- Measure customer dwell time
- Optimize store layouts
- Evaluate promotional areas
- Analyze brand performance
- Compare category performance
- Track salesperson effectiveness

## Project Structure

```text
yolo/
├── CCTV Footage/
│   ├── CAM 1.mp4
│   ├── CAM 2.mp4
│   ├── CAM 3.mp4
│   ├── CAM 4.mp4
│   └── CAM 5.mp4
│
├── csv/
│   ├── Brigade_Bangalore_10_April_26 (1)bc6219c.csv
│   └── Brigade Road - Store layoutc5f5d56.xlsx
│
├── dashboard/
│   └── app.py
│
├── outputs/
│   ├── annotated_cam1.mp4
│   ├── annotated_cam2.mp4
│   ├── cam1_first_frame.jpg
│   ├── cam2_first_frame.jpg
│   ├── cam1_zones_preview.png
│   ├── cam2_zones_preview.png
│   ├── persons_cam1.json
│   ├── persons_cam2.json
│   ├── person_ranking_cam1.json
│   ├── person_ranking_cam2.json
│   ├── summary_cam1.json
│   ├── summary_cam2.json
│   ├── summary_raw_cam1.json
│   ├── summary_raw_cam2.json
│   ├── summary_filtered_cam1.json
│   ├── summary_filtered_cam2.json
│   ├── summary_strict_cam1.json
│   ├── summary_strict_cam2.json
│   ├── zone_analytics_cam1.json
│   ├── zone_analytics_cam2.json
│   ├── zone_summary_cam1.json
│   ├── zone_summary_cam2.json
│   ├── top_zones_cam1.json
│   ├── top_zones_cam2.json
│   ├── combined_summary.json
│   ├── sales_summary.json
│   ├── brand_summary.json
│   ├── category_summary.json
│   ├── salesperson_summary.json
│   └── events.jsonl
│
├── src/
│   └── cctv_tracker/
│       ├── __init__.py
│       ├── cli.py
│       ├── sales_analytics.py
│       ├── sales_analytics.ipynb
│       └── generate_events.py
│
├── dashboard/
│   └── app.py
│
├── CHOICES.md
├── DESIGN.md
├── README.md
├── requirements.txt
├── pyproject.local.toml
└── yolov8n.pt
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

## Key Outputs

### CCTV Analytics

- Unique visitor counts
- Zone-wise dwell time
- Zone visit frequency
- Customer rankings
- Staff candidate identification

### Sales Analytics

- Total Orders
- Total Quantity Sold
- Gross Merchandise Value (GMV)
- Net Merchandise Value (NMV)
- Average Bill Value
- Brand Rankings
- Category Rankings
- Salesperson Rankings

## Future Improvements

- Real-time CCTV stream processing
- Store heatmap generation
- Queue detection and alerts
- Customer path visualization
- Product conversion analytics
- Multi-store analytics dashboard
- Cloud deployment and centralized monitoring
