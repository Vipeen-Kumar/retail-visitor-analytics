# Retail Visitor Analytics — Engineering Choices & Rationale

This document explains the technical decisions, architecture choices, and implementation strategies selected for the **Retail Visitor Analytics** platform.

---

## Model Selection

To capture reliable store visitor metrics, the platform requires high-accuracy object detection and persistent multi-object tracking (MOT) capable of running on resource-constrained edge hardware.

### Object Detection: YOLOv8
We selected the **YOLOv8** (You Only Look Once, Version 8) framework, specifically the **YOLOv8 Nano** (`yolov8n.pt`) model, for person detection.

#### Advantages:
1. **State-of-the-Art Accuracy**: YOLOv8 utilizes an anchor-free detection head and a decoupled architecture that separates classification and bounding box regression. This architecture significantly improves detection accuracy on occluded figures and diverse retail lighting profiles.
2. **Computational Real-time Capability**: On typical edge deployment hardware (including modern CPUs and low-tier edge GPUs), YOLOv8 Nano performs inference in single-digit milliseconds per frame. By utilizing the `--skip-frames` parameter to skip redundant frames, we achieve processing speeds exceeding **50+ FPS** on standard CPU systems.
3. **Developer-Friendly API**: The native `ultralytics` Python API simplifies integration, configuration, and weights export (e.g., converting to ONNX or TensorRT) compared to older model architectures.

---

### Multi-Object Tracking: ByteTrack
Tracking individual customers across frames is necessary to calculate dwell times and filter duplicate counts. We selected **ByteTrack** as our tracking algorithm.

#### Advantages:
1. **Stable Identity Tracking**: Unlike older trackers (like SORT) which discard low-confidence bounding boxes, ByteTrack matches almost every detection box. It runs a second-stage association on low-score boxes using Kalman filter motion predictions. In busy retail environments where shoppers frequently walk behind store aisles, displays, or other customers, this method reduces **ID switching** and maintains stable visitor tracking.
2. **Lightweight & Out-of-the-Box Integration**: ByteTrack is natively supported by Ultralytics, allowing us to initialize the tracker directly inside the PyTorch inference generator loop (`model.track(tracker="bytetrack.yaml", persist=True)`) without introducing heavy external compiled library dependencies.

---

## Zone Analytics Design

Retail store managers need spatial segmentation rather than simple bounding-box coordinate tracking. 

```
   Perspective Camera View:
   ┌──────────────────────────────────────────────┐
   │                                              │
   │      /──────────────────────────\            │
   │     /   Skincare Wall Zone     /             │
   │    /__________________________/              │
   │                                              │
   │       /\                                     │
   │      /  \  Makeup Unit                       │
   │     /____\  Zone                             │
   │                                              │
   └──────────────────────────────────────────────┘
```

### Why Polygon-Based Zones?
* **Perspective Calibration**: Camera feeds in retail environments are typically mounted overhead at oblique angles, causing perspective distortion. Axis-aligned rectangles are inaccurate because they include background traffic. Polygons allow us to map skewed quadrilateral shapes to match actual 3D floor regions.
* **Exact Geometric Boundaries**: Polygons conform to irregular store shapes, such as diagonal walkways, L-shaped checkout counters, and curved promotional displays.
* **Efficient Containment Matching**: By resolving the person's location down to a single bottom-center coordinate (representing their feet on the store floor) and utilizing OpenCV's contour checking (`cv2.pointPolygonTest`), we perform spatial filtering with negligible CPU overhead.

### Zone Descriptions
1. **`skincare_wall`**: Tracks traffic in the skincare aisle. This zone captures customer interest in high-margin skincare brands and premium products.
2. **`makeup_unit`**: Tracks the central island cosmetic and tester stands. Dwell time in this area indicates high-engagement browsing.
3. **`mirror_area`**: Positioned in front of the vanity or wall mirrors. Lingering in this zone indicates that customers are actively testing or applying products.
4. **`front_display`**: Placed near the entrance to monitor how many incoming shoppers stop to inspect new promotional displays.

---

## Output Schema Design

The platform outputs metrics to flat JSON files under the `outputs/` folder, separating spatial, transactional, and aggregated dimensions.

### Why JSON Flat-File Architecture?
1. **Serverless Footprint**: Generating flat files removes the overhead of deploying and configuring database servers (e.g., PostgreSQL, MongoDB) during initial deployments, making edge hardware setups lightweight.
2. **Interoperability**: JSON is natively supported across almost all programming languages, APIs, and business intelligence (BI) systems, allowing for easy integration.
3. **Flexible Schema**: Nested tracking records (such as mapping a single visitor’s visits across multiple zones) are represented easily in JSON without requiring complex table joins or database schema migrations.

### Key Outputs and Schemas

* **`persons_cam1.json` / `persons_cam2.json`**
  Contains the master tracking registry. Each record logs the tracked ID, absolute arrival/departure times, total frames tracked, and key metadata flags (`valid_person` and `staff_candidate`).
* **`zone_summary_cam1.json` / `zone_summary_cam2.json`**
  Aggregates traffic performance per zone. Stores total visitor counts and cumulative dwell seconds, which the dashboard uses to rank engagement zones.
* **`sales_summary.json`**
  Summarizes store performance metrics, compiling total order volume, quantity sold, Gross Merchandise Value (GMV), Net Merchandise Value (NMV), and average ticket value.
* **`brand_summary.json`**
  Renders a sorted leaderboard of sales performance by brand, displaying quantity sold and net revenue contribution.
* **`category_summary.json`**
  Groups sales data by department code (`dep_name`) to identify which product categories generate the highest revenue.
* **`salesperson_summary.json`**
  Tracks employee performance, ranking sales staff by transaction volume and net value.

---

## Dashboard Architecture

The frontend is designed to run in retail store offices or corporate headquarters to display visitor analytics.

### Streamlit Selection
* **Python-Native Integration**: Since our YOLO object tracking and Pandas analytics scripts are written in Python, Streamlit allows us to build interactive web applications without context-switching to JavaScript frameworks (like React or Vue).
* **Stateless Top-to-Bottom Execution**: Streamlit's model executes the script from top to bottom on user interaction. This simplifies the process of reloading and rendering updated JSON files from the disk.
* **Interactive Layout Controls**: Layout features like `st.columns`, `st.sidebar`, and `st.metric` allow us to build clean dashboards with minimal UI code.

### Plotly Selection
* **Interactive Hover States**: Plotly generates responsive HTML5 charts. Shoppers' dwell times and brand performance metrics can be inspected directly on hover, which is essential for business reporting.
* **Declarative API (Plotly Express)**: Integrates directly with Pandas DataFrames, allowing developers to generate complex visualizations (like donut charts for zone attention share and sorted bar charts) in single lines of code.

---

## Future Improvements

To scale this platform into a enterprise-grade SaaS product, we recommend the following roadmap:

### 1. Cumulative Spatial Heatmaps
Generate 2D spatial density maps by tracking and accumulating the coordinates of every visitor across all frames. These coordinates can be plotted as semi-transparent color overlays (red-to-blue) on top of the store layout frame, allowing managers to visualize path flows and physical bottlenecks.

### 2. Live Stream Processing
Transition from offline MP4 video files to live edge ingestion. We can implement this by subscribing to RTSP (Real-Time Streaming Protocol) network camera streams and feeding bounding-box coordinates to a lightweight queue broker (e.g., Redis or MQTT) for processing.

### 3. Queue Detection & Alerts
Define custom polygon regions near cash registers. By tracking the number of active customer IDs lingering in these regions, the system can trigger automated alerts (via Slack, SMS, or Telegram) to open new registers when queue lengths exceed a target threshold.

### 4. Conversion Analytics
By integrating store transactions with visitor tracking, we can calculate true conversion rates:
$$\text{Zone Conversion Rate} = \frac{\text{POS Purchases from Category } X}{\text{Unique Customer Visits to Zone } X}$$
This metric isolates whether a sales decline is caused by low foot traffic (poor display/marketing) or a low close rate (poor pricing/sales assistance).
