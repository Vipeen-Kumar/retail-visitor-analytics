from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT_DIR / "outputs"


def output_path(filename: str) -> Path:
    return OUTPUTS_DIR / filename


def load_json_file(path: Path) -> Any | None:
    """Read a JSON file and show a warning instead of crashing when it is missing."""
    if not path.exists():
        st.warning(f"Missing file: {path.name}")
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        st.error(f"Invalid JSON: {path.name}")
        return None


def metric_card(column, label: str, value: Any) -> None:
    column.metric(label, value)


def format_number(value: Any, decimals: int = 0) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:,.{decimals}f}"
    return str(value)


def format_currency(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"Rs. {float(value):,.2f}"


def extract_person_metrics(persons: list[dict[str, Any]] | None) -> tuple[int, float, int]:
    if not persons:
        return 0, 0.0, 0

    valid_people = [person for person in persons if person.get("valid_person")]
    total_people = len(valid_people)
    total_dwell_seconds = sum(
        float(person.get("visible_duration_seconds", 0.0)) for person in valid_people
    )
    return total_people, total_dwell_seconds, len(persons)


def zone_dwell_total(zone_summary: dict[str, Any] | None) -> float:
    if not zone_summary:
        return 0.0
    return sum(
        float(zone_data.get("total_dwell_seconds", 0.0))
        for zone_data in zone_summary.values()
    )


def zone_summary_to_frame(zone_summary: dict[str, Any] | None) -> pd.DataFrame:
    if not zone_summary:
        return pd.DataFrame(columns=["zone", "total_visits", "total_dwell_seconds"])

    records = []
    for zone_name, zone_data in zone_summary.items():
        records.append(
            {
                "zone": zone_name,
                "total_visits": zone_data.get("total_visits", 0),
                "total_dwell_seconds": zone_data.get("total_dwell_seconds", 0.0),
            }
        )
    return pd.DataFrame(records).sort_values("total_dwell_seconds", ascending=False)


def render_zone_distribution(title: str, zone_summary: dict[str, Any] | None) -> None:
    st.subheader(title)
    zone_df = zone_summary_to_frame(zone_summary)

    if zone_df.empty:
        st.info("No zone summary available.")
        return

    chart_col, table_col = st.columns([1.3, 1])

    with chart_col:
        # Bar chart is the clearest way to compare dwell time across zones.
        dwell_chart = px.bar(
            zone_df,
            x="zone",
            y="total_dwell_seconds",
            title="Zone Dwell Time",
            labels={"zone": "Zone", "total_dwell_seconds": "Dwell (sec)"},
        )
        dwell_chart.update_layout(height=360)
        st.plotly_chart(dwell_chart, use_container_width=True)

    with table_col:
        # Donut view gives a quick share-of-attention summary for each camera.
        pie_chart = px.pie(
            zone_df,
            names="zone",
            values="total_dwell_seconds",
            hole=0.45,
            title="Zone Share",
        )
        pie_chart.update_layout(height=360)
        st.plotly_chart(pie_chart, use_container_width=True)

    top_zone = zone_df.iloc[0]["zone"]
    least_zone = zone_df.iloc[-1]["zone"]
    st.caption(f"Top zone: `{top_zone}` | Least engaged zone: `{least_zone}`")
    st.dataframe(zone_df, use_container_width=True, hide_index=True)


def render_table_chart(
    title: str,
    data: list[dict[str, Any]] | None,
    label_column: str,
    value_column: str,
    table_columns: list[str],
) -> None:
    st.subheader(title)
    if not data:
        st.info("No data available.")
        return

    frame = pd.DataFrame(data)
    if frame.empty or label_column not in frame or value_column not in frame:
        st.info("Data format not available for this section.")
        return

    frame = frame.sort_values(value_column, ascending=False)

    chart = px.bar(
        frame,
        x=label_column,
        y=value_column,
        text_auto=".2s",
        labels={label_column: label_column.replace("_", " ").title(), value_column: value_column.upper()},
    )
    chart.update_layout(height=380)
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(frame[table_columns], use_container_width=True, hide_index=True)

def render_zone_preview(image_name: str, title: str, caption: str) -> None:
    image_path = output_path(image_name)

    if image_path.exists():
        st.subheader(title)

        col1, col2, col3 = st.columns([1, 3, 1])

        with col2:
            st.image(str(image_path), use_container_width=True)

        st.caption(caption)
    else:
        st.warning(f"Missing image: {image_name}")

def render_cctv_section() -> None:
    cam1_persons = load_json_file(output_path("persons_cam1.json"))
    cam2_persons = load_json_file(output_path("persons_cam2.json"))
    cam1_zone_summary = load_json_file(output_path("zone_summary_cam1.json"))
    cam2_zone_summary = load_json_file(output_path("zone_summary_cam2.json"))

    st.title("Store Analytics Dashboard")
    st.header("CCTV Analytics")

    cam1_people, cam1_dwell_seconds, cam1_records = extract_person_metrics(
        cam1_persons if isinstance(cam1_persons, list) else None
    )
    cam2_people, cam2_dwell_seconds, cam2_records = extract_person_metrics(
        cam2_persons if isinstance(cam2_persons, list) else None
    )
    cam1_zone_dwell_total = zone_dwell_total(cam1_zone_summary)
    cam2_zone_dwell_total = zone_dwell_total(cam2_zone_summary)

    cam1_cols = st.columns(3)
    metric_card(cam1_cols[0], "CAM1 Total People", format_number(cam1_people))
    metric_card(
        cam1_cols[1],
        "CAM1 Total Dwell Time",
        f"{format_number(cam1_dwell_seconds, 2)} sec",
    )
    cam1_top = zone_summary_to_frame(cam1_zone_summary)
    metric_card(
        cam1_cols[2],
        "CAM1 Top Zone",
        cam1_top.iloc[0]["zone"] if not cam1_top.empty else "N/A",
    )

    if cam1_people == 0 and cam1_zone_dwell_total > 0:
        st.warning("Inconsistent analytics detected. Check source files.")

    render_zone_preview(
        "cam1_zones_preview.png",
        "CAM1 Zone Layout",
        "CAM1 business zones used for dwell analytics",
    )

    st.markdown(
        "[🎥 View CAM1 Annotated Video](https://drive.google.com/file/d/1v3pf04I5NoWY6np5nFQwuSvN0DU-MGM1/view?usp=sharing)"
    )

    render_zone_distribution(
        "CAM1 Zone Distribution",
        cam1_zone_summary,
    )

    st.divider()

    cam2_cols = st.columns(3)
    metric_card(cam2_cols[0], "CAM2 Total People", format_number(cam2_people))
    metric_card(
        cam2_cols[1],
        "CAM2 Total Dwell Time",
        f"{format_number(cam2_dwell_seconds, 2)} sec",
    )
    cam2_top = zone_summary_to_frame(cam2_zone_summary)
    metric_card(
        cam2_cols[2],
        "CAM2 Top Zone",
        cam2_top.iloc[0]["zone"] if not cam2_top.empty else "N/A",
    )

    if cam2_people == 0 and cam2_zone_dwell_total > 0:
        st.warning("Inconsistent analytics detected. Check source files.")

    render_zone_preview(
        "cam2_zones_preview.png",
        "CAM2 Zone Layout",
        "CAM2 business zones used for dwell analytics",
    )

    st.markdown(
        "[🎥 View CAM2 Annotated Video](https://drive.google.com/file/d/1PXDrcz5Wm2G7K54TMKW7gNCl9k6S39vc/view?usp=sharing)"
    )

    render_zone_distribution(
        "CAM2 Zone Distribution",
        cam2_zone_summary,
    )

    with st.expander("Debug Logs"):
        st.write("Loaded: persons_cam1.json", f"records={cam1_records}")
        st.write("Loaded: persons_cam2.json", f"records={cam2_records}")
        st.write("Loaded: zone_summary_cam1.json", f"zones={len(cam1_zone_summary or {})}")
        st.write("Loaded: zone_summary_cam2.json", f"zones={len(cam2_zone_summary or {})}")
        st.write(
            "CAM1 totals",
            f"people={cam1_people}",
            f"dwell_seconds={format_number(cam1_dwell_seconds, 2)}",
            f"zone_dwell_seconds={format_number(cam1_zone_dwell_total, 2)}",
        )
        st.write(
            "CAM2 totals",
            f"people={cam2_people}",
            f"dwell_seconds={format_number(cam2_dwell_seconds, 2)}",
            f"zone_dwell_seconds={format_number(cam2_zone_dwell_total, 2)}",
        )


def render_sales_section() -> None:
    sales_summary = load_json_file(output_path("sales_summary.json")) or {}
    brand_summary = load_json_file(output_path("brand_summary.json")) or []
    category_summary = load_json_file(output_path("category_summary.json")) or []
    salesperson_summary = load_json_file(output_path("salesperson_summary.json")) or []

    st.title("Store Analytics Dashboard")
    st.header("Sales Analytics")

    cols = st.columns(5)
    metric_card(cols[0], "Total Orders", format_number(sales_summary.get("total_orders", 0)))
    metric_card(cols[1], "Total Quantity", format_number(sales_summary.get("total_qty", 0), 0))
    metric_card(cols[2], "Total GMV", format_currency(sales_summary.get("total_gmv", 0)))
    metric_card(cols[3], "Total NMV", format_currency(sales_summary.get("total_nmv", 0)))
    metric_card(cols[4], "Average Bill", format_currency(sales_summary.get("average_bill", 0)))

    st.divider()

    render_table_chart(
        "Top Brands by NMV",
        brand_summary,
        label_column="brand_name",
        value_column="total_nmv",
        table_columns=["ranking", "brand_name", "total_qty", "total_nmv"],
    )

    st.divider()

    render_table_chart(
        "Top Categories by NMV",
        category_summary,
        label_column="dep_name",
        value_column="total_nmv",
        table_columns=["ranking", "dep_name", "total_qty", "total_nmv"],
    )

    st.divider()

    render_table_chart(
        "Top Salespersons by NMV",
        salesperson_summary,
        label_column="salesperson_name",
        value_column="total_nmv",
        table_columns=["ranking", "salesperson_name", "total_qty", "total_nmv"],
    )


def render_combined_overview() -> None:
    combined_summary = load_json_file(output_path("combined_summary.json")) or {}
    sales_summary = load_json_file(output_path("sales_summary.json")) or {}
    brand_summary = load_json_file(output_path("brand_summary.json")) or []
    salesperson_summary = load_json_file(output_path("salesperson_summary.json")) or []

    st.title("Store Analytics Dashboard")
    st.header("Combined Overview")

    zone_totals = combined_summary.get("zone_totals", {})
    top_engagement_zone = (
        max(zone_totals.items(), key=lambda item: item[1])[0] if zone_totals else "N/A"
    )

    top_brand = brand_summary[0]["brand_name"] if brand_summary else "N/A"
    top_salesperson = salesperson_summary[0]["salesperson_name"] if salesperson_summary else "N/A"

    st.subheader("CCTV Metrics")
    cctv_cols = st.columns(3)
    metric_card(cctv_cols[0], "Total People", format_number(combined_summary.get("combined_people", combined_summary.get("total_people", 0))))
    metric_card(cctv_cols[1], "Total Dwell Time", f"{format_number(combined_summary.get('total_dwell_seconds', 0), 2)} sec")
    metric_card(cctv_cols[2], "Top Engagement Zone", top_engagement_zone)

    st.subheader("Sales Metrics")
    sales_cols = st.columns(4)
    metric_card(sales_cols[0], "Total Orders", format_number(sales_summary.get("total_orders", 0)))
    metric_card(sales_cols[1], "Total Revenue", format_currency(sales_summary.get("total_nmv", 0)))
    metric_card(sales_cols[2], "Top Brand", top_brand)
    metric_card(sales_cols[3], "Top Salesperson", top_salesperson)

    zone_df = pd.DataFrame(
        [{"zone": zone, "dwell_seconds": dwell} for zone, dwell in zone_totals.items()]
    )
    if not zone_df.empty:
        st.divider()
        # Combined zone view helps connect store engagement with sales outcomes.
        chart = px.bar(
            zone_df.sort_values("dwell_seconds", ascending=False),
            x="zone",
            y="dwell_seconds",
            title="Combined Zone Engagement",
            labels={"zone": "Zone", "dwell_seconds": "Dwell (sec)"},
        )
        chart.update_layout(height=360)
        st.plotly_chart(chart, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Store Analytics Dashboard", layout="wide")
    st.sidebar.title("Store Analytics Dashboard")
    section = st.sidebar.radio(
        "Navigation",
        ["CCTV Analytics", "Sales Analytics", "Combined Overview"],
    )

    if section == "CCTV Analytics":
        render_cctv_section()
    elif section == "Sales Analytics":
        render_sales_section()
    else:
        render_combined_overview()


if __name__ == "__main__":
    main()
