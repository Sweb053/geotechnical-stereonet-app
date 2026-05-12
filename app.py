from __future__ import annotations

import json
import re
from io import BytesIO
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import streamlit as st

COLORS = {"Slope": "#d1495b", "Discontinuity": "#00798c"}
PLANAR = "#7c2d12"
WEDGE = "#4f46e5"
TOPPLE = "#9333ea"
DEFAULTS_PATH = Path(__file__).with_name("defaults.json")
REPORTS_DIR = Path(__file__).with_name("reports")


def load_defaults() -> dict:
    fallback = {
        "location_id": "",
        "slope_dip_direction": 135.0,
        "slope_dip": 45.0,
        "friction_angle": 30.0,
        "lateral_limit": 20.0,
        "enable_planar": True,
        "enable_wedge": True,
        "enable_toppling": True,
        "show_analysis_zones": True,
        "show_table": True,
        "sets": [
            {"Plot": True, "Name": "S1", "Dip direction": 120.0, "Dip": 35.0},
            {"Plot": True, "Name": "S2", "Dip direction": 95.0, "Dip": 70.0},
            {"Plot": True, "Name": "S3", "Dip direction": 210.0, "Dip": 55.0},
        ],
    }
    try:
        loaded = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    defaults = fallback | loaded
    if "sets" not in loaded:
        defaults["sets"] = []
        for row in loaded.get("foliation", fallback["sets"][:1]):
            row = dict(row)
            row.pop("Type", None)
            defaults["sets"].append(row)
        for row in loaded.get("joints", fallback["sets"][1:]):
            row = dict(row)
            row.pop("Type", None)
            defaults["sets"].append(row)
    else:
        defaults["sets"] = [{k: v for k, v in row.items() if k != "Type"} for row in defaults.get("sets", fallback["sets"])]
    return defaults


def clean(value, limit, name):
    if pd.isna(value):
        raise ValueError(f"{name} is missing")
    value = float(value)
    if not 0 <= value <= limit:
        raise ValueError(f"{name} must be between 0 and {limit:g}")
    return value


def angle_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def angle_diff_array(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def pole_from_plane(dip_direction, dip):
    return (dip_direction + 180.0) % 360.0, 90.0 - dip


def vector_from_trend_plunge(trend, plunge):
    trend_rad = np.deg2rad(trend)
    plunge_rad = np.deg2rad(plunge)
    return np.array([
        np.sin(trend_rad) * np.cos(plunge_rad),
        np.cos(trend_rad) * np.cos(plunge_rad),
        -np.sin(plunge_rad),
    ])


def vector_components(trend, plunge):
    trend_rad = np.deg2rad(trend)
    plunge_rad = np.deg2rad(plunge)
    return (
        np.sin(trend_rad) * np.cos(plunge_rad),
        np.cos(trend_rad) * np.cos(plunge_rad),
        -np.sin(plunge_rad),
    )


def trend_plunge_from_vectors(vectors):
    vectors = np.asarray(vectors, dtype=float)
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    vectors = np.where(vectors[:, 2:3] > 0, -vectors, vectors)
    horizontal = np.hypot(vectors[:, 0], vectors[:, 1])
    trend = (np.rad2deg(np.arctan2(vectors[:, 0], vectors[:, 1])) + 360.0) % 360.0
    plunge = np.rad2deg(np.arctan2(-vectors[:, 2], horizontal))
    return trend, plunge


def project(trend, plunge):
    trend_rad = np.deg2rad(trend)
    radius = np.tan(np.deg2rad((90.0 - plunge) / 2.0))
    return radius * np.sin(trend_rad), radius * np.cos(trend_rad)


def plane_normal(dip_direction, dip):
    trend, plunge = pole_from_plane(dip_direction, dip)
    return vector_from_trend_plunge(trend, plunge)


def plane_daylights(dip_direction, dip, slope_dd, slope_dip):
    slope_normal = plane_normal(slope_dd, slope_dip)
    dip_vector = vector_from_trend_plunge(dip_direction, dip)
    return float(np.dot(slope_normal, dip_vector)) < -1e-10


def line_daylights(trend, plunge, slope_dd, slope_dip):
    slope_normal = plane_normal(slope_dd, slope_dip)
    line_vector = vector_from_trend_plunge(trend, plunge)
    return float(np.dot(slope_normal, line_vector)) < -1e-10


def intersection(first, second):
    line = np.cross(
        plane_normal(float(first["dip_direction"]), float(first["dip"])),
        plane_normal(float(second["dip_direction"]), float(second["dip"])),
    )
    if np.linalg.norm(line) < 1e-9:
        return None
    trend, plunge = trend_plunge_from_vectors(np.array([line]))
    return float(trend[0]), float(plunge[0])


def great_circle(dip_direction, dip, samples=361):
    normal = plane_normal(dip_direction, dip)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, reference))) > 0.95:
        reference = np.array([1.0, 0.0, 0.0])
    axis_a = np.cross(normal, reference)
    axis_a = axis_a / np.linalg.norm(axis_a)
    axis_b = np.cross(normal, axis_a)
    axis_b = axis_b / np.linalg.norm(axis_b)
    angles = np.linspace(0.0, 2.0 * np.pi, samples)
    vectors = np.cos(angles)[:, None] * axis_a + np.sin(angles)[:, None] * axis_b
    trend, plunge = trend_plunge_from_vectors(vectors)
    x, y = project(trend, plunge)
    breaks = np.where(np.hypot(np.diff(x), np.diff(y)) > 0.35)[0] + 1
    return [(xp, yp) for xp, yp in zip(np.split(x, breaks), np.split(y, breaks)) if len(xp) > 1]


def build_orientations(slope_dd, slope_dip, sets):
    orientations = [{
        "type": "Slope",
        "label": "Slope",
        "dip_direction": clean(slope_dd, 360, "Slope dip direction") % 360,
        "dip": clean(slope_dip, 90, "Slope dip"),
    }]
    for index, row in sets.iterrows():
        if not bool(row.get("Plot", True)):
            continue
        label = str(row.get("Name") or f"S{index + 1}")
        orientations.append({
            "type": "Discontinuity",
            "label": label,
            "dip_direction": clean(row.get("Dip direction"), 360, f"{label} dip direction") % 360,
            "dip": clean(row.get("Dip"), 90, f"{label} dip"),
        })
    return orientations


def planar_result(item, slope_dd, slope_dip, friction, lateral):
    alignment = angle_diff(item["dip_direction"], slope_dd)
    aligned = alignment <= lateral
    exceeds_friction = item["dip"] > friction
    daylights = plane_daylights(item["dip_direction"], item["dip"], slope_dd, slope_dip)
    susceptible = aligned and exceeds_friction and daylights
    reasons = []
    if not aligned:
        reasons.append("outside lateral limit")
    if not exceeds_friction:
        reasons.append("dip <= friction")
    if not daylights:
        reasons.append("does not daylight")
    return {
        "alignment": alignment,
        "daylights": daylights,
        "exceeds_friction": exceeds_friction,
        "susceptible": susceptible,
        "reason": "Meets planar sliding criteria" if susceptible else ", ".join(reasons),
    }


def toppling_result(item, slope_dd, slope_dip, friction, lateral):
    threshold = max(0.0, 90.0 - slope_dip + friction)
    alignment = angle_diff(item["dip_direction"], (slope_dd + 180.0) % 360.0)
    into_slope = alignment <= lateral
    steep = item["dip"] > threshold
    susceptible = into_slope and steep
    reasons = []
    if not into_slope:
        reasons.append("not dipping into slope")
    if not steep:
        reasons.append("dip below toppling threshold")
    return {
        "alignment": alignment,
        "threshold": threshold,
        "into_slope": into_slope,
        "steep": steep,
        "susceptible": susceptible,
        "reason": "Meets toppling criteria" if susceptible else ", ".join(reasons),
    }


def analyse_planar(orientations, slope_dd, slope_dip, friction, lateral):
    rows = []
    for item in orientations:
        if item["type"] == "Slope":
            item["planar"] = None
            continue
        result = planar_result(item, slope_dd, slope_dip, friction, lateral)
        item["planar"] = result
        rows.append({
            "Type": item["type"],
            "Name": item["label"],
            "Dip direction": f"{item['dip_direction']:03.0f}",
            "Dip": f"{item['dip']:02.0f}",
            "Alignment": f"{result['alignment']:.1f}",
            "Daylights": "Yes" if result["daylights"] else "No",
            "Dip > friction": "Yes" if result["exceeds_friction"] else "No",
            "Planar sliding": "Potential" if result["susceptible"] else "No",
            "Reason": result["reason"],
        })
    return pd.DataFrame(rows)


def analyse_toppling(orientations, slope_dd, slope_dip, friction, lateral):
    rows = []
    for item in orientations:
        if item["type"] == "Slope":
            item["toppling"] = None
            continue
        result = toppling_result(item, slope_dd, slope_dip, friction, lateral)
        item["toppling"] = result
        rows.append({
            "Type": item["type"],
            "Name": item["label"],
            "Dip direction": f"{item['dip_direction']:03.0f}",
            "Dip": f"{item['dip']:02.0f}",
            "Into-slope alignment": f"{result['alignment']:.1f}",
            "Threshold dip": f"{result['threshold']:.1f}",
            "Dips into slope": "Yes" if result["into_slope"] else "No",
            "Toppling": "Potential" if result["susceptible"] else "No",
            "Reason": result["reason"],
        })
    return pd.DataFrame(rows)


def analyse_wedge(orientations, slope_dd, slope_dip, friction, lateral):
    discontinuities = [item for item in orientations if item["type"] != "Slope"]
    rows, results = [], []
    for first, second in combinations(discontinuities, 2):
        line = intersection(first, second)
        if line is None:
            continue
        trend, plunge = line
        alignment = angle_diff(trend, slope_dd)
        aligned = alignment <= lateral
        exceeds_friction = plunge > friction
        daylights = line_daylights(trend, plunge, slope_dd, slope_dip)
        susceptible = aligned and exceeds_friction and daylights
        reasons = []
        if not aligned:
            reasons.append("outside lateral limit")
        if not exceeds_friction:
            reasons.append("plunge <= friction")
        if not daylights:
            reasons.append("does not daylight")
        result = {
            "first": first["label"],
            "second": second["label"],
            "trend": trend,
            "plunge": plunge,
            "alignment": alignment,
            "daylights": daylights,
            "exceeds_friction": exceeds_friction,
            "susceptible": susceptible,
            "reason": "Meets wedge sliding criteria" if susceptible else ", ".join(reasons),
        }
        results.append(result)
        rows.append({
            "Planes": f"{first['label']} + {second['label']}",
            "Trend": f"{trend:03.0f}",
            "Plunge": f"{plunge:02.0f}",
            "Alignment": f"{alignment:.1f}",
            "Daylights": "Yes" if daylights else "No",
            "Plunge > friction": "Yes" if exceeds_friction else "No",
            "Wedge sliding": "Potential" if susceptible else "No",
            "Reason": result["reason"],
        })
    return pd.DataFrame(rows), results


def add_grid(ax):
    ax.add_patch(plt.Circle((0, 0), 1, edgecolor="#202124", facecolor="none", linewidth=1.4))
    theta = np.linspace(0, 2 * np.pi, 361)
    for angle in range(0, 360, 10):
        rad = np.deg2rad(angle)
        ax.plot(
            [0, np.sin(rad)],
            [0, np.cos(rad)],
            color="#b7c0c9" if angle % 30 == 0 else "#d7dde2",
            linewidth=1.1 if angle % 30 == 0 else 0.8,
            zorder=0,
        )
    for radius in np.tan(np.deg2rad((90.0 - np.arange(10, 90, 10)) / 2.0)):
        ax.plot(radius * np.sin(theta), radius * np.cos(theta), color="#d7dde2", linewidth=0.8, zorder=0)
    for text, x, y, ha, va in [
        ("N", 0, 1.08, "center", "bottom"),
        ("E", 1.08, 0, "left", "center"),
        ("S", 0, -1.08, "center", "top"),
        ("W", -1.08, 0, "right", "center"),
    ]:
        ax.text(x, y, text, ha=ha, va=va, fontsize=11, weight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-1.13, 1.13)
    ax.set_ylim(-1.13, 1.13)
    ax.axis("off")


def draw_zones(ax, show_planar, show_wedge, show_toppling, slope_dd, slope_dip, friction, lateral):
    grid = np.linspace(-1.0, 1.0, 251)
    xx, yy = np.meshgrid(grid, grid)
    radius = np.hypot(xx, yy)
    inside = radius <= 1.0
    trend = (np.rad2deg(np.arctan2(xx, yy)) + 360.0) % 360.0
    plunge = 90.0 - np.rad2deg(2.0 * np.arctan(radius))
    normal = plane_normal(slope_dd, slope_dip)

    def fill(mask, color):
        if np.any(mask):
            values = np.ma.masked_where(~mask, mask.astype(float))
            ax.contourf(xx, yy, values, levels=[0.5, 1.5], colors=[color], alpha=0.18, zorder=0.45)

    if show_planar:
        dd = (trend + 180.0) % 360.0
        dip = 90.0 - plunge
        dx, dy, dz = vector_components(dd, dip)
        daylight = normal[0] * dx + normal[1] * dy + normal[2] * dz < -1e-10
        fill(inside & (angle_diff_array(dd, slope_dd) <= lateral) & (dip > friction) & daylight, PLANAR)
    if show_wedge:
        lx, ly, lz = vector_components(trend, plunge)
        daylight = normal[0] * lx + normal[1] * ly + normal[2] * lz < -1e-10
        fill(inside & (angle_diff_array(trend, slope_dd) <= lateral) & (plunge > friction) & daylight, WEDGE)
    if show_toppling:
        dd = (trend + 180.0) % 360.0
        dip = 90.0 - plunge
        threshold = max(0.0, 90.0 - slope_dip + friction)
        fill(inside & (angle_diff_array(dd, (slope_dd + 180.0) % 360.0) <= lateral) & (dip > threshold), TOPPLE)


def add_guides(ax, show_planar, show_wedge, show_toppling, slope_dd, slope_dip, friction, lateral):
    for center in [slope_dd, (slope_dd + 180.0) % 360.0]:
        for bearing in [(center - lateral) % 360.0, (center + lateral) % 360.0]:
            rad = np.deg2rad(bearing)
            ax.plot([0, np.sin(rad)], [0, np.cos(rad)], color=PLANAR, linewidth=1.2, linestyle="--", alpha=0.55, zorder=1)
    theta = np.linspace(0, 2 * np.pi, 361)
    if show_planar or show_toppling:
        radius = np.tan(np.deg2rad(friction / 2.0))
        ax.plot(radius * np.sin(theta), radius * np.cos(theta), color="#9a3412", linewidth=1.4, linestyle=":")
    if show_wedge:
        radius = np.tan(np.deg2rad((90.0 - friction) / 2.0))
        ax.plot(radius * np.sin(theta), radius * np.cos(theta), color=WEDGE, linewidth=1.4, linestyle=":")
    for x, y in great_circle(slope_dd, slope_dip):
        ax.plot(x, y, color="#57534e", linewidth=1.5, linestyle="-.")


def plot_stereonet(orientations, wedge_results, show_planar, show_wedge, show_toppling, show_zones, slope_dd, slope_dip, friction, lateral):
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=140)
    add_grid(ax)
    show_any = show_planar or show_wedge or show_toppling
    if show_zones and show_any:
        draw_zones(ax, show_planar, show_wedge, show_toppling, slope_dd, slope_dip, friction, lateral)
    if show_any:
        add_guides(ax, show_planar, show_wedge, show_toppling, slope_dd, slope_dip, friction, lateral)

    for item in orientations:
        color = COLORS.get(item["type"], "#555555")
        planar_susceptible = bool(item.get("planar") and item["planar"].get("susceptible"))
        toppling_susceptible = bool(item.get("toppling") and item["toppling"].get("susceptible"))
        highlighted = planar_susceptible or toppling_susceptible
        for x, y in great_circle(item["dip_direction"], item["dip"]):
            ax.plot(x, y, color=color, linewidth=3 if highlighted else 2, alpha=1.0 if highlighted else 0.88)
        pole_trend, pole_plunge = pole_from_plane(item["dip_direction"], item["dip"])
        x, y = project(np.array([pole_trend]), np.array([pole_plunge]))
        marker = "D" if planar_susceptible else "s" if toppling_susceptible else "o"
        ax.scatter([x[0]], [y[0]], marker=marker, s=96 if highlighted else 62, color=color, edgecolor="white", linewidth=1.2, zorder=5)
        ax.annotate(item["label"], (x[0], y[0]), xytext=(6, 5), textcoords="offset points", fontsize=8.5, bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": color, "linewidth": 0.8})
        item["pole"] = f"{pole_trend:03.0f}/{pole_plunge:02.0f}"

    for index, result in enumerate(wedge_results, 1):
        x, y = project(np.array([result["trend"]]), np.array([result["plunge"]]))
        susceptible = bool(result["susceptible"])
        ax.scatter([x[0]], [y[0]], marker="X", s=100 if susceptible else 58, color=WEDGE if susceptible else "#818cf8", edgecolor="white", zorder=6)
        ax.annotate(f"W{index}", (x[0], y[0]), xytext=(6, -10), textcoords="offset points", fontsize=8.5, bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": WEDGE, "linewidth": 0.8})

    handles = [plt.Line2D([0], [0], color=color, marker="o", markersize=6, linewidth=2, label=kind) for kind, color in COLORS.items() if any(item["type"] == kind for item in orientations)]
    if show_zones and show_any:
        if show_planar:
            handles.append(plt.Line2D([0], [0], color=PLANAR, linewidth=8, alpha=0.35, label="Planar zone"))
        if show_wedge:
            handles.append(plt.Line2D([0], [0], color=WEDGE, linewidth=8, alpha=0.35, label="Wedge zone"))
        if show_toppling:
            handles.append(plt.Line2D([0], [0], color=TOPPLE, linewidth=8, alpha=0.35, label="Toppling zone"))
    if handles:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    return fig


def clean_report_table(dataframe):
    table = dataframe.copy()
    table = table.where(pd.notnull(table), "")
    for column in table.columns:
        table[column] = table[column].map(lambda value: str(value))
    return table


def report_table_without_columns(dataframe, columns):
    return dataframe.drop(columns=[column for column in columns if column in dataframe.columns])


def draw_report_table(ax, title, dataframe, bbox, font_size=6.1, row_height=0.031):
    table = clean_report_table(dataframe)
    if table.empty:
        table = pd.DataFrame({"Result": ["No records to report."]})
    x, y, width, height = bbox
    table_height = min(height, max(0.055, (len(table) + 1) * row_height))
    table_y = y + height - table_height
    ax.text(x, y + height + 0.012, title, fontsize=9.0, fontweight="bold", color="#202124", transform=ax.transAxes)
    rendered = ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="left", colLoc="left", bbox=[x, table_y, width, table_height])
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(font_size)
    for (row, _column), cell in rendered.get_celld().items():
        cell.set_edgecolor("#d6d3d1")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#e7e5e4")
            cell.set_text_props(weight="bold", color="#202124")
        else:
            cell.set_facecolor("#ffffff")
            cell.set_text_props(color="#202124")


def add_report_tables_page(pdf, location_id, slope_dd, slope_dip, friction, lateral, sets, planar, wedge, toppling, include_planar, include_wedge, include_toppling):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.06, 0.955, "Stereonet and Kinematic Analysis", fontsize=18, fontweight="bold", color="#202124", transform=ax.transAxes)
    ax.text(0.06, 0.925, f"Location ID: {location_id or '-'}", fontsize=11, color="#202124", transform=ax.transAxes)
    inputs = pd.DataFrame([
        {"Parameter": "Slope dip direction", "Value": f"{slope_dd:.0f} deg"},
        {"Parameter": "Slope dip", "Value": f"{slope_dip:.0f} deg"},
        {"Parameter": "Friction angle", "Value": f"{friction:.0f} deg"},
        {"Parameter": "Lateral limit", "Value": f"{lateral:.0f} deg"},
    ])
    table_font_size = 6.1
    draw_report_table(ax, "Analysis inputs", inputs, [0.06, 0.70, 0.88, 0.14], font_size=table_font_size, row_height=0.031)
    draw_report_table(ax, "Discontinuity sets", sets, [0.06, 0.52, 0.88, 0.12], font_size=table_font_size, row_height=0.026)
    planar_table = report_table_without_columns(planar, ["Reason"]) if include_planar else pd.DataFrame({"Result": ["Planar sliding analysis disabled."]})
    wedge_table = report_table_without_columns(wedge, ["Reason"]) if include_wedge else pd.DataFrame({"Result": ["Wedge sliding analysis disabled."]})
    toppling_table = report_table_without_columns(toppling, ["Reason"]) if include_toppling else pd.DataFrame({"Result": ["Toppling analysis disabled."]})
    ax.text(0.06, 0.487, "Kinematic analysis", fontsize=11, fontweight="bold", color="#202124", transform=ax.transAxes)
    draw_report_table(ax, "Planar sliding", planar_table, [0.06, 0.345, 0.88, 0.10], font_size=table_font_size, row_height=0.024)
    draw_report_table(ax, "Wedge sliding", wedge_table, [0.06, 0.21, 0.88, 0.10], font_size=table_font_size, row_height=0.024)
    draw_report_table(ax, "Toppling", toppling_table, [0.06, 0.06, 0.88, 0.10], font_size=table_font_size, row_height=0.024)
    pdf.savefig(fig)
    plt.close(fig)


def add_report_plot_page(pdf, stereonet_fig):
    image_buffer = BytesIO()
    stereonet_fig.savefig(image_buffer, format="png", dpi=220)
    image_buffer.seek(0)
    image = plt.imread(image_buffer)
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    image_ax = fig.add_axes([0.05, 0.06, 0.90, 0.88])
    image_ax.imshow(image)
    image_ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def build_pdf_report(location_id, slope_dd, slope_dip, friction, lateral, sets, stereonet_fig, summary, planar, wedge, toppling, include_planar, include_wedge, include_toppling):
    report_buffer = BytesIO()
    with PdfPages(report_buffer) as pdf:
        add_report_tables_page(pdf, location_id, slope_dd, slope_dip, friction, lateral, sets, planar, wedge, toppling, include_planar, include_wedge, include_toppling)
        add_report_plot_page(pdf, stereonet_fig)
    return report_buffer.getvalue()


def safe_name(value, fallback):
    clean_value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("_")
    return clean_value or fallback


def report_file_name(location_id):
    return f"{safe_name(location_id, 'stereonet_kinematic_analysis_report')}.pdf"


def project_file_name(location_id):
    return f"{safe_name(location_id, 'stereonet_project')}.json"


def clean_sets_for_project(sets):
    rows = sets.where(pd.notnull(sets), None).to_dict(orient="records")
    for row in rows:
        if "Plot" in row:
            row["Plot"] = bool(row["Plot"])
    return rows


def build_project_data(location_id, slope_dd, slope_dip, friction, lateral, sets, enable_planar, enable_wedge, enable_toppling, show_zones, show_summary):
    return {
        "project_version": 1,
        "location_id": location_id,
        "slope_dip_direction": float(slope_dd),
        "slope_dip": float(slope_dip),
        "friction_angle": float(friction),
        "lateral_limit": float(lateral),
        "enable_planar": bool(enable_planar),
        "enable_wedge": bool(enable_wedge),
        "enable_toppling": bool(enable_toppling),
        "show_analysis_zones": bool(show_zones),
        "show_table": bool(show_summary),
        "sets": clean_sets_for_project(sets),
    }


def normalise_project_data(data, fallback):
    project = fallback | data
    project["location_id"] = str(project.get("location_id", ""))
    project["slope_dip_direction"] = clean(project.get("slope_dip_direction"), 360, "Project slope dip direction") % 360
    project["slope_dip"] = clean(project.get("slope_dip"), 90, "Project slope dip")
    project["friction_angle"] = clean(project.get("friction_angle"), 89, "Project friction angle")
    project["lateral_limit"] = clean(project.get("lateral_limit"), 90, "Project lateral limit")
    project["enable_planar"] = bool(project.get("enable_planar", True))
    project["enable_wedge"] = bool(project.get("enable_wedge", True))
    project["enable_toppling"] = bool(project.get("enable_toppling", True))
    project["show_analysis_zones"] = bool(project.get("show_analysis_zones", True))
    project["show_table"] = bool(project.get("show_table", True))
    project["sets"] = [{k: v for k, v in row.items() if k != "Type"} for row in project.get("sets", fallback["sets"])]
    return project


def apply_project_to_session(project):
    st.session_state["location_id"] = project["location_id"]
    st.session_state["slope_dd"] = float(project["slope_dip_direction"])
    st.session_state["slope_dip"] = float(project["slope_dip"])
    st.session_state["friction"] = float(project["friction_angle"])
    st.session_state["lateral"] = float(project["lateral_limit"])
    st.session_state["enable_planar"] = bool(project["enable_planar"])
    st.session_state["enable_wedge"] = bool(project["enable_wedge"])
    st.session_state["enable_toppling"] = bool(project["enable_toppling"])
    st.session_state["show_zones"] = bool(project["show_analysis_zones"])
    st.session_state["show_summary"] = bool(project["show_table"])
    st.session_state["project_sets"] = project["sets"]
    st.session_state["project_load_count"] = int(st.session_state.get("project_load_count", 0)) + 1


def initialise_session(defaults):
    values = {
        "location_id": str(defaults.get("location_id", "")),
        "slope_dd": float(defaults["slope_dip_direction"]),
        "slope_dip": float(defaults["slope_dip"]),
        "friction": float(defaults["friction_angle"]),
        "lateral": float(defaults["lateral_limit"]),
        "enable_planar": bool(defaults.get("enable_planar", True)),
        "enable_wedge": bool(defaults.get("enable_wedge", True)),
        "enable_toppling": bool(defaults.get("enable_toppling", True)),
        "show_zones": bool(defaults.get("show_analysis_zones", True)),
        "show_summary": bool(defaults.get("show_table", True)),
    }
    for key, value in values.items():
        st.session_state.setdefault(key, value)


def main():
    st.set_page_config(page_title="Stereonet and Kinematic Analysis", layout="wide")
    st.title("Stereonet and Kinematic Analysis")
    defaults = load_defaults()
    initialise_session(defaults)

    with st.sidebar:
        st.header("Project files")
        upload = st.file_uploader("Load project JSON", type=["json"])
        if upload is not None:
            payload = upload.getvalue()
            signature = f"{upload.name}:{len(payload)}"
            if st.session_state.get("loaded_project_signature") != signature:
                try:
                    project = normalise_project_data(json.loads(payload.decode("utf-8")), defaults)
                    apply_project_to_session(project)
                    st.session_state["loaded_project_signature"] = signature
                    st.success(f"Loaded project: {upload.name}")
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    st.error(f"Could not load project file: {exc}")

    initial_sets = st.session_state.get("project_sets", defaults["sets"])
    editor_key = f"sets_{st.session_state.get('project_load_count', 0)}"
    location_id = st.text_input("Location ID:", placeholder="Enter location ID", key="location_id")

    with st.sidebar:
        st.header("Slope")
        slope_dd = st.number_input("Slope dip direction", 0.0, 360.0, 1.0, key="slope_dd")
        slope_dip = st.number_input("Slope dip", 0.0, 90.0, 1.0, key="slope_dip")
        st.header("Kinematic analysis")
        enable_planar = st.toggle("Planar sliding", key="enable_planar")
        enable_wedge = st.toggle("Wedge sliding", key="enable_wedge")
        enable_toppling = st.toggle("Toppling", key="enable_toppling")
        show_zones = st.toggle("Show analysis zones", key="show_zones")
        friction = st.number_input("Friction angle", 0.0, 89.0, 1.0, key="friction")
        lateral = st.number_input("Lateral limit", 0.0, 90.0, 1.0, key="lateral")
        show_summary = st.toggle("Show orientation summary", key="show_summary")

    st.subheader("Discontinuity Sets")
    sets = st.data_editor(
        pd.DataFrame(initial_sets),
        key=editor_key,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Plot": st.column_config.CheckboxColumn(default=True, width="small"),
            "Name": st.column_config.TextColumn(required=False, width="medium"),
            "Dip direction": st.column_config.NumberColumn(min_value=0.0, max_value=360.0, step=1.0, width="medium"),
            "Dip": st.column_config.NumberColumn(min_value=0.0, max_value=90.0, step=1.0, width="medium"),
        },
    )

    project_json = json.dumps(
        build_project_data(location_id, slope_dd, slope_dip, friction, lateral, sets, enable_planar, enable_wedge, enable_toppling, show_zones, show_summary),
        indent=2,
    )
    with st.sidebar:
        st.download_button("Save project JSON", project_json, project_file_name(location_id), "application/json")

    try:
        orientations = build_orientations(slope_dd, slope_dip, sets)
        planar_df = analyse_planar(orientations, slope_dd, slope_dip, friction, lateral) if enable_planar else pd.DataFrame()
        wedge_df, wedge_results = analyse_wedge(orientations, slope_dd, slope_dip, friction, lateral) if enable_wedge else (pd.DataFrame(), [])
        toppling_df = analyse_toppling(orientations, slope_dd, slope_dip, friction, lateral) if enable_toppling else pd.DataFrame()
        fig = plot_stereonet(orientations, wedge_results, enable_planar, enable_wedge, enable_toppling, show_zones, slope_dd, slope_dip, friction, lateral)
        st.pyplot(fig, clear_figure=False)

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight")
        st.download_button("Download PNG", buffer.getvalue(), "geotechnical_stereonet.png", "image/png")

        summary = pd.DataFrame([
            {"Type": item["type"], "Name": item["label"], "Dip direction": f"{item['dip_direction']:03.0f}", "Dip": f"{item['dip']:02.0f}", "Pole trend/plunge": item["pole"]}
            for item in orientations
        ])
        if show_summary:
            st.subheader("Orientation summary")
            st.dataframe(summary, hide_index=True, use_container_width=True)
        if enable_planar:
            st.subheader("Planar sliding analysis")
            if planar_df.empty:
                st.info("Add discontinuity sets to analyse planar sliding.")
            else:
                st.metric("Potential planar sliding planes", int((planar_df["Planar sliding"] == "Potential").sum()))
                st.dataframe(planar_df, hide_index=True, use_container_width=True)
        if enable_wedge:
            st.subheader("Wedge sliding analysis")
            if wedge_df.empty:
                st.info("Add at least two discontinuity sets to analyse wedge sliding.")
            else:
                st.metric("Potential wedge intersections", int((wedge_df["Wedge sliding"] == "Potential").sum()))
                st.dataframe(wedge_df, hide_index=True, use_container_width=True)
        if enable_toppling:
            st.subheader("Toppling analysis")
            if toppling_df.empty:
                st.info("Add discontinuity sets to analyse toppling.")
            else:
                st.metric("Potential toppling planes", int((toppling_df["Toppling"] == "Potential").sum()))
                st.dataframe(toppling_df, hide_index=True, use_container_width=True)

        st.subheader("Export report")
        if st.button("Generate PDF report", type="primary"):
            pdf_bytes = build_pdf_report(location_id, slope_dd, slope_dip, friction, lateral, sets, fig, summary, planar_df, wedge_df, toppling_df, enable_planar, enable_wedge, enable_toppling)
            file_name = report_file_name(location_id)
            REPORTS_DIR.mkdir(exist_ok=True)
            report_path = REPORTS_DIR / file_name
            report_path.write_bytes(pdf_bytes)
            st.session_state["report_pdf_bytes"] = pdf_bytes
            st.session_state["report_pdf_name"] = file_name
            st.session_state["report_pdf_path"] = str(report_path)
        if "report_pdf_bytes" in st.session_state:
            st.success(f"PDF report generated: {st.session_state['report_pdf_path']}")
            st.download_button("Download PDF report", st.session_state["report_pdf_bytes"], st.session_state["report_pdf_name"], "application/pdf")
    except ValueError as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
