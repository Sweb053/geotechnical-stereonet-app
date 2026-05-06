from __future__ import annotations

import json
from io import BytesIO
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

COLORS = {"Slope": "#d1495b", "Discontinuity": "#00798c"}
PLANAR = "#7c2d12"
WEDGE = "#4f46e5"
TOPPLE = "#9333ea"
DEFAULTS_PATH = Path(__file__).with_name("defaults.json")


def load_defaults() -> dict:
    fallback = {"slope_dip_direction": 135.0, "slope_dip": 45.0, "friction_angle": 30.0, "lateral_limit": 20.0, "sets": [{"Plot": True, "Name": "S1", "Dip direction": 120.0, "Dip": 35.0}, {"Plot": True, "Name": "S2", "Dip direction": 95.0, "Dip": 70.0}, {"Plot": True, "Name": "S3", "Dip direction": 210.0, "Dip": 55.0}]}
    try:
        loaded = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    defaults = fallback | loaded
    defaults["sets"] = [{k: v for k, v in row.items() if k != "Type"} for row in defaults.get("sets", fallback["sets"])]
    return defaults


def save_defaults(slope_dd, slope_dip, friction, lateral, sets) -> None:
    data = {"slope_dip_direction": float(slope_dd), "slope_dip": float(slope_dip), "friction_angle": float(friction), "lateral_limit": float(lateral), "sets": sets.where(pd.notnull(sets), None).to_dict(orient="records")}
    DEFAULTS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


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
    tr = np.deg2rad(trend)
    pl = np.deg2rad(plunge)
    return np.array([np.sin(tr) * np.cos(pl), np.cos(tr) * np.cos(pl), -np.sin(pl)])


def vector_components(trend, plunge):
    tr = np.deg2rad(trend)
    pl = np.deg2rad(plunge)
    return np.sin(tr) * np.cos(pl), np.cos(tr) * np.cos(pl), -np.sin(pl)


def trend_plunge_from_vectors(vectors):
    vectors = np.asarray(vectors, dtype=float)
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    vectors = np.where(vectors[:, 2:3] > 0, -vectors, vectors)
    horizontal = np.hypot(vectors[:, 0], vectors[:, 1])
    trend = (np.rad2deg(np.arctan2(vectors[:, 0], vectors[:, 1])) + 360.0) % 360.0
    plunge = np.rad2deg(np.arctan2(-vectors[:, 2], horizontal))
    return trend, plunge


def project(trend, plunge):
    tr = np.deg2rad(trend)
    radius = np.tan(np.deg2rad((90.0 - plunge) / 2.0))
    return radius * np.sin(tr), radius * np.cos(tr)


def plane_normal(dip_direction, dip):
    trend, plunge = pole_from_plane(dip_direction, dip)
    return vector_from_trend_plunge(trend, plunge)


def plane_daylights(dip_direction, dip, slope_dd, slope_dip):
    return float(np.dot(plane_normal(slope_dd, slope_dip), vector_from_trend_plunge(dip_direction, dip))) < -1e-10


def line_daylights(trend, plunge, slope_dd, slope_dip):
    return float(np.dot(plane_normal(slope_dd, slope_dip), vector_from_trend_plunge(trend, plunge))) < -1e-10


def intersection(first, second):
    line = np.cross(plane_normal(first["dip_direction"], first["dip"]), plane_normal(second["dip_direction"], second["dip"]))
    if np.linalg.norm(line) < 1e-9:
        return None
    trend, plunge = trend_plunge_from_vectors(np.array([line]))
    return float(trend[0]), float(plunge[0])


def great_circle(dip_direction, dip, samples=361):
    normal = plane_normal(dip_direction, dip)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, ref))) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    axis_a = np.cross(normal, ref)
    axis_a = axis_a / np.linalg.norm(axis_a)
    axis_b = np.cross(normal, axis_a)
    axis_b = axis_b / np.linalg.norm(axis_b)
    angles = np.linspace(0, 2 * np.pi, samples)
    vectors = np.cos(angles)[:, None] * axis_a + np.sin(angles)[:, None] * axis_b
    trend, plunge = trend_plunge_from_vectors(vectors)
    x, y = project(trend, plunge)
    breaks = np.where(np.hypot(np.diff(x), np.diff(y)) > 0.35)[0] + 1
    return [(xp, yp) for xp, yp in zip(np.split(x, breaks), np.split(y, breaks)) if len(xp) > 1]


def build_orientations(slope_dd, slope_dip, sets):
    orientations = [{"type": "Slope", "label": "Slope", "dip_direction": clean(slope_dd, 360, "Slope dip direction") % 360, "dip": clean(slope_dip, 90, "Slope dip")}]
    for index, row in sets.iterrows():
        if not bool(row.get("Plot", True)):
            continue
        label = str(row.get("Name") or f"S{index + 1}")
        orientations.append({"type": "Discontinuity", "label": label, "dip_direction": clean(row.get("Dip direction"), 360, f"{label} dip direction") % 360, "dip": clean(row.get("Dip"), 90, f"{label} dip")})
    return orientations


def planar_result(item, slope_dd, slope_dip, friction, lateral):
    align = angle_diff(item["dip_direction"], slope_dd)
    result = {"alignment": align, "daylights": plane_daylights(item["dip_direction"], item["dip"], slope_dd, slope_dip), "exceeds_friction": item["dip"] > friction, "aligned": align <= lateral}
    result["susceptible"] = result["aligned"] and result["exceeds_friction"] and result["daylights"]
    return result


def toppling_result(item, slope_dd, slope_dip, friction, lateral):
    threshold = max(0.0, 90.0 - slope_dip + friction)
    align = angle_diff(item["dip_direction"], (slope_dd + 180.0) % 360.0)
    result = {"alignment": align, "threshold": threshold, "into_slope": align <= lateral, "steep": item["dip"] > threshold}
    result["susceptible"] = result["into_slope"] and result["steep"]
    return result


def analyse_planar(orientations, slope_dd, slope_dip, friction, lateral):
    rows = []
    for item in orientations:
        if item["type"] == "Slope":
            continue
        result = planar_result(item, slope_dd, slope_dip, friction, lateral)
        item["planar"] = result
        rows.append({"Name": item["label"], "Dip direction": f"{item['dip_direction']:03.0f}", "Dip": f"{item['dip']:02.0f}", "Alignment": f"{result['alignment']:.1f}", "Daylights": "Yes" if result["daylights"] else "No", "Dip > friction": "Yes" if result["exceeds_friction"] else "No", "Planar sliding": "Potential" if result["susceptible"] else "No"})
    return pd.DataFrame(rows)


def analyse_toppling(orientations, slope_dd, slope_dip, friction, lateral):
    rows = []
    for item in orientations:
        if item["type"] == "Slope":
            continue
        result = toppling_result(item, slope_dd, slope_dip, friction, lateral)
        item["toppling"] = result
        rows.append({"Name": item["label"], "Dip direction": f"{item['dip_direction']:03.0f}", "Dip": f"{item['dip']:02.0f}", "Into-slope alignment": f"{result['alignment']:.1f}", "Threshold dip": f"{result['threshold']:.1f}", "Dips into slope": "Yes" if result["into_slope"] else "No", "Toppling": "Potential" if result["susceptible"] else "No"})
    return pd.DataFrame(rows)


def analyse_wedge(orientations, slope_dd, slope_dip, friction, lateral):
    discontinuities = [item for item in orientations if item["type"] != "Slope"]
    rows, results = [], []
    for first, second in combinations(discontinuities, 2):
        line = intersection(first, second)
        if line is None:
            continue
        trend, plunge = line
        align = angle_diff(trend, slope_dd)
        result = {"first": first["label"], "second": second["label"], "trend": trend, "plunge": plunge, "alignment": align, "daylights": line_daylights(trend, plunge, slope_dd, slope_dip), "exceeds_friction": plunge > friction, "aligned": align <= lateral}
        result["susceptible"] = result["aligned"] and result["exceeds_friction"] and result["daylights"]
        results.append(result)
        rows.append({"Planes": f"{first['label']} + {second['label']}", "Trend": f"{trend:03.0f}", "Plunge": f"{plunge:02.0f}", "Alignment": f"{align:.1f}", "Daylights": "Yes" if result["daylights"] else "No", "Plunge > friction": "Yes" if result["exceeds_friction"] else "No", "Wedge sliding": "Potential" if result["susceptible"] else "No"})
    return pd.DataFrame(rows), results


def add_grid(ax):
    ax.add_patch(plt.Circle((0, 0), 1, edgecolor="#202124", facecolor="none", linewidth=1.4))
    theta = np.linspace(0, 2 * np.pi, 361)
    for angle in range(0, 360, 10):
        rad = np.deg2rad(angle)
        ax.plot([0, np.sin(rad)], [0, np.cos(rad)], color="#b7c0c9" if angle % 30 == 0 else "#d7dde2", linewidth=1.1 if angle % 30 == 0 else 0.8, zorder=0)
    for radius in np.tan(np.deg2rad((90.0 - np.arange(10, 90, 10)) / 2.0)):
        ax.plot(radius * np.sin(theta), radius * np.cos(theta), color="#d7dde2", linewidth=0.8, zorder=0)
    for text, x, y in [("N", 0, 1.08), ("E", 1.08, 0), ("S", 0, -1.08), ("W", -1.08, 0)]:
        ax.text(x, y, text, ha="center", va="center", fontsize=11, weight="bold")
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
            ax.contourf(xx, yy, np.ma.masked_where(~mask, mask.astype(float)), levels=[0.5, 1.5], colors=[color], alpha=0.18, zorder=0.45)

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


def plot_stereonet(orientations, wedge_results, show_planar, show_wedge, show_toppling, show_zones, slope_dd, slope_dip, friction, lateral):
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=140)
    add_grid(ax)
    if show_zones and (show_planar or show_wedge or show_toppling):
        draw_zones(ax, show_planar, show_wedge, show_toppling, slope_dd, slope_dip, friction, lateral)
    if show_planar or show_wedge or show_toppling:
        theta = np.linspace(0, 2 * np.pi, 361)
        if show_planar or show_toppling:
            r = np.tan(np.deg2rad(friction / 2.0))
            ax.plot(r * np.sin(theta), r * np.cos(theta), color="#9a3412", linewidth=1.4, linestyle=":")
        if show_wedge:
            r = np.tan(np.deg2rad((90.0 - friction) / 2.0))
            ax.plot(r * np.sin(theta), r * np.cos(theta), color=WEDGE, linewidth=1.4, linestyle=":")
        for x, y in great_circle(slope_dd, slope_dip):
            ax.plot(x, y, color="#57534e", linewidth=1.5, linestyle="-.")

    for item in orientations:
        color = COLORS.get(item["type"], "#555")
        highlighted = bool(item.get("planar", {}).get("susceptible") or item.get("toppling", {}).get("susceptible"))
        for x, y in great_circle(item["dip_direction"], item["dip"]):
            ax.plot(x, y, color=color, linewidth=3 if highlighted else 2, alpha=1.0 if highlighted else 0.88)
        ptrend, pplunge = pole_from_plane(item["dip_direction"], item["dip"])
        x, y = project(np.array([ptrend]), np.array([pplunge]))
        ax.scatter([x[0]], [y[0]], s=86 if highlighted else 58, color=color, edgecolor="white", zorder=5)
        ax.annotate(item["label"], (x[0], y[0]), xytext=(6, 5), textcoords="offset points", fontsize=8.5, bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": color, "linewidth": 0.8})
        item["pole"] = f"{ptrend:03.0f}/{pplunge:02.0f}"

    for i, result in enumerate(wedge_results, 1):
        x, y = project(np.array([result["trend"]]), np.array([result["plunge"]]))
        ax.scatter([x[0]], [y[0]], marker="X", s=100 if result["susceptible"] else 58, color=WEDGE if result["susceptible"] else "#818cf8", edgecolor="white", zorder=6)
        ax.annotate(f"W{i}", (x[0], y[0]), xytext=(6, -10), textcoords="offset points", fontsize=8.5, bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": WEDGE, "linewidth": 0.8})

    handles = [plt.Line2D([0], [0], color=c, marker="o", linewidth=2, label=k) for k, c in COLORS.items() if any(o["type"] == k for o in orientations)]
    if show_zones:
        if show_planar:
            handles.append(plt.Line2D([0], [0], color=PLANAR, linewidth=8, alpha=0.35, label="Planar zone"))
        if show_wedge:
            handles.append(plt.Line2D([0], [0], color=WEDGE, linewidth=8, alpha=0.35, label="Wedge zone"))
        if show_toppling:
            handles.append(plt.Line2D([0], [0], color=TOPPLE, linewidth=8, alpha=0.35, label="Toppling zone"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    return fig


def main():
    st.set_page_config(page_title="Stereonet and Kinematic Analysis", layout="wide")
    st.title("Stereonet and Kinematic Analysis")
    defaults = load_defaults()

    with st.sidebar:
        st.header("Slope")
        slope_dd = st.number_input("Slope dip direction", 0.0, 360.0, float(defaults["slope_dip_direction"]), 1.0, key="slope_dd")
        slope_dip = st.number_input("Slope dip", 0.0, 90.0, float(defaults["slope_dip"]), 1.0, key="slope_dip")
        st.header("Kinematic analysis")
        enable_planar = st.toggle("Planar sliding", value=True)
        enable_wedge = st.toggle("Wedge sliding", value=True)
        enable_toppling = st.toggle("Toppling", value=True)
        show_zones = st.toggle("Show analysis zones", value=True)
        friction = st.number_input("Friction angle", 0.0, 89.0, float(defaults["friction_angle"]), 1.0, key="friction")
        lateral = st.number_input("Lateral limit", 0.0, 90.0, float(defaults["lateral_limit"]), 1.0, key="lateral")
        show_summary = st.toggle("Show orientation summary", value=True)

    st.subheader("Discontinuity Sets")
    sets = st.data_editor(pd.DataFrame(defaults["sets"]), key="sets", num_rows="dynamic", use_container_width=True, hide_index=True, column_config={"Plot": st.column_config.CheckboxColumn(default=True, width="small"), "Name": st.column_config.TextColumn(required=False, width="medium"), "Dip direction": st.column_config.NumberColumn(min_value=0.0, max_value=360.0, step=1.0, width="medium"), "Dip": st.column_config.NumberColumn(min_value=0.0, max_value=90.0, step=1.0, width="medium")})

    if st.button("Save current inputs as defaults"):
        save_defaults(slope_dd, slope_dip, friction, lateral, sets)
        st.success("Saved defaults for this deployment session.")

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

        if show_summary:
            st.subheader("Orientation summary")
            st.dataframe(pd.DataFrame([{"Type": o["type"], "Name": o["label"], "Dip direction": f"{o['dip_direction']:03.0f}", "Dip": f"{o['dip']:02.0f}", "Pole trend/plunge": o["pole"]} for o in orientations]), hide_index=True, use_container_width=True)
        if enable_planar:
            st.subheader("Planar sliding analysis")
            st.metric("Potential planar sliding planes", int((planar_df.get("Planar sliding", pd.Series(dtype=str)) == "Potential").sum()))
            st.dataframe(planar_df, hide_index=True, use_container_width=True)
        if enable_wedge:
            st.subheader("Wedge sliding analysis")
            st.metric("Potential wedge intersections", int((wedge_df.get("Wedge sliding", pd.Series(dtype=str)) == "Potential").sum()))
            st.dataframe(wedge_df, hide_index=True, use_container_width=True)
        if enable_toppling:
            st.subheader("Toppling analysis")
            st.metric("Potential toppling planes", int((toppling_df.get("Toppling", pd.Series(dtype=str)) == "Potential").sum()))
            st.dataframe(toppling_df, hide_index=True, use_container_width=True)
        with st.expander("Analysis theory and sources"):
            st.markdown("""
This is a screening-level kinematic analysis. It checks orientation feasibility on a stereonet; it does not calculate factor of safety, block size, persistence, groundwater pressure, cohesion, seismic loading, or release-plane geometry.

Sources: Rocscience Dips kinematic analysis documentation, Markland (1972), and Wyllie & Mah, Rock Slope Engineering.
""")
    except ValueError as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
