from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import streamlit as st


COLORS = {
    "Slope": "#d1495b",
    "Discontinuity": "#00798c",
}


KINEMATIC_HIGHLIGHT = "#7c2d12"
WEDGE_HIGHLIGHT = "#4f46e5"
TOPPLING_HIGHLIGHT = "#9333ea"
BLOCK_TOPPLING_HIGHLIGHT = "#f59e0b"
DEFAULTS_PATH = Path(__file__).with_name("defaults.json")
REPORTS_DIR = Path(__file__).with_name("reports")


def clean_orientation(value: float, limit: float, name: str) -> float:
    if pd.isna(value):
        raise ValueError(f"{name} is missing")
    value = float(value)
    if not 0 <= value <= limit:
        raise ValueError(f"{name} must be between 0 and {limit:g}")
    return value


def load_defaults() -> dict[str, object]:
    fallback = {
        "slope_dip_direction": 135.0,
        "slope_dip": 45.0,
        "friction_angle": 30.0,
        "lateral_limit": 20.0,
        "sets": [
            {"Plot": True, "Name": "S1", "Dip direction": 120.0, "Dip": 35.0},
            {"Plot": True, "Name": "S2", "Dip direction": 95.0, "Dip": 70.0},
            {"Plot": True, "Name": "S3", "Dip direction": 210.0, "Dip": 55.0},
        ],
    }
    if not DEFAULTS_PATH.exists():
        return fallback

    try:
        loaded = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
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
        defaults["sets"] = [{key: value for key, value in row.items() if key != "Type"} for row in defaults["sets"]]
    return defaults


def pole_from_plane(dip_direction: float, dip: float) -> tuple[float, float]:
    trend = (dip_direction + 180.0) % 360.0
    plunge = 90.0 - dip
    return trend, plunge


def vector_from_trend_plunge(trend: float, plunge: float) -> np.ndarray:
    trend_rad = np.deg2rad(trend)
    plunge_rad = np.deg2rad(plunge)
    return np.array(
        [
            np.sin(trend_rad) * np.cos(plunge_rad),
            np.cos(trend_rad) * np.cos(plunge_rad),
            -np.sin(plunge_rad),
        ]
    )


def project_lower_hemisphere(trend: np.ndarray, plunge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trend_rad = np.deg2rad(trend)
    radius = np.tan(np.deg2rad((90.0 - plunge) / 2.0))
    return radius * np.sin(trend_rad), radius * np.cos(trend_rad)


def trend_plunge_from_vector(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=float)
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    vectors = np.where(vectors[:, 2:3] > 0, -vectors, vectors)
    horizontal = np.hypot(vectors[:, 0], vectors[:, 1])
    trend = (np.rad2deg(np.arctan2(vectors[:, 0], vectors[:, 1])) + 360.0) % 360.0
    plunge = np.rad2deg(np.arctan2(-vectors[:, 2], horizontal))
    return trend, plunge


def great_circle_xy(dip_direction: float, dip: float, samples: int = 361) -> list[tuple[np.ndarray, np.ndarray]]:
    pole_trend, pole_plunge = pole_from_plane(dip_direction, dip)
    normal = vector_from_trend_plunge(pole_trend, pole_plunge)

    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, reference))) > 0.95:
        reference = np.array([1.0, 0.0, 0.0])

    axis_a = np.cross(normal, reference)
    axis_a = axis_a / np.linalg.norm(axis_a)
    axis_b = np.cross(normal, axis_a)
    axis_b = axis_b / np.linalg.norm(axis_b)

    angles = np.linspace(0.0, 2.0 * np.pi, samples)
    vectors = np.cos(angles)[:, None] * axis_a + np.sin(angles)[:, None] * axis_b
    trend, plunge = trend_plunge_from_vector(vectors)
    x, y = project_lower_hemisphere(trend, plunge)

    breaks = np.where(np.hypot(np.diff(x), np.diff(y)) > 0.35)[0] + 1
    x_parts = np.split(x, breaks)
    y_parts = np.split(y, breaks)
    return [(xp, yp) for xp, yp in zip(x_parts, y_parts) if len(xp) > 1]


def pole_xy(dip_direction: float, dip: float) -> tuple[float, float, float, float]:
    trend, plunge = pole_from_plane(dip_direction, dip)
    x, y = project_lower_hemisphere(np.array([trend]), np.array([plunge]))
    return float(x[0]), float(y[0]), trend, plunge


def angular_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def angular_difference_array(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def plane_normal(dip_direction: float, dip: float) -> np.ndarray:
    trend, plunge = pole_from_plane(dip_direction, dip)
    return vector_from_trend_plunge(trend, plunge)


def intersection_line(
    first: dict[str, object],
    second: dict[str, object],
) -> tuple[float, float] | None:
    normal_a = plane_normal(float(first["dip_direction"]), float(first["dip"]))
    normal_b = plane_normal(float(second["dip_direction"]), float(second["dip"]))
    line = np.cross(normal_a, normal_b)
    if np.linalg.norm(line) < 1e-9:
        return None
    trend, plunge = trend_plunge_from_vector(np.array([line]))
    return float(trend[0]), float(plunge[0])


def dip_vector_from_plane(dip_direction: float, dip: float) -> np.ndarray:
    return vector_from_trend_plunge(dip_direction, dip)


def plane_daylights_from_slope(
    dip_direction: float,
    dip: float,
    slope_dip_direction: float,
    slope_dip: float,
) -> bool:
    slope_normal = plane_normal(slope_dip_direction, slope_dip)
    dip_vector = dip_vector_from_plane(dip_direction, dip)
    return float(np.dot(slope_normal, dip_vector)) < -1e-10


def line_daylights_from_slope(
    trend: float,
    plunge: float,
    slope_dip_direction: float,
    slope_dip: float,
) -> bool:
    slope_normal = plane_normal(slope_dip_direction, slope_dip)
    line_vector = vector_from_trend_plunge(trend, plunge)
    return float(np.dot(slope_normal, line_vector)) < -1e-10


def planar_sliding_result(
    item: dict[str, object],
    slope_dip_direction: float,
    slope_dip: float,
    friction_angle: float,
    lateral_limit: float,
) -> dict[str, object]:
    dip_direction = float(item["dip_direction"])
    dip = float(item["dip"])
    alignment = angular_difference(dip_direction, slope_dip_direction)
    dips_out_of_slope = alignment <= lateral_limit
    exceeds_friction = dip > friction_angle
    daylights = plane_daylights_from_slope(dip_direction, dip, slope_dip_direction, slope_dip)
    susceptible = dips_out_of_slope and exceeds_friction and daylights

    reasons = []
    if not dips_out_of_slope:
        reasons.append("outside lateral limit")
    if not exceeds_friction:
        reasons.append("dip <= friction")
    if not daylights:
        reasons.append("does not daylight")

    return {
        "alignment": alignment,
        "dips_out_of_slope": dips_out_of_slope,
        "exceeds_friction": exceeds_friction,
        "daylights": daylights,
        "susceptible": susceptible,
        "reason": "Meets planar sliding criteria" if susceptible else ", ".join(reasons),
    }


def annotate_planar_sliding(
    orientations: list[dict[str, object]],
    slope_dip_direction: float,
    slope_dip: float,
    friction_angle: float,
    lateral_limit: float,
) -> pd.DataFrame:
    rows = []
    for item in orientations:
        if item["type"] == "Slope":
            item["planar"] = None
            continue

        result = planar_sliding_result(item, slope_dip_direction, slope_dip, friction_angle, lateral_limit)
        item["planar"] = result
        rows.append(
            {
                "Type": item["type"],
                "Name": item["label"],
                "Dip direction": f"{float(item['dip_direction']):03.0f}",
                "Dip": f"{float(item['dip']):02.0f}",
                "Alignment": f"{result['alignment']:.0f}",
                "Daylights": "Yes" if result["daylights"] else "No",
                "Dip > friction": "Yes" if result["exceeds_friction"] else "No",
                "Planar sliding": "Potential" if result["susceptible"] else "No",
                "Reason": result["reason"],
            }
        )
    return pd.DataFrame(rows)


def wedge_sliding_result(
    first: dict[str, object],
    second: dict[str, object],
    slope_dip_direction: float,
    slope_dip: float,
    friction_angle: float,
    lateral_limit: float,
) -> dict[str, object] | None:
    line = intersection_line(first, second)
    if line is None:
        return None

    trend, plunge = line
    alignment = angular_difference(trend, slope_dip_direction)
    trends_out_of_slope = alignment <= lateral_limit
    exceeds_friction = plunge > friction_angle
    daylights = line_daylights_from_slope(trend, plunge, slope_dip_direction, slope_dip)
    susceptible = trends_out_of_slope and exceeds_friction and daylights

    reasons = []
    if not trends_out_of_slope:
        reasons.append("outside lateral limit")
    if not exceeds_friction:
        reasons.append("plunge <= friction")
    if not daylights:
        reasons.append("does not daylight")

    return {
        "first": first["label"],
        "second": second["label"],
        "trend": trend,
        "plunge": plunge,
        "alignment": alignment,
        "trends_out_of_slope": trends_out_of_slope,
        "exceeds_friction": exceeds_friction,
        "daylights": daylights,
        "susceptible": susceptible,
        "reason": "Meets wedge sliding criteria" if susceptible else ", ".join(reasons),
    }


def analyse_wedge_sliding(
    orientations: list[dict[str, object]],
    slope_dip_direction: float,
    slope_dip: float,
    friction_angle: float,
    lateral_limit: float,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    discontinuities = [item for item in orientations if item["type"] != "Slope"]
    results = []
    rows = []

    for first_index, first in enumerate(discontinuities):
        for second in discontinuities[first_index + 1 :]:
            result = wedge_sliding_result(first, second, slope_dip_direction, slope_dip, friction_angle, lateral_limit)
            if result is None:
                continue
            results.append(result)
            rows.append(
                {
                    "Planes": f"{result['first']} + {result['second']}",
                    "Trend": f"{result['trend']:03.0f}",
                    "Plunge": f"{result['plunge']:02.0f}",
                    "Alignment": f"{result['alignment']:.0f}",
                    "Daylights": "Yes" if result["daylights"] else "No",
                    "Plunge > friction": "Yes" if result["exceeds_friction"] else "No",
                    "Wedge sliding": "Potential" if result["susceptible"] else "No",
                    "Reason": result["reason"],
                }
            )

    return pd.DataFrame(rows), results


def toppling_result(
    item: dict[str, object],
    slope_dip_direction: float,
    slope_dip: float,
    friction_angle: float,
    lateral_limit: float,
) -> dict[str, object]:
    dip_direction = float(item["dip_direction"])
    dip = float(item["dip"])
    opposite_slope = (slope_dip_direction + 180.0) % 360.0
    alignment = angular_difference(dip_direction, opposite_slope)
    dips_into_slope = alignment <= lateral_limit
    threshold = max(0.0, 90.0 - slope_dip + friction_angle)
    steep_enough = dip > threshold
    susceptible = dips_into_slope and steep_enough

    reasons = []
    if not dips_into_slope:
        reasons.append("not dipping into slope")
    if not steep_enough:
        reasons.append("dip below toppling threshold")

    return {
        "alignment": alignment,
        "threshold": threshold,
        "dips_into_slope": dips_into_slope,
        "steep_enough": steep_enough,
        "susceptible": susceptible,
        "reason": "Meets toppling criteria" if susceptible else ", ".join(reasons),
    }


def annotate_toppling(
    orientations: list[dict[str, object]],
    slope_dip_direction: float,
    slope_dip: float,
    friction_angle: float,
    lateral_limit: float,
) -> pd.DataFrame:
    rows = []
    for item in orientations:
        if item["type"] == "Slope":
            item["toppling"] = None
            continue

        result = toppling_result(item, slope_dip_direction, slope_dip, friction_angle, lateral_limit)
        item["toppling"] = result
        rows.append(
            {
                "Type": item["type"],
                "Name": item["label"],
                "Dip direction": f"{float(item['dip_direction']):03.0f}",
                "Dip": f"{float(item['dip']):02.0f}",
                "Into-slope alignment": f"{result['alignment']:.0f}",
                "Threshold dip": f"{result['threshold']:.0f}",
                "Dips into slope": "Yes" if result["dips_into_slope"] else "No",
                "Toppling": "Potential" if result["susceptible"] else "No",
                "Reason": result["reason"],
            }
        )
    return pd.DataFrame(rows)


def direct_toppling_zone(
    trend: float,
    plunge: float,
    slope_dip_direction: float,
    slope_dip: float,
    friction_angle: float,
    lateral_limit: float,
) -> str:
    into_slope = (slope_dip_direction + 180.0) % 360.0
    alignment = angular_difference(trend, into_slope)
    inside_lateral_limits = alignment <= lateral_limit
    inside_friction_cone = plunge >= max(0.0, 90.0 - friction_angle)
    inside_slope_angle_limit = plunge >= max(0.0, 90.0 - slope_dip)

    if inside_lateral_limits and inside_slope_angle_limit:
        return "Zone 2" if inside_friction_cone else "Zone 1"
    if not inside_lateral_limits and inside_friction_cone:
        return "Zone 3"
    return "Outside"


def release_plane_result(
    item: dict[str, object],
    slope_dip_direction: float,
    slope_dip: float,
    friction_angle: float,
    lateral_limit: float,
) -> dict[str, object]:
    pole_trend, pole_plunge = pole_from_plane(float(item["dip_direction"]), float(item["dip"]))
    zone = direct_toppling_zone(pole_trend, pole_plunge, slope_dip_direction, slope_dip, friction_angle, lateral_limit)
    if zone == "Zone 1":
        release_mode = "Sliding release"
    elif zone == "Zone 2":
        release_mode = "Base release"
    elif zone == "Zone 3":
        release_mode = "Oblique base release"
    else:
        release_mode = "Outside release zone"

    return {
        "label": item["label"],
        "pole_trend": pole_trend,
        "pole_plunge": pole_plunge,
        "zone": zone,
        "release_mode": release_mode,
        "valid": zone != "Outside",
    }


def block_toppling_result(
    first: dict[str, object],
    second: dict[str, object],
    discontinuities: list[dict[str, object]],
    slope_dip_direction: float,
    slope_dip: float,
    friction_angle: float,
    lateral_limit: float,
) -> dict[str, object] | None:
    line = intersection_line(first, second)
    if line is None:
        return None

    trend, plunge = line
    alignment = angular_difference(trend, (slope_dip_direction + 180.0) % 360.0)
    intersection_zone = direct_toppling_zone(trend, plunge, slope_dip_direction, slope_dip, friction_angle, lateral_limit)
    critical_intersection = intersection_zone != "Outside"

    release_candidates = [
        release_plane_result(item, slope_dip_direction, slope_dip, friction_angle, lateral_limit)
        for item in discontinuities
        if item["label"] not in {first["label"], second["label"]}
    ]
    valid_release_planes = [candidate for candidate in release_candidates if candidate["valid"]]
    susceptible = critical_intersection and bool(valid_release_planes)

    reasons = []
    if not critical_intersection:
        reasons.append("intersection outside block toppling zone")
    if not valid_release_planes:
        reasons.append("no geometrically valid release/base plane")

    return {
        "first": first["label"],
        "second": second["label"],
        "trend": trend,
        "plunge": plunge,
        "alignment": alignment,
        "intersection_zone": intersection_zone,
        "release_planes": valid_release_planes,
        "critical_intersection": critical_intersection,
        "susceptible": susceptible,
        "reason": "Meets block toppling screening criteria" if susceptible else ", ".join(reasons),
    }


def analyse_block_toppling(
    orientations: list[dict[str, object]],
    slope_dip_direction: float,
    slope_dip: float,
    friction_angle: float,
    lateral_limit: float,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    discontinuities = [item for item in orientations if item["type"] != "Slope"]
    results = []
    rows = []

    for first_index, first in enumerate(discontinuities):
        for second in discontinuities[first_index + 1 :]:
            result = block_toppling_result(
                first,
                second,
                discontinuities,
                slope_dip_direction,
                slope_dip,
                friction_angle,
                lateral_limit,
            )
            if result is None:
                continue
            results.append(result)
            rows.append(
                {
                    "Planes": f"{result['first']} + {result['second']}",
                    "Trend": f"{result['trend']:03.0f}",
                    "Plunge": f"{result['plunge']:02.0f}",
                    "Into-slope alignment": f"{result['alignment']:.0f}",
                    "Intersection zone": result["intersection_zone"],
                    "Release planes": ", ".join(
                        f"{plane['label']} ({plane['zone']}: {plane['release_mode']})" for plane in result["release_planes"]
                    )
                    if result["release_planes"]
                    else "-",
                    "Block toppling": "Potential" if result["susceptible"] else "No",
                    "Reason": result["reason"],
                }
            )

    return pd.DataFrame(rows), results


def sector_polygon(
    center_bearing: float,
    half_width: float,
    inner_radius: float,
    outer_radius: float,
    samples: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.linspace(-half_width, half_width, samples)
    outer_bearings = np.deg2rad((center_bearing + offsets) % 360.0)
    inner_bearings = np.deg2rad((center_bearing + offsets[::-1]) % 360.0)
    x = np.concatenate([outer_radius * np.sin(outer_bearings), inner_radius * np.sin(inner_bearings)])
    y = np.concatenate([outer_radius * np.cos(outer_bearings), inner_radius * np.cos(inner_bearings)])
    return x, y


def draw_annular_sector(
    ax: plt.Axes,
    center_bearing: float,
    half_width: float,
    inner_radius: float,
    outer_radius: float,
    color: str,
    label: str,
    alpha: float = 0.14,
) -> None:
    inner_radius = max(0.0, min(1.0, inner_radius))
    outer_radius = max(0.0, min(1.0, outer_radius))
    if outer_radius <= inner_radius:
        return
    x, y = sector_polygon(center_bearing, half_width, inner_radius, outer_radius)
    ax.fill(x, y, color=color, alpha=alpha, zorder=0.5, label=label)


def vector_components(trend: np.ndarray, plunge: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trend_rad = np.deg2rad(trend)
    plunge_rad = np.deg2rad(plunge)
    return (
        np.sin(trend_rad) * np.cos(plunge_rad),
        np.cos(trend_rad) * np.cos(plunge_rad),
        -np.sin(plunge_rad),
    )


def draw_sampled_zone(ax: plt.Axes, membership: np.ndarray, xx: np.ndarray, yy: np.ndarray, color: str) -> None:
    if not np.any(membership):
        return
    values = np.ma.masked_where(~membership, membership.astype(float))
    ax.contourf(xx, yy, values, levels=[0.5, 1.5], colors=[color], alpha=0.18, zorder=0.45)


def add_stereonet_grid(ax: plt.Axes) -> None:
    perimeter = plt.Circle((0, 0), 1, edgecolor="#202124", facecolor="none", linewidth=1.4)
    ax.add_patch(perimeter)

    for angle in range(0, 360, 10):
        rad = np.deg2rad(angle)
        width = 0.8 if angle % 30 else 1.1
        color = "#d7dde2" if angle % 30 else "#b7c0c9"
        ax.plot([0, np.sin(rad)], [0, np.cos(rad)], color=color, linewidth=width, zorder=0)

    theta = np.linspace(0, 2 * np.pi, 361)
    for radius in np.tan(np.deg2rad((90.0 - np.arange(10, 90, 10)) / 2.0)):
        ax.plot(radius * np.sin(theta), radius * np.cos(theta), color="#d7dde2", linewidth=0.8, zorder=0)

    for label, x, y, ha, va in [
        ("N", 0, 1.08, "center", "bottom"),
        ("E", 1.08, 0, "left", "center"),
        ("S", 0, -1.08, "center", "top"),
        ("W", -1.08, 0, "right", "center"),
    ]:
        ax.text(x, y, label, ha=ha, va=va, fontsize=11, weight="bold", color="#202124")

    ax.set_aspect("equal")
    ax.set_xlim(-1.13, 1.13)
    ax.set_ylim(-1.13, 1.13)
    ax.axis("off")


def add_analysis_zones(
    ax: plt.Axes,
    show_planar: bool,
    show_wedge: bool,
    show_toppling: bool,
    slope_dip_direction: float,
    slope_dip: float,
    friction_angle: float,
    lateral_limit: float,
) -> None:
    grid = np.linspace(-1.0, 1.0, 251)
    xx, yy = np.meshgrid(grid, grid)
    radius = np.hypot(xx, yy)
    inside_net = radius <= 1.0
    trend = (np.rad2deg(np.arctan2(xx, yy)) + 360.0) % 360.0
    plunge = 90.0 - np.rad2deg(2.0 * np.arctan(radius))

    slope_normal = plane_normal(slope_dip_direction, slope_dip)

    if show_planar:
        plane_dip_direction = (trend + 180.0) % 360.0
        plane_dip = 90.0 - plunge
        dx, dy, dz = vector_components(plane_dip_direction, plane_dip)
        daylights = slope_normal[0] * dx + slope_normal[1] * dy + slope_normal[2] * dz < -1e-10
        planar_membership = (
            inside_net
            & (angular_difference_array(plane_dip_direction, slope_dip_direction) <= lateral_limit)
            & (plane_dip > friction_angle)
            & daylights
        )
        draw_sampled_zone(ax, planar_membership, xx, yy, KINEMATIC_HIGHLIGHT)

    if show_wedge:
        lx, ly, lz = vector_components(trend, plunge)
        line_daylights = slope_normal[0] * lx + slope_normal[1] * ly + slope_normal[2] * lz < -1e-10
        wedge_membership = (
            inside_net
            & (angular_difference_array(trend, slope_dip_direction) <= lateral_limit)
            & (plunge > friction_angle)
            & line_daylights
        )
        draw_sampled_zone(ax, wedge_membership, xx, yy, WEDGE_HIGHLIGHT)

    if show_toppling:
        toppling_threshold = max(0.0, 90.0 - slope_dip + friction_angle)
        plane_dip_direction = (trend + 180.0) % 360.0
        plane_dip = 90.0 - plunge
        toppling_membership = (
            inside_net
            & (angular_difference_array(plane_dip_direction, (slope_dip_direction + 180.0) % 360.0) <= lateral_limit)
            & (plane_dip > toppling_threshold)
        )
        draw_sampled_zone(ax, toppling_membership, xx, yy, TOPPLING_HIGHLIGHT)


def add_analysis_guides(
    ax: plt.Axes,
    show_planar: bool,
    show_wedge: bool,
    show_toppling: bool,
    slope_dip_direction: float,
    slope_dip: float,
    friction_angle: float,
    lateral_limit: float,
) -> None:
    for center in [slope_dip_direction, (slope_dip_direction + 180.0) % 360.0]:
        left = (center - lateral_limit) % 360.0
        right = (center + lateral_limit) % 360.0
        for bearing in [left, right]:
            rad = np.deg2rad(bearing)
            ax.plot(
                [0, np.sin(rad)],
                [0, np.cos(rad)],
                color=KINEMATIC_HIGHLIGHT,
                linewidth=1.2,
                linestyle="--",
                alpha=0.55,
                zorder=1,
            )

    for bearing in [slope_dip_direction, (slope_dip_direction + 180.0) % 360.0]:
        rad = np.deg2rad(bearing)
        ax.plot(
            [0, np.sin(rad)],
            [0, np.cos(rad)],
            color="#44403c",
            linewidth=1.0,
            linestyle="-",
            alpha=0.35,
            zorder=1,
        )

    theta = np.linspace(0, 2 * np.pi, 361)
    if show_planar or show_toppling:
        pole_friction_radius = np.tan(np.deg2rad(friction_angle / 2.0))
        ax.plot(
            pole_friction_radius * np.sin(theta),
            pole_friction_radius * np.cos(theta),
            color="#9a3412",
            linewidth=1.6,
            linestyle=":",
            alpha=0.9,
            label=f"Pole friction {friction_angle:.0f} deg",
        )
    if show_wedge:
        plane_friction_radius = np.tan(np.deg2rad((90.0 - friction_angle) / 2.0))
        ax.plot(
            plane_friction_radius * np.sin(theta),
            plane_friction_radius * np.cos(theta),
            color="#4f46e5",
            linewidth=1.5,
            linestyle=":",
            alpha=0.75,
            label=f"Line friction {friction_angle:.0f} deg",
        )
    for x, y in great_circle_xy(slope_dip_direction, slope_dip):
        ax.plot(x, y, color="#57534e", linewidth=1.6, linestyle="-.", alpha=0.85, label="Slope/daylight boundary")


def plot_stereonet(
    orientations: list[dict[str, object]],
    show_planar_analysis: bool = False,
    show_wedge_analysis: bool = False,
    show_toppling_analysis: bool = False,
    show_block_toppling_analysis: bool = False,
    show_analysis_zones: bool = True,
    wedge_results: list[dict[str, object]] | None = None,
    block_toppling_results: list[dict[str, object]] | None = None,
    slope_dip_direction: float | None = None,
    slope_dip: float | None = None,
    friction_angle: float | None = None,
    lateral_limit: float | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=140)
    fig.patch.set_facecolor("#ffffff")
    add_stereonet_grid(ax)

    show_any_analysis = show_planar_analysis or show_wedge_analysis or show_toppling_analysis or show_block_toppling_analysis
    if show_analysis_zones and show_any_analysis:
        add_analysis_zones(
            ax,
            show_planar_analysis,
            show_wedge_analysis,
            show_toppling_analysis,
            float(slope_dip_direction),
            float(slope_dip),
            float(friction_angle),
            float(lateral_limit),
        )

    if show_any_analysis:
        add_analysis_guides(
            ax,
            show_planar_analysis,
            show_wedge_analysis,
            show_toppling_analysis,
            float(slope_dip_direction),
            float(slope_dip),
            float(friction_angle),
            float(lateral_limit),
        )

    for item in orientations:
        kind = str(item["type"])
        label = str(item["label"])
        dip_direction = float(item["dip_direction"])
        dip = float(item["dip"])
        color = COLORS.get(kind, "#555555")
        planar = item.get("planar")
        toppling = item.get("toppling")
        planar_susceptible = bool(planar and planar["susceptible"])
        toppling_susceptible = bool(toppling and toppling["susceptible"])
        susceptible = planar_susceptible or toppling_susceptible
        line_width = 3.0 if susceptible else 2.0
        line_alpha = 1.0 if susceptible else 0.88

        for x, y in great_circle_xy(dip_direction, dip):
            ax.plot(x, y, color=color, linewidth=line_width, alpha=line_alpha)

        px, py, trend, plunge = pole_xy(dip_direction, dip)
        marker = "o"
        edgecolor = "white"
        if planar_susceptible and toppling_susceptible:
            marker = "*"
            edgecolor = TOPPLING_HIGHLIGHT
        elif planar_susceptible:
            marker = "D"
            edgecolor = KINEMATIC_HIGHLIGHT
        elif toppling_susceptible:
            marker = "s"
            edgecolor = TOPPLING_HIGHLIGHT
        ax.scatter([px], [py], s=96 if susceptible else 62, marker=marker, color=color, edgecolor=edgecolor, linewidth=1.3, zorder=5)
        ax.annotate(
            label,
            (px, py),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8.5,
            color="#202124",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": color, "linewidth": 0.8},
        )
        item["pole"] = f"{trend:03.0f}/{plunge:02.0f}"

    wedge_results = wedge_results or []
    for index, result in enumerate(wedge_results, start=1):
        trend = float(result["trend"])
        plunge = float(result["plunge"])
        x, y = project_lower_hemisphere(np.array([trend]), np.array([plunge]))
        susceptible = bool(result["susceptible"])
        ax.scatter(
            [x[0]],
            [y[0]],
            s=100 if susceptible else 58,
            marker="X",
            color=WEDGE_HIGHLIGHT if susceptible else "#818cf8",
            edgecolor="white",
            linewidth=1.0,
            zorder=6,
        )
        ax.annotate(
            f"W{index}",
            (x[0], y[0]),
            xytext=(6, -10),
            textcoords="offset points",
            fontsize=8.5,
            color="#202124",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": WEDGE_HIGHLIGHT, "linewidth": 0.8},
        )

    block_toppling_results = block_toppling_results or []
    for index, result in enumerate(block_toppling_results, start=1):
        trend = float(result["trend"])
        plunge = float(result["plunge"])
        x, y = project_lower_hemisphere(np.array([trend]), np.array([plunge]))
        susceptible = bool(result["susceptible"])
        ax.scatter(
            [x[0]],
            [y[0]],
            s=108 if susceptible else 60,
            marker="P",
            color=BLOCK_TOPPLING_HIGHLIGHT if susceptible else "#fbbf24",
            edgecolor="white",
            linewidth=1.0,
            zorder=7,
        )
        ax.annotate(
            f"B{index}",
            (x[0], y[0]),
            xytext=(6, 10),
            textcoords="offset points",
            fontsize=8.5,
            color="#202124",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": BLOCK_TOPPLING_HIGHLIGHT, "linewidth": 0.8},
        )

    handles = [
        plt.Line2D([0], [0], color=color, marker="o", markersize=6, linewidth=2, label=kind)
        for kind, color in COLORS.items()
        if any(item["type"] == kind for item in orientations)
    ]
    if any(item.get("planar") and item["planar"]["susceptible"] for item in orientations):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=KINEMATIC_HIGHLIGHT,
                marker="D",
                markersize=6,
                linewidth=2.5,
                label="Planar sliding potential",
            )
        )
    if any(item.get("toppling") and item["toppling"]["susceptible"] for item in orientations):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=TOPPLING_HIGHLIGHT,
                marker="s",
                markersize=6,
                linewidth=2.5,
                label="Toppling potential",
            )
        )
    if any(result["susceptible"] for result in wedge_results):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=WEDGE_HIGHLIGHT,
                marker="X",
                markersize=7,
                linewidth=0,
                label="Wedge sliding potential",
            )
        )
    if any(result["susceptible"] for result in block_toppling_results):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=BLOCK_TOPPLING_HIGHLIGHT,
                marker="P",
                markersize=7,
                linewidth=0,
                label="Block toppling potential",
            )
        )
    if show_any_analysis:
        if show_planar_analysis or show_toppling_analysis or show_block_toppling_analysis:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="#9a3412",
                    linewidth=1.6,
                    linestyle=":",
                    label=f"Pole friction angle ({float(friction_angle):.0f} deg)",
                )
            )
        if show_wedge_analysis:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="#4f46e5",
                    linewidth=1.5,
                    linestyle=":",
                    label=f"Line friction angle ({float(friction_angle):.0f} deg)",
                )
            )
        handles.extend(
            [
                plt.Line2D([0], [0], color="#57534e", linewidth=1.4, linestyle="-.", label="Slope/daylight boundary"),
                plt.Line2D([0], [0], color=KINEMATIC_HIGHLIGHT, linewidth=1.5, linestyle="--", label="Lateral limits"),
            ]
        )
    if show_analysis_zones and show_any_analysis:
        if show_planar_analysis:
            handles.append(plt.Line2D([0], [0], color=KINEMATIC_HIGHLIGHT, linewidth=8, alpha=0.35, label="Planar zone"))
        if show_wedge_analysis:
            handles.append(plt.Line2D([0], [0], color=WEDGE_HIGHLIGHT, linewidth=8, alpha=0.35, label="Wedge zone"))
        if show_toppling_analysis:
            handles.append(plt.Line2D([0], [0], color=TOPPLING_HIGHLIGHT, linewidth=8, alpha=0.35, label="Toppling zone"))
    if handles:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=3,
            frameon=False,
            fontsize=8.5,
        )

    fig.tight_layout(rect=(0, 0.13, 1, 1))
    return fig


def build_orientations(slope_dd: float, slope_dip: float, sets: pd.DataFrame, joints: pd.DataFrame | None = None) -> list[dict[str, object]]:
    orientations: list[dict[str, object]] = [
        {
            "type": "Slope",
            "label": "Slope",
            "dip_direction": clean_orientation(slope_dd, 360, "Slope dip direction") % 360,
            "dip": clean_orientation(slope_dip, 90, "Slope dip"),
        }
    ]

    if joints is not None:
        foliation = sets.copy()
        joints = joints.copy()
        sets = pd.concat([foliation, joints], ignore_index=True)

    for idx, row in sets.iterrows():
        if not bool(row.get("Plot", True)):
            continue
        label = str(row.get("Name") or f"S{idx + 1}")
        orientations.append(
            {
                "type": "Discontinuity",
                "label": label,
                "dip_direction": clean_orientation(row.get("Dip direction"), 360, f"{label} dip direction") % 360,
                "dip": clean_orientation(row.get("Dip"), 90, f"{label} dip"),
            }
        )

    return orientations


def clean_report_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    table = dataframe.copy()
    table = table.where(pd.notnull(table), "")
    for column in table.columns:
        table[column] = table[column].map(lambda value: str(value))
    return table


def report_table_without_columns(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return dataframe.drop(columns=[column for column in columns if column in dataframe.columns])


def draw_report_table(
    ax: plt.Axes,
    title: str,
    dataframe: pd.DataFrame,
    bbox: list[float],
    font_size: float = 6.1,
    row_height: float = 0.031,
) -> None:
    table = clean_report_table(dataframe)
    if table.empty:
        table = pd.DataFrame({"Result": ["No records to report."]})

    x, y, width, height = bbox
    table_height = min(height, max(0.055, (len(table) + 1) * row_height))
    table_y = y + height - table_height
    ax.text(x, y + height + 0.012, title, fontsize=9.0, fontweight="bold", color="#202124", transform=ax.transAxes)
    rendered_table = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        bbox=[x, table_y, width, table_height],
    )
    rendered_table.auto_set_font_size(False)
    rendered_table.set_fontsize(font_size)
    for (row, _column), cell in rendered_table.get_celld().items():
        cell.set_edgecolor("#d6d3d1")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#e7e5e4")
            cell.set_text_props(weight="bold", color="#202124")
        else:
            cell.set_facecolor("#ffffff")
            cell.set_text_props(color="#202124")


def add_report_tables_page(
    pdf: PdfPages,
    location_id: str,
    slope_dip_direction: float,
    slope_dip: float,
    friction_angle: float,
    lateral_limit: float,
    sets: pd.DataFrame,
    planar: pd.DataFrame,
    wedge: pd.DataFrame,
    toppling: pd.DataFrame,
    block_toppling: pd.DataFrame,
    include_planar: bool,
    include_wedge: bool,
    include_toppling: bool,
    include_block_toppling: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.06, 0.955, "Stereonet and Kinematic Analysis", fontsize=18, fontweight="bold", color="#202124", transform=ax.transAxes)
    ax.text(0.06, 0.925, f"Location ID: {location_id or '-'}", fontsize=11, color="#202124", transform=ax.transAxes)

    inputs = pd.DataFrame(
        [
            {"Parameter": "Slope dip direction", "Value": f"{slope_dip_direction:.0f} deg"},
            {"Parameter": "Slope dip", "Value": f"{slope_dip:.0f} deg"},
            {"Parameter": "Friction angle", "Value": f"{friction_angle:.0f} deg"},
            {"Parameter": "Lateral limit", "Value": f"{lateral_limit:.0f} deg"},
        ]
    )
    table_font_size = 6.1
    draw_report_table(ax, "Analysis inputs", inputs, [0.06, 0.70, 0.88, 0.14], font_size=table_font_size, row_height=0.031)
    draw_report_table(ax, "Discontinuity sets", sets, [0.06, 0.52, 0.88, 0.12], font_size=table_font_size, row_height=0.026)

    planar_table = report_table_without_columns(planar, ["Reason"]) if include_planar else pd.DataFrame({"Result": ["Planar sliding analysis disabled."]})
    wedge_table = report_table_without_columns(wedge, ["Reason"]) if include_wedge else pd.DataFrame({"Result": ["Wedge sliding analysis disabled."]})
    toppling_table = report_table_without_columns(toppling, ["Reason"]) if include_toppling else pd.DataFrame({"Result": ["Toppling analysis disabled."]})
    block_toppling_table = (
        report_table_without_columns(block_toppling, ["Reason"])
        if include_block_toppling
        else pd.DataFrame({"Result": ["Block toppling analysis disabled."]})
    )

    ax.text(0.06, 0.487, "Kinematic analysis", fontsize=11, fontweight="bold", color="#202124", transform=ax.transAxes)
    draw_report_table(ax, "Planar sliding", planar_table, [0.06, 0.37, 0.88, 0.075], font_size=table_font_size, row_height=0.020)
    draw_report_table(ax, "Wedge sliding", wedge_table, [0.06, 0.265, 0.88, 0.075], font_size=table_font_size, row_height=0.020)
    draw_report_table(ax, "Flexural toppling", toppling_table, [0.06, 0.16, 0.88, 0.075], font_size=table_font_size, row_height=0.020)
    draw_report_table(ax, "Block toppling", block_toppling_table, [0.06, 0.055, 0.88, 0.075], font_size=table_font_size, row_height=0.020)
    pdf.savefig(fig)
    plt.close(fig)


def add_report_plot_page(pdf: PdfPages, stereonet_fig: plt.Figure) -> None:
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


def build_pdf_report(
    location_id: str,
    slope_dip_direction: float,
    slope_dip: float,
    friction_angle: float,
    lateral_limit: float,
    sets: pd.DataFrame,
    stereonet_fig: plt.Figure,
    summary: pd.DataFrame,
    planar: pd.DataFrame,
    wedge: pd.DataFrame,
    toppling: pd.DataFrame,
    block_toppling: pd.DataFrame,
    include_planar: bool,
    include_wedge: bool,
    include_toppling: bool,
    include_block_toppling: bool,
) -> bytes:
    report_buffer = BytesIO()
    with PdfPages(report_buffer) as pdf:
        add_report_tables_page(
            pdf,
            location_id,
            slope_dip_direction,
            slope_dip,
            friction_angle,
            lateral_limit,
            sets,
            planar,
            wedge,
            toppling,
            block_toppling,
            include_planar,
            include_wedge,
            include_toppling,
            include_block_toppling,
        )
        add_report_plot_page(pdf, stereonet_fig)
    return report_buffer.getvalue()


def report_file_name(location_id: str) -> str:
    clean_location = re.sub(r"[^A-Za-z0-9._-]+", "_", location_id.strip()).strip("_")
    return f"{clean_location or 'stereonet_kinematic_analysis_report'}.pdf"


def project_file_name(location_id: str) -> str:
    clean_location = re.sub(r"[^A-Za-z0-9._-]+", "_", location_id.strip()).strip("_")
    return f"{clean_location or 'stereonet_project'}.json"


def clean_sets_for_project(sets: pd.DataFrame) -> list[dict[str, object]]:
    cleaned = sets.where(pd.notnull(sets), None).to_dict(orient="records")
    for row in cleaned:
        if "Plot" in row:
            row["Plot"] = bool(row["Plot"])
        for column in ["Dip direction", "Dip"]:
            if row.get(column) is not None:
                row[column] = int(round(float(row[column])))
    return cleaned


def build_project_data(
    location_id: str,
    slope_dip_direction: float,
    slope_dip: float,
    friction_angle: float,
    lateral_limit: float,
    sets: pd.DataFrame,
    enable_planar: bool,
    enable_wedge: bool,
    enable_toppling: bool,
    enable_block_toppling: bool,
    show_analysis_zones: bool,
    show_table: bool,
) -> dict[str, object]:
    return {
        "project_version": 1,
        "location_id": location_id,
        "slope_dip_direction": int(round(float(slope_dip_direction))),
        "slope_dip": int(round(float(slope_dip))),
        "friction_angle": int(round(float(friction_angle))),
        "lateral_limit": int(round(float(lateral_limit))),
        "enable_planar": bool(enable_planar),
        "enable_wedge": bool(enable_wedge),
        "enable_toppling": bool(enable_toppling),
        "enable_block_toppling": bool(enable_block_toppling),
        "show_analysis_zones": bool(show_analysis_zones),
        "show_table": bool(show_table),
        "sets": clean_sets_for_project(sets),
    }


def normalise_project_data(data: dict[str, object], fallback: dict[str, object]) -> dict[str, object]:
    project = fallback | data
    project["location_id"] = str(project.get("location_id", ""))
    project["slope_dip_direction"] = clean_orientation(project.get("slope_dip_direction"), 360, "Project slope dip direction") % 360
    project["slope_dip"] = clean_orientation(project.get("slope_dip"), 90, "Project slope dip")
    project["friction_angle"] = clean_orientation(project.get("friction_angle"), 89, "Project friction angle")
    project["lateral_limit"] = clean_orientation(project.get("lateral_limit"), 90, "Project lateral limit")
    project["enable_planar"] = bool(project.get("enable_planar", True))
    project["enable_wedge"] = bool(project.get("enable_wedge", True))
    project["enable_toppling"] = bool(project.get("enable_toppling", True))
    project["enable_block_toppling"] = bool(project.get("enable_block_toppling", False))
    project["show_analysis_zones"] = bool(project.get("show_analysis_zones", True))
    project["show_table"] = bool(project.get("show_table", True))
    project["sets"] = [{key: value for key, value in row.items() if key != "Type"} for row in project.get("sets", fallback["sets"])]
    return project


def apply_project_to_session(project: dict[str, object]) -> None:
    st.session_state["location_id"] = project["location_id"]
    st.session_state["slope_dip_direction"] = int(round(float(project["slope_dip_direction"])))
    st.session_state["slope_dip"] = int(round(float(project["slope_dip"])))
    st.session_state["friction_angle"] = int(round(float(project["friction_angle"])))
    st.session_state["lateral_limit"] = int(round(float(project["lateral_limit"])))
    st.session_state["enable_planar"] = bool(project["enable_planar"])
    st.session_state["enable_wedge"] = bool(project["enable_wedge"])
    st.session_state["enable_toppling"] = bool(project["enable_toppling"])
    st.session_state["enable_block_toppling"] = bool(project["enable_block_toppling"])
    st.session_state["show_analysis_zones"] = bool(project["show_analysis_zones"])
    st.session_state["show_table"] = bool(project["show_table"])
    st.session_state["project_sets"] = project["sets"]
    st.session_state["project_load_count"] = int(st.session_state.get("project_load_count", 0)) + 1


def coerce_integer_session_values() -> None:
    for key in ["slope_dip_direction", "slope_dip", "friction_angle", "lateral_limit"]:
        if key in st.session_state:
            st.session_state[key] = int(round(float(st.session_state[key])))


def main() -> None:
    st.set_page_config(page_title="Stereonet and Kinematic Analysis", layout="wide")
    st.title("Stereonet and Kinematic Analysis")
    defaults = load_defaults()

    with st.sidebar:
        st.header("Project files")
        project_upload = st.file_uploader("Load project JSON", type=["json"])
        if project_upload is not None:
            uploaded_bytes = project_upload.getvalue()
            upload_signature = f"{project_upload.name}:{len(uploaded_bytes)}"
            if st.session_state.get("loaded_project_signature") != upload_signature:
                try:
                    project_data = json.loads(uploaded_bytes.decode("utf-8"))
                    project = normalise_project_data(project_data, defaults)
                    apply_project_to_session(project)
                    st.session_state["loaded_project_signature"] = upload_signature
                    st.success(f"Loaded project: {project_upload.name}")
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    st.error(f"Could not load project file: {exc}")

    coerce_integer_session_values()
    initial_sets = st.session_state.get("project_sets", defaults["sets"])
    editor_key = f"sets_editor_{st.session_state.get('project_load_count', 0)}"
    location_id = st.text_input(
        "Location ID:",
        value=str(st.session_state.get("location_id", defaults.get("location_id", ""))),
        placeholder="Enter location ID",
        key="location_id",
    )

    with st.sidebar:
        st.header("Slope")
        slope_dd = st.number_input(
            "Slope dip direction",
            min_value=0,
            max_value=360,
            value=int(round(float(st.session_state.get("slope_dip_direction", defaults["slope_dip_direction"])))),
            step=1,
            format="%d",
            key="slope_dip_direction",
        )
        slope_dip = st.number_input(
            "Slope dip",
            min_value=0,
            max_value=90,
            value=int(round(float(st.session_state.get("slope_dip", defaults["slope_dip"])))),
            step=1,
            format="%d",
            key="slope_dip",
        )
        st.header("Kinematic analysis")
        enable_planar = st.toggle("Planar sliding", value=bool(st.session_state.get("enable_planar", True)), key="enable_planar")
        enable_wedge = st.toggle("Wedge sliding", value=bool(st.session_state.get("enable_wedge", True)), key="enable_wedge")
        enable_toppling = st.toggle("Flexural toppling", value=bool(st.session_state.get("enable_toppling", True)), key="enable_toppling")
        enable_block_toppling = st.toggle(
            "Block toppling",
            value=bool(st.session_state.get("enable_block_toppling", False)),
            key="enable_block_toppling",
        )
        show_analysis_zones = st.toggle(
            "Show analysis zones",
            value=bool(st.session_state.get("show_analysis_zones", True)),
            key="show_analysis_zones",
        )
        friction_angle = st.number_input(
            "Friction angle",
            min_value=0,
            max_value=89,
            value=int(round(float(st.session_state.get("friction_angle", defaults["friction_angle"])))),
            step=1,
            format="%d",
            key="friction_angle",
        )
        lateral_limit = st.number_input(
            "Lateral limit",
            min_value=0,
            max_value=90,
            value=int(round(float(st.session_state.get("lateral_limit", defaults["lateral_limit"])))),
            step=1,
            format="%d",
            key="lateral_limit",
        )
        show_table = st.toggle("Show orientation summary", value=bool(st.session_state.get("show_table", True)), key="show_table")

    default_sets = pd.DataFrame(initial_sets)

    st.subheader("Discontinuity Sets")
    sets_df = st.data_editor(
        default_sets,
        key=editor_key,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Plot": st.column_config.CheckboxColumn(default=True, width="small"),
            "Name": st.column_config.TextColumn(required=False, width="medium"),
            "Dip direction": st.column_config.NumberColumn(min_value=0, max_value=360, step=1, format="%d", width="medium"),
            "Dip": st.column_config.NumberColumn(min_value=0, max_value=90, step=1, format="%d", width="medium"),
        },
    )

    project_json = json.dumps(
        build_project_data(
            location_id=location_id,
            slope_dip_direction=slope_dd,
            slope_dip=slope_dip,
            friction_angle=friction_angle,
            lateral_limit=lateral_limit,
            sets=sets_df,
            enable_planar=enable_planar,
            enable_wedge=enable_wedge,
            enable_toppling=enable_toppling,
            enable_block_toppling=enable_block_toppling,
            show_analysis_zones=show_analysis_zones,
            show_table=show_table,
        ),
        indent=2,
    )
    with st.sidebar:
        st.download_button(
            "Save project JSON",
            data=project_json,
            file_name=project_file_name(location_id),
            mime="application/json",
        )

    try:
        orientations = build_orientations(slope_dd, slope_dip, sets_df)
        planar_df = pd.DataFrame()
        wedge_df = pd.DataFrame()
        wedge_results: list[dict[str, object]] = []
        toppling_df = pd.DataFrame()
        block_toppling_df = pd.DataFrame()
        block_toppling_results: list[dict[str, object]] = []
        if enable_planar:
            planar_df = annotate_planar_sliding(orientations, slope_dd, slope_dip, friction_angle, lateral_limit)
        if enable_wedge:
            wedge_df, wedge_results = analyse_wedge_sliding(orientations, slope_dd, slope_dip, friction_angle, lateral_limit)
        if enable_toppling:
            toppling_df = annotate_toppling(orientations, slope_dd, slope_dip, friction_angle, lateral_limit)
        if enable_block_toppling:
            block_toppling_df, block_toppling_results = analyse_block_toppling(
                orientations,
                slope_dd,
                slope_dip,
                friction_angle,
                lateral_limit,
            )

        fig = plot_stereonet(
            orientations,
            show_planar_analysis=enable_planar,
            show_wedge_analysis=enable_wedge,
            show_toppling_analysis=enable_toppling,
            show_block_toppling_analysis=enable_block_toppling,
            show_analysis_zones=show_analysis_zones,
            wedge_results=wedge_results,
            block_toppling_results=block_toppling_results,
            slope_dip_direction=slope_dd,
            slope_dip=slope_dip,
            friction_angle=friction_angle,
            lateral_limit=lateral_limit,
        )
        st.pyplot(fig, clear_figure=False)

        image_buffer = BytesIO()
        fig.savefig(image_buffer, format="png", dpi=220, bbox_inches="tight")
        st.download_button(
            "Download PNG",
            data=image_buffer.getvalue(),
            file_name="geotechnical_stereonet.png",
            mime="image/png",
        )

        summary = pd.DataFrame(
            [
                {
                    "Type": item["type"],
                    "Name": item["label"],
                    "Dip direction": f"{float(item['dip_direction']):03.0f}",
                    "Dip": f"{float(item['dip']):02.0f}",
                    "Pole trend/plunge": item["pole"],
                }
                for item in orientations
            ]
        )

        if show_table:
            st.subheader("Orientation summary")
            st.dataframe(summary, hide_index=True, use_container_width=True)

        if enable_planar:
            st.subheader("Planar sliding analysis")
            if planar_df.empty:
                st.info("Add discontinuity sets to analyse planar sliding.")
            else:
                susceptible_count = int((planar_df["Planar sliding"] == "Potential").sum())
                st.metric("Potential planar sliding planes", susceptible_count)
                st.dataframe(planar_df, hide_index=True, use_container_width=True)

        if enable_wedge:
            st.subheader("Wedge sliding analysis")
            if wedge_df.empty:
                st.info("Add at least two discontinuity sets to analyse wedge sliding.")
            else:
                susceptible_count = int((wedge_df["Wedge sliding"] == "Potential").sum())
                st.metric("Potential wedge intersections", susceptible_count)
                st.dataframe(wedge_df, hide_index=True, use_container_width=True)

        if enable_toppling:
            st.subheader("Flexural toppling analysis")
            if toppling_df.empty:
                st.info("Add discontinuity sets to analyse flexural toppling.")
            else:
                susceptible_count = int((toppling_df["Toppling"] == "Potential").sum())
                st.metric("Potential flexural toppling planes", susceptible_count)
                st.dataframe(toppling_df, hide_index=True, use_container_width=True)

        if enable_block_toppling:
            st.subheader("Block toppling analysis")
            if block_toppling_df.empty:
                st.info("Add at least three discontinuity sets to analyse block toppling with a release/base plane.")
            else:
                susceptible_count = int((block_toppling_df["Block toppling"] == "Potential").sum())
                st.metric("Potential block toppling intersections", susceptible_count)
                st.dataframe(block_toppling_df, hide_index=True, use_container_width=True)

        st.subheader("Export report")
        if st.button("Generate PDF report", type="primary"):
            pdf_bytes = build_pdf_report(
                location_id=location_id,
                slope_dip_direction=slope_dd,
                slope_dip=slope_dip,
                friction_angle=friction_angle,
                lateral_limit=lateral_limit,
                sets=sets_df,
                stereonet_fig=fig,
                summary=summary,
                planar=planar_df,
                wedge=wedge_df,
                toppling=toppling_df,
                block_toppling=block_toppling_df,
                include_planar=enable_planar,
                include_wedge=enable_wedge,
                include_toppling=enable_toppling,
                include_block_toppling=enable_block_toppling,
            )
            file_name = report_file_name(location_id)
            REPORTS_DIR.mkdir(exist_ok=True)
            report_path = REPORTS_DIR / file_name
            report_path.write_bytes(pdf_bytes)
            st.session_state["report_pdf_bytes"] = pdf_bytes
            st.session_state["report_pdf_name"] = file_name
            st.session_state["report_pdf_path"] = str(report_path)

        if "report_pdf_bytes" in st.session_state:
            st.success(f"PDF report generated: {st.session_state['report_pdf_path']}")
            st.download_button(
                "Download PDF report",
                data=st.session_state["report_pdf_bytes"],
                file_name=st.session_state["report_pdf_name"],
                mime="application/pdf",
            )

    except ValueError as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
