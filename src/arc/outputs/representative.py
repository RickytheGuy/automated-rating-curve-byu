"""Representative cross sections: each reach's hydraulics averaged over its cross sections, stage by stage
(legacy hydraulic_data.build_representative_cross_section_dataframe).

Every 0.10 m above each cross section's stream cell, up to 25 m, the cross section's area, wetted perimeter, top
width, discharge and velocity are worked out on its hydraulic profile (arc.rating_curve.section_hydraulics), which
is a carved channel's exact shape where it has one. With banks, the velocity is the channel's (its discharge over
its area); without them it's the whole section's. At each stage, the cross sections whose area, velocity and top
width are all within two standard deviations of the reach's means are averaged. The representative cross section is
then built from the averages: its area and top width kept from falling as the stage rises, and its depth at each
stage the one that grows the width linearly from the stage below to give that area.

Numerical differences
---------------------
- The hydraulics aren't rounded. Legacy rounded area, perimeter, top width, discharge and velocity to 3 decimals, and
  each water surface elevation to the millimetre.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd

from arc import LOG
from arc.hydraulics import DepthRoughness, _check_roughness, _check_slope_factor, hydraulic_profile
from arc.xsection.xsection import Profile, XSection

DEPTH_INCREMENT = 0.10  # metres between stages
MAX_DEPTH = 25.0  # the highest stage above the stream cell

REPRESENTATIVE_CROSS_SECTION_COLUMNS = [
    'COMID', 'Cross_Section_Count', 'Hydraulic_Sample_Count', 'Depth_Stage_Index', 'Depth_Stage_Meters',
    'Stream_Slope', 'Reach_Inflect_Terrace_Depth', 'Representative_Thalweg_Elevation', 'Mean_Discharge', 'Mean_Depth',
    'Mean_Velocity', 'Representative_Velocity', 'Mean_Top_Width', 'Mean_Cross_Sectional_Area', 'Mean_WSE',
    'Representative_Cross_Sectional_Area', 'Representative_Depth_Increment', 'Representative_Depth',
    'Representative_Top_Width', 'Representative_Stage_Elevation', 'Representative_Left_Station',
    'Representative_Right_Station',
]
_INTEGER_COLUMNS = ['COMID', 'Cross_Section_Count', 'Hydraulic_Sample_Count', 'Depth_Stage_Index']


class RepresentativeSample(NamedTuple):
    """One cross section of a reach, with its banks set on it for the channel velocity."""
    comid: int
    xs: XSection
    thalweg: float
    slope: float


def monotonic_cumulative_max(values: np.ndarray) -> np.ndarray:
    """values made non-decreasing, skipping NaNs (legacy _monotonic_cumulative_max)."""
    result = np.asarray(values, dtype=np.float64).copy()
    running_max = -np.inf
    for i in range(result.size):
        if np.isnan(result[i]):
            continue
        if result[i] < running_max:
            result[i] = running_max
        else:
            running_max = result[i]
    return result


def depths_from_width_and_area(widths: np.ndarray, areas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each stage's depth increment and depth, growing the width linearly between stages to give each area
    (legacy _derive_depths_from_width_and_area)."""
    increments = np.zeros(widths.size)
    depths = np.zeros(widths.size)
    previous_width = previous_area = previous_depth = 0.0
    for i in range(widths.size):
        width, area = float(widths[i]), float(areas[i])
        delta_area = max(area - previous_area, 0.0)
        denominator = previous_width + width
        delta = (2.0 * delta_area) / denominator if denominator > 0.0 and delta_area > 0.0 else 0.0
        increments[i] = delta
        depths[i] = previous_depth + delta
        previous_width, previous_area, previous_depth = width, area, depths[i]
    return increments, depths


def _two_standard_deviations(areas, velocities, top_widths) -> np.ndarray:
    """Which samples have their area, velocity and top width all within two standard deviations of the means
    (legacy _filter_stage_samples_to_two_standard_deviations)."""
    keep = np.isfinite(areas) & np.isfinite(velocities) & np.isfinite(top_widths)
    for values in (areas, velocities, top_widths):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.zeros(values.shape, dtype=bool)
        mean, deviation = float(np.nanmean(finite)), float(np.nanstd(finite))
        keep &= (values == mean) if deviation == 0.0 else (np.abs(values - mean) <= 2.0 * deviation)
    return keep


def _stage_hydraulics(sample: RepresentativeSample, profile: Profile, depth: float, parameters, slope_factor: float):
    """Area, perimeter, velocity, discharge and top width at a stage (legacy _calculate_all with
    channel_velocity=True), on the sample's hydraulic profile."""
    from arc.rating_curve import section_hydraulics

    xs = sample.xs
    left, right = float(xs.left_bank_distance), float(xs.right_bank_distance)
    area, perimeter, top_width, k, channel_area, channel_k = section_hydraulics(
        *profile, left, right, sample.thalweg + depth, *parameters)
    scale = np.sqrt(sample.slope) * slope_factor
    if area <= 0.0 or perimeter <= 0.0 or k <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, top_width
    discharge = k * scale
    if left >= 0.0 or right >= 0.0:
        velocity = channel_k * scale / channel_area if channel_area > 0.0 else 0.0
    else:
        velocity = discharge / area
    return area, perimeter, velocity, discharge, top_width


def _reach_rows(comid: int, group: Sequence[RepresentativeSample], parameters, slope_factor: float) -> list[dict]:
    """The staged means for one reach (legacy _build_representative_hydraulic_rows_for_reach)."""
    stage_count = max(int(round(min(MAX_DEPTH, 25.0) / DEPTH_INCREMENT)), 0)
    thalweg = float(np.nanmean([s.thalweg for s in group]))
    slopes = [s.slope for s in group if s.slope > 0.0]
    stream_slope = float(np.nanmean(slopes)) if slopes else np.nan
    profiles = [hydraulic_profile(sample.xs) for sample in group]
    rows: list[dict] = []
    last_depth = 0.0
    for stage in range(1, stage_count + 1):
        depth = float(stage * DEPTH_INCREMENT)
        values = []
        for sample, profile in zip(group, profiles):
            if not sample.slope > 0.0:
                continue
            result = _stage_hydraulics(sample, profile, depth, parameters, slope_factor)
            if not all(np.isfinite(v) for v in result):
                for row in rows:
                    row['Reach_Inflect_Terrace_Depth'] = last_depth
                return rows
            area, _, velocity, discharge, top_width = result
            if area <= 0.0 or top_width <= 0.0 or discharge < 0.0:
                continue
            values.append((discharge, velocity, top_width, area, sample.thalweg + depth))
        if not values:
            continue
        discharges, velocities, top_widths, areas, wses = (np.array(column) for column in zip(*values))
        keep = _two_standard_deviations(areas, velocities, top_widths)
        if not keep.any():
            continue
        rows.append({
            'COMID': int(comid), 'Cross_Section_Count': len(group), 'Hydraulic_Sample_Count': int(keep.sum()),
            'Depth_Stage_Index': stage, 'Depth_Stage_Meters': depth, 'Stream_Slope': stream_slope,
            'Reach_Inflect_Terrace_Depth': MAX_DEPTH, 'Representative_Thalweg_Elevation': thalweg,
            'Mean_Discharge': float(np.nanmean(discharges[keep])), 'Mean_Depth': depth,
            'Mean_Velocity': float(np.nanmean(velocities[keep])), 'Mean_Top_Width': float(np.nanmean(top_widths[keep])),
            'Mean_Cross_Sectional_Area': float(np.nanmean(areas[keep])), 'Mean_WSE': float(np.nanmean(wses[keep])),
        })
        last_depth = depth
    for row in rows:
        row['Reach_Inflect_Terrace_Depth'] = last_depth
    return rows


def representative_cross_section_dataframe(samples: Sequence[RepresentativeSample], *,
                                           roughness: DepthRoughness | None = None,
                                           slope_factor: float = 1.0) -> pd.DataFrame:
    """Each reach's representative cross section, one row per 0.10 m stage (see the notes above). The reaches are
    the samples' COMIDs."""
    parameters = (1.0, 1.0, 1.0) if roughness is None else _check_roughness(roughness)
    slope_factor = _check_slope_factor(slope_factor)
    grouped: dict[int, list[RepresentativeSample]] = {}
    for sample in samples:
        grouped.setdefault(int(sample.comid), []).append(sample)
    rows = []
    for comid, group in grouped.items():
        rows.extend(_reach_rows(comid, group, parameters, slope_factor))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=REPRESENTATIVE_CROSS_SECTION_COLUMNS)

    groups = []
    for _, group in df.groupby('COMID', sort=True):
        group = group.sort_values('Depth_Stage_Index').copy()
        area = monotonic_cumulative_max(group['Mean_Cross_Sectional_Area'].to_numpy(dtype=np.float64))
        width_seed = group['Mean_Top_Width'].to_numpy(dtype=np.float64)
        fallback = area / np.maximum(group['Mean_Depth'].to_numpy(dtype=np.float64), 1e-9)
        width = monotonic_cumulative_max(np.where(width_seed > 0.0, width_seed, fallback))
        increments, depths = depths_from_width_and_area(width, area)
        group['Representative_Cross_Sectional_Area'] = area
        group['Representative_Depth_Increment'] = increments
        group['Representative_Top_Width'] = width
        group['Representative_Depth'] = depths
        group['Representative_Velocity'] = group['Mean_Velocity']
        group['Representative_Stage_Elevation'] = \
            group['Representative_Thalweg_Elevation'].to_numpy(dtype=np.float64) + depths
        group['Representative_Left_Station'] = -0.5 * width
        group['Representative_Right_Station'] = 0.5 * width
        groups.append(group)
    df = pd.concat(groups, ignore_index=True)[REPRESENTATIVE_CROSS_SECTION_COLUMNS]
    df[_INTEGER_COLUMNS] = df[_INTEGER_COLUMNS].astype(int)
    return df


def write_representative_cross_sections(df: pd.DataFrame | None, path: os.PathLike) -> pd.DataFrame:
    """Write the representative cross sections as CSV, rounded to 6 decimals (legacy
    save_representative_cross_section_file)."""
    df = pd.DataFrame(columns=REPRESENTATIVE_CROSS_SECTION_COLUMNS) if df is None else df.copy()
    if not df.empty:
        numeric = [c for c in df.columns if c not in _INTEGER_COLUMNS]
        df[numeric] = df[numeric].round(6)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOG.info('Finished writing ' + str(path))
    return df
