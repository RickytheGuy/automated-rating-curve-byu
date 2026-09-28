"""Variants of the new pipeline for the experiments, applied by patching arc.pipeline for the length of a run.

Settings (all optional):
- test_depth: the angle search's depth above the stream cell (the pipeline's is 5 m since 2026-09-26; legacy's
  0.5 m).
- angle_rule: "width" (the pipeline's) or "area" (the smallest flow area below the test depth, which weighs the
  widths at every depth up to it). test_depths: several depths, each candidate scored by the sum of its width (or
  area) at each over the smallest there among the candidates.
- bed_grade: the bed cap's grade, or None for no cap (the pipeline's is MAX_SLOPE since 2026-09-26; legacy's 0.01).
- bank_smoothing: "legacy" for legacy's lower envelope (the pipeline's until 2026-09-26).
- bank_ceiling: "observed" or "main_stem", legacy's network step with another rule for which inflows count
  (vc_smoothing).
- bank_reference: "clamp", "local", "local_median", or the new smoothings "falling_quantile" and
  "water_plus_height", the level the channel is carved below (bank_reference), with bank_quantile (0.25) and
  bank_half_window (500 m) for the new two, and bank_falling ("both", "water" or "none") for water_plus_height.
- bed_at_most_stream: a test only, not a proposal (the user ruled that bank-based bathymetry may raise the DEM):
  each channel carved at least down to its stream cell, so no bed is above the DEM's water.
- pivot: "each" for the cross section pivoting to the narrowest candidate at every increment of its rating curve,
  or "top" for the candidate narrowest at the (pivoted) top for the whole curve (vc_pivot).
- configs: {attribute: value} set on the Configs for the run.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial

import numpy as np

import vc_search

# the bank_reference rules that change legacy's smoothing (the pipeline's is water_plus_height since 2026-09-26)
LEGACY_BASED = ("clamp", "local", "local_median", "falling_quantile")


def angle_search(settings: dict):
    """The pipeline's narrowest_direction for these settings, or None to keep the pipeline's own."""
    rule = settings.get("angle_rule", "width")
    depths = settings.get("test_depths")
    if depths is None and "test_depth" in settings:
        depths = [settings["test_depth"]]
    if depths is None and rule == "width":
        return None
    depths = list(depths or [0.5])
    use_area = rule == "area"

    def narrowest_direction(dem, row, col, stream_direction, cross_section_length, dx, dy, offsets, test_depth=None):
        return vc_search.narrowest_by(dem, row, col, stream_direction, cross_section_length, dx, dy, offsets, depths,
                                      use_area)
    return narrowest_direction


def _running_median(values, order, window=5):
    """A running median of a reach's values along its order (NaN left out), NaN where the window has none."""
    ordered = np.asarray(values, dtype=np.float64)[order]
    out = np.full(ordered.size, np.nan)
    half = window // 2
    for i in range(ordered.size):
        part = ordered[max(i - half, 0):i + half + 1]
        part = part[np.isfinite(part)]
        if part.size:
            out[i] = np.median(part)
    result = np.full(ordered.size, np.nan)
    result[order] = out
    return result


def _falling_fit(values, quantile=None):
    """The non-increasing sequence closest to values (in order, NaN left out): least squares, or with a quantile
    the quantile of each run of values it pools (pool-adjacent-violators). NaN where values are NaN."""
    values = np.asarray(values, dtype=np.float64)
    known = np.flatnonzero(np.isfinite(values))
    blocks = []  # [value, members]
    for k in known:
        blocks.append([values[k], [values[k]]])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:  # a rise: pool them
            members = blocks[-2][1] + blocks[-1][1]
            level = float(np.quantile(members, quantile)) if quantile is not None else float(np.mean(members))
            blocks[-2:] = [[level, members]]
    fitted = np.full(values.size, np.nan)
    fitted[known] = np.concatenate([[b[0]] * len(b[1]) for b in blocks]) if blocks else []
    return fitted


def _fill_along(values, stations):
    """values (in order) with their gaps filled linearly by station, and held flat beyond the ends."""
    known = np.isfinite(values)
    if not known.any():
        return values
    return np.interp(stations, stations[known], values[known])


def _running_quantile(values, stations, quantile, half_window):
    """Each position's quantile of the values within half_window metres of it along the reach (NaN left out), NaN
    where there are none."""
    out = np.full(values.size, np.nan)
    for i in range(values.size):
        near = np.abs(stations - stations[i]) <= half_window
        part = values[near]
        part = part[np.isfinite(part)]
        if part.size:
            out[i] = np.quantile(part, quantile)
    return out


def _running_median_along(values, stations, half_window):
    """Each position's median of the values within half_window metres of it along the reach."""
    out = np.full(values.size, np.nan)
    for i in range(values.size):
        part = values[np.abs(stations - stations[i]) <= half_window]
        part = part[np.isfinite(part)]
        if part.size:
            out[i] = np.median(part)
    return out


def bank_reference(original, rule: str, quantile: float = 0.25, half_window: float = 500.0, falling: str = "both"):
    """smooth_bank_elevations, with each reach's bank elevations then replaced by another reference level:
    - "clamp": the smoothed bank elevation, but never below the stream cell;
    - "local": each cross section's own observation, or the smoothed value where it has none;
    - "local_median": a running median of five cross sections' observations along the reach, or the smoothed value;
    - "falling_quantile": a new smoothing. The level falling downstream that best fits the reach's observations,
      each run of them pooled at their quantile (a low one, since single cells observe the neighbouring ground and
      width-to-depth banks the valley's shoulders); without observations, the smoothed value;
    - "water_plus_height": a new smoothing. The DEM's water surface along the reach, fitted to fall downstream, plus
      the quantile of the observations' heights above their stream cells within half_window metres (the reach's, and
      failing that the site's, where there are none near), fitted to fall downstream again;
    each also never below the stream cell. The new two don't use the network's outlets, so no inflow caps a reach."""
    def smooth_bank_elevations(network, reaches, dx, dy):
        smoothed = original(network, reaches, dx, dy)
        heights_everywhere = []
        prepared = {}
        for reach, result in smoothed.items():
            thalwegs = np.array([float(xs.elevations[xs.elevations.size // 2]) for xs in reaches[reach].sections])
            observations = np.asarray(result.observations, dtype=np.float64)
            heights = observations - thalwegs
            heights[~(heights > 0.0)] = np.nan
            heights_everywhere.append(heights[np.isfinite(heights)])
            prepared[reach] = (result, thalwegs, observations, heights)
        everywhere = np.concatenate(heights_everywhere) if heights_everywhere else np.array([])
        site_height = float(np.quantile(everywhere, quantile)) if everywhere.size else np.nan
        out = {}
        for reach, (result, thalwegs, observations, heights) in prepared.items():
            bank = np.asarray(result.bank_elevations, dtype=np.float64).copy()
            order, stations = np.asarray(result.order), np.asarray(result.stations, dtype=np.float64)
            if rule == "local":
                bank = np.where(np.isfinite(observations), observations, bank)
            elif rule == "local_median":
                median = _running_median(observations, order)
                bank = np.where(np.isfinite(median), median, bank)
            elif rule == "falling_quantile":
                fitted = _falling_fit(observations[order], quantile)
                if np.isfinite(fitted).any():
                    bank[order] = _fill_along(fitted, stations)
            elif rule == "water_plus_height":
                # falling: "both" (the water surface and then the bank elevation fitted to fall downstream),
                # "water" (only the water surface), or "none" (a running median of the stream cells instead)
                if falling in ("both", "water"):
                    water = _fill_along(_falling_fit(thalwegs[order]), stations)
                else:
                    water = _running_median_along(thalwegs[order], stations, half_window)
                height = _running_quantile(heights[order], stations, quantile, half_window)
                reach_height = np.nanquantile(heights, quantile) if np.isfinite(heights).any() else site_height
                height = np.where(np.isfinite(height), height, reach_height)
                if np.isfinite(height).all():
                    bank[order] = _falling_fit(water + height) if falling == "both" else water + height
            elif rule != "clamp":
                raise ValueError(f"unknown bank_reference {rule!r}")
            finite = np.isfinite(bank)
            bank[finite] = np.maximum(bank[finite], thalwegs[finite])
            out[reach] = result._replace(bank_elevations=bank)
        return out
    return smooth_bank_elevations


@contextmanager
def applied(settings: dict, configs=None):
    """Patch arc.pipeline (and the configs, if given) for these settings, and put everything back afterwards."""
    from arc import pipeline
    patches = []

    def patch(owner, name, value):
        patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    search = angle_search(settings)
    if search is not None:
        patch(pipeline, "narrowest_direction", search)
    if "bed_grade" in settings:
        grade = settings["bed_grade"]
        original = pipeline.smooth_channel_depths

        def smooth_channel_depths(*args, **kwargs):
            kwargs["max_bed_grade"] = grade
            return original(*args, **kwargs)
        patch(pipeline, "smooth_channel_depths", smooth_channel_depths)
    if settings.get("bank_smoothing") == "legacy":  # legacy's lower envelope along the network
        patch(pipeline, "smooth_bank_elevations", partial(pipeline.smooth_bank_elevations, method="legacy"))
    if settings.get("bank_ceiling"):  # a copy of legacy's network step, with another rule for the inflows
        import vc_smoothing
        patch(pipeline, "smooth_bank_elevations", vc_smoothing.smoother(settings))
        if settings["bank_ceiling"] == "main_stem":
            patch(pipeline, "stream_network", vc_smoothing.network_with_areas(pipeline.stream_network))
    if settings.get("bank_reference"):
        rule = settings["bank_reference"]
        base = pipeline.smooth_bank_elevations
        if rule in LEGACY_BASED and not settings.get("bank_ceiling") and settings.get("bank_smoothing") != "legacy":
            base = partial(base, method="legacy")  # these change legacy's smoothing
        patch(pipeline, "smooth_bank_elevations", bank_reference(
            base, rule, settings.get("bank_quantile", 0.25), settings.get("bank_half_window", 500.0),
            settings.get("bank_falling", "both")))
    if settings.get("pivot"):  # the cross section pivoting as the rating curve rises ("each" or "top", vc_pivot)
        import vc_pivot
        for owner, name, value in vc_pivot.patches(pipeline, settings["pivot"]):
            patch(owner, name, value)
    if settings.get("bed_at_most_stream"):  # a test only: no bed above the stream cell, so no channel is filled
        carve = pipeline.carve_channel

        def carve_channel(xs, banks, depth, **kwargs):
            bank = kwargs.get("bank_elevation")
            if bank is not None:
                depth = max(float(depth), float(bank) - float(xs.elevations[xs.elevations.size // 2]))
            return carve(xs, banks, depth, **kwargs)
        patch(pipeline, "carve_channel", carve_channel)
    if configs is not None:
        for key, value in settings.get("configs", {}).items():
            patches.append((configs, key, getattr(configs, key)))
            setattr(configs, key, value)
    try:
        yield
    finally:
        for owner, name, value in reversed(patches):
            setattr(owner, name, value)
