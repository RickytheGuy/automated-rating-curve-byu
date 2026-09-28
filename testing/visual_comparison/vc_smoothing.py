"""Variants of the bank smoothing's network step (arc.bathymetry.smoothing), for the experiments.

The pipeline's network step takes each reach's lowest inflow as its ceiling and the start of its grade, whether or
not that inflow had any observation of its own. An inflow without one gets its outlet elevation extrapolated from
the reaches downstream of it, at their grade or the minimum grade, so a reach below it is held to about its own
outlet elevation all the way up (Du Page: the reach's bank elevation 5.5 m under its stream cells). The variants
choose which inflows count, everywhere the network step and the reach surface use the lowest inflow:

- "observed": only inflows with observations of their own; with none, the reach is treated like a headwater.
- "main_stem": only the inflow with the largest drainage area among those with observations.

Otherwise these are copies of network_outlet_elevations and smooth_bank_elevations.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np

from arc.bathymetry.smoothing import (MIN_GRADE, SmoothedReach, _fill_path, _length, filter_reach_widths,
                                      order_reach, reach_bank_observations, reach_bank_surface, reach_baseline)


def _counted_inflows(network, reach, outlets, minima, rule):
    inflows = [p for p in network.predecessors(reach) if p in outlets]
    if rule in ("observed", "main_stem"):
        inflows = [p for p in inflows if p in minima]
    if rule == "main_stem" and inflows:
        inflows = [max(inflows, key=lambda p: network.nodes[p].get("area", 0.0))]
    return inflows


def network_outlet_elevations(network: nx.DiGraph, minima, maxima, rule: str):
    if len(minima) == 0:
        return {}, {}
    candidates: dict[int, list[float]] = {}
    for reach, elevation in minima.items():
        if reach in network and math.isfinite(elevation):
            candidates.setdefault(reach, []).append(float(elevation))
    headwaters = [reach for reach in network if network.in_degree(reach) == 0] or list(network)
    unconnected = {reach for reach in network if network.in_degree(reach) == 0 and network.out_degree(reach) == 0}
    for headwater in headwaters:
        path, visited, reach = [headwater], set(), headwater
        while network.out_degree(reach) > 0 and reach not in visited:
            visited.add(reach)
            reach = next(iter(network.successors(reach)))
            path.append(reach)
        for reach, elevation in zip(path, _fill_path(network, path, minima)):
            if math.isfinite(elevation):
                candidates.setdefault(reach, []).append(float(elevation))
    outlets = {reach: min(values) for reach, values in candidates.items()}

    def lowest_inflow(reach):
        inflows = _counted_inflows(network, reach, outlets, minima, rule)
        return min(outlets[p] for p in inflows) if inflows else None

    for _ in range(max(network.number_of_nodes(), 1)):
        changed = False
        for reach in network:
            inflow = lowest_inflow(reach)
            if reach in outlets and inflow is not None:
                highest = inflow - MIN_GRADE * _length(network, reach)
                if outlets[reach] > highest:
                    outlets[reach] = highest
                    changed = True
        if not changed:
            break

    grades: dict[int, float] = {}
    for reach, outlet in outlets.items():
        inflow = lowest_inflow(reach)
        if inflow is not None:
            grades[reach] = max((inflow - outlet) / _length(network, reach), MIN_GRADE)
        elif reach in minima and reach in maxima and network.in_degree(reach) > 0:
            # its inflows don't count: like a headwater, from its highest observation to its lowest
            grades[reach] = max((maxima[reach] - minima[reach]) / _length(network, reach), MIN_GRADE)

    for reach in network:
        if network.out_degree(reach) != 0 or reach in unconnected:
            continue
        inflow = lowest_inflow(reach)
        length = _length(network, reach)
        if inflow is None:
            grades.setdefault(reach, MIN_GRADE)
        elif math.isfinite(minima.get(reach, math.nan)):
            outlets[reach] = min(float(minima[reach]), inflow - MIN_GRADE * length)
            grades[reach] = max((inflow - outlets[reach]) / length, MIN_GRADE)
        else:
            inflow_grades = [grades[p] for p in network.predecessors(reach) if p in grades]
            grades[reach] = max(float(np.nanmedian(inflow_grades)), MIN_GRADE) if inflow_grades else MIN_GRADE
            outlets[reach] = inflow - grades[reach] * length

    for reach in unconnected:
        lowest, highest = minima.get(reach, math.nan), maxima.get(reach, math.nan)
        if math.isfinite(lowest):
            fall = highest - lowest if math.isfinite(highest) else 0.0
            grades[reach] = max(fall / _length(network, reach), MIN_GRADE)
            outlets[reach] = float(lowest)

    for headwater in headwaters:
        if headwater in unconnected:
            continue
        lowest, highest = minima.get(headwater, math.nan), maxima.get(headwater, math.nan)
        if math.isfinite(lowest) and math.isfinite(highest):
            grades[headwater] = max((highest - lowest) / _length(network, headwater), MIN_GRADE)
        else:
            downstream_grades = [grades[s] for s in network.successors(headwater) if s in grades]
            grades[headwater] = max(float(np.nanmedian(downstream_grades)), MIN_GRADE) if downstream_grades \
                else MIN_GRADE
    return outlets, grades


def smooth_bank_elevations(network, reaches, dx, dy, rule: str):
    cells = {reach: (np.asarray(r.rows, dtype=np.int64), np.asarray(r.cols, dtype=np.int64))
             for reach, r in reaches.items() if len(r.sections) > 0}
    prepared = {}
    for reach, sections in reaches.items():
        if len(sections.sections) == 0:
            continue
        banks, widths = filter_reach_widths(sections.sections, sections.banks)
        observations, lower, upper = reach_bank_observations(sections.sections, banks)
        order, stations = order_reach(network, reach, sections.rows, sections.cols, dx, dy, cells, observations)
        thalwegs = np.array([float(xs.elevations[xs.elevations.size // 2]) for xs in sections.sections])
        prepared[reach] = (banks, widths, observations, lower, upper, order, stations, thalwegs)
    finite = {reach: p[2][np.isfinite(p[2])] for reach, p in prepared.items()}
    minima = {reach: float(values.min()) for reach, values in finite.items() if values.size > 0}
    maxima = {reach: float(values.max()) for reach, values in finite.items() if values.size > 0}
    outlets, grades = network_outlet_elevations(network, minima, maxima, rule)
    smoothed = {}
    for reach, (banks, widths, observations, lower, upper, order, stations, thalwegs) in prepared.items():
        bank_elevations = np.full(observations.size, math.nan)
        anchors = np.zeros(observations.size, dtype=bool)
        if reach in outlets:
            baseline, fractions = reach_baseline(network, reach, stations, outlets, grades)
            inflows = [outlets[p] for p in _counted_inflows(network, reach, outlets, minima, rule)]
            surface, _, ordered_anchors = reach_bank_surface(
                observations[order], baseline, fractions, _length(network, reach), outlets[reach],
                ceiling=min(inflows) if inflows else None, thalwegs=thalwegs[order], lower=lower, upper=upper)
            bank_elevations[order] = surface
            anchors[order] = ordered_anchors
        smoothed[reach] = SmoothedReach(banks, bank_elevations, observations, anchors, order, stations, widths)
    return smoothed


def smoother(settings: dict):
    rule = settings["bank_ceiling"]

    def smooth(network, reaches, dx, dy):
        return smooth_bank_elevations(network, reaches, dx, dy, rule)
    return smooth


def network_with_areas(original):
    """The pipeline's stream_network, with each reach's drainage area as its "area" (for main_stem)."""
    def stream_network(configs, layer):
        network = original(configs, layer)
        if network is None or layer is None or not configs.drainage_area_field \
                or configs.drainage_area_field not in layer.columns:
            return network
        for reach, area in zip(layer[configs.reach_id], layer[configs.drainage_area_field]):
            try:
                reach = int(reach)
            except (TypeError, ValueError):
                continue
            if reach in network and "area" not in network.nodes[reach]:
                network.nodes[reach]["area"] = float(area) if area is not None else 0.0
        return network
    return stream_network
