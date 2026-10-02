"""What the figure modules share: the figure registry, the run outputs they read, and the plotting conventions.

Conventions
-----------
- Legacy ARC is drawn in vermillion and the new code in blue, everywhere.
- Maps are in metres east and north of the corner of the window shown, with equal scales, so a cell of the
  geographic rasters (about 23 m wide and 31 m tall at these sites) is drawn as the rectangle it is on the ground.
- Cross sections are in metres from the stream cell, negative to the left looking upstream (the new code's order).
  A legacy cross section's side 1 is drawn on the side the new code's side_one says legacy numbered side 1, which is
  the same side of the map. They're drawn between their banks, with a spacing or so either side.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402

import vc_runs  # noqa: E402

LEGACY = "#D55E00"
NEW = "#0072B2"
GROUND = "#6b6b6b"
WATER = "#56B4E9"
BANK = "#009E73"
ACCENT = "#E69F00"
PINK = "#CC79A7"
DPI = 150

plt.rcParams.update({
    "figure.dpi": DPI, "savefig.dpi": DPI, "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.grid": True,
    "grid.color": "#e3e3e3", "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "legend.frameon": False, "figure.constrained_layout.use": True,
})

REGISTRY: list[dict] = []


def figure(fid: str, title: str, section: str, *, needs_timing: bool = False, needs_fim: bool = False,
           needs_other: bool = False):
    """Register a figure function. It takes the Context and returns (matplotlib figure, record), where the record
    says what the figure shows ("caption") and what it found ("stats"). needs_timing, needs_fim and needs_other say
    it needs the timing runs, the FIM benchmark's results (fim_benchmark.py) or another ARC's runs (--other)."""
    def register(function):
        REGISTRY.append(dict(id=fid, title=title, section=section, function=function, needs_timing=needs_timing,
                             needs_fim=needs_fim, needs_other=needs_other))
        return function
    return register


def draw(entry: dict, context: "Context") -> dict:
    fig, record = entry["function"](context)
    path = context.figures / f"{entry['id']}.png"
    fig.savefig(path)
    plt.close(fig)
    return dict(title=entry["title"], section=entry["section"], file=path.name, **record)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def rounded(value, digits=3):
    """A number for the manifest, rounded, or None for NaN."""
    if value is None:
        return None
    value = float(value)
    return None if not math.isfinite(value) else round(value, digits)


# --- The runs --------------------------------------------------------------------------------------------------------


@dataclass
class Context:
    out: Path
    runs: Path
    sites_root: Path
    sites: list
    detail_sites: list
    timing: bool = True
    fim: Path | None = None  # the FIM benchmark's folder (fim_benchmark.py --out), if it has been run
    others: dict = field(default_factory=dict)  # other ARCs run as legacy (make_figures --other): name -> src
    _cache: dict = field(default_factory=dict)

    @cached_property
    def figures(self) -> Path:
        path = self.out / "figures"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _cached(self, key, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def capture(self, config: str, code: str, site: str):
        return self._cached(("capture", config, code, site), lambda: vc_runs.load_capture(self.runs, config, code,
                                                                                         site))

    def vdt(self, config: str, code: str, site: str) -> pd.DataFrame | None:
        def read():
            path = vc_runs.run_directory(self.runs, config, code, site) / "vdt.parquet"
            return pd.read_parquet(path) if path.exists() else None
        return self._cached(("vdt", config, code, site), read)

    def bathymetry(self, config: str, code: str, site: str) -> np.ndarray | None:
        def read():
            from arc.io import Raster
            path = vc_runs.run_directory(self.runs, config, code, site) / "bathy.tif"
            return Raster(path).read_array(np.float64) if path.exists() else None
        return self._cached(("bathy", config, code, site), read)

    def xs_file(self, config: str, code: str, site: str) -> pd.DataFrame | None:
        def read():
            path = vc_runs.run_directory(self.runs, config, code, site) / "xs.txt"
            return pd.read_csv(path, sep="\t") if path.exists() else None
        return self._cached(("xs", config, code, site), read)

    def seconds(self, config: str, code: str, site: str) -> float:
        path = vc_runs.run_directory(self.runs, config, code, site) / "done.json"
        return json.loads(path.read_text())["seconds"] if path.exists() else math.nan

    def configs(self, site: str):
        def make():
            from arc.config import Configs
            inputs = vc_runs.site_inputs(self.sites_root, site, self.out / "scratch" / site, {}, ())
            return Configs.from_mapping(inputs)
        return self._cached(("configs", site), make)

    def grid(self, site: str):
        def make():
            from arc import pipeline
            return pipeline.read_grid(self.configs(site))
        return self._cached(("grid", site), make)

    def site_label(self, site: str) -> str:
        return site.replace("_", " ")

    def merged_vdt(self, config: str, site: str) -> pd.DataFrame | None:
        """Legacy's and the new VDT rows for the same cells (the cross section's centre), suffixed _l and _n."""
        legacy, new = self.vdt(config, "legacy", site), self.vdt(config, "new", site)
        if legacy is None or new is None:
            return None
        return legacy.merge(new, on=["COMID", "Row", "Col"], suffixes=("_l", "_n"))


def increments(df: pd.DataFrame, suffix: str = "") -> int:
    return len([c for c in df.columns if c.startswith("q_") and c.endswith(suffix)])


# --- Maps ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """Rows r0..r1 and columns c0..c1 of a raster, drawn in metres east and north of the window's lower left."""
    r0: int
    r1: int
    c0: int
    c1: int
    dx: float
    dy: float

    @classmethod
    def around(cls, rows, cols, dx, dy, pad_metres=150.0, shape=None):
        rows, cols = np.asarray(rows), np.asarray(cols)
        pr, pc = int(math.ceil(pad_metres / dy)), int(math.ceil(pad_metres / dx))
        r0, r1, c0, c1 = rows.min() - pr, rows.max() + pr, cols.min() - pc, cols.max() + pc
        if shape is not None:
            r0, c0 = max(r0, 0), max(c0, 0)
            r1, c1 = min(r1, shape[0] - 1), min(c1, shape[1] - 1)
        return cls(int(r0), int(r1), int(c0), int(c1), float(dx), float(dy))

    @property
    def extent(self):
        return (-0.5 * self.dx, (self.c1 - self.c0 + 0.5) * self.dx, -0.5 * self.dy, (self.r1 - self.r0 + 0.5) * self.dy)

    def xy(self, rows, cols):
        """Map coordinates of fractional rows and columns."""
        return (np.asarray(cols, dtype=float) - self.c0) * self.dx, (self.r1 - np.asarray(rows, dtype=float)) * self.dy

    def clip(self, raster: np.ndarray) -> np.ndarray:
        return raster[self.r0:self.r1 + 1, self.c0:self.c1 + 1]

    def contains(self, rows, cols) -> np.ndarray:
        rows, cols = np.asarray(rows), np.asarray(cols)
        return (rows >= self.r0) & (rows <= self.r1) & (cols >= self.c0) & (cols <= self.c1)


def show_raster(ax, window: Window, raster: np.ndarray, **kwargs):
    """A raster window on its true cell shape (see the module notes)."""
    image = ax.imshow(window.clip(raster), extent=window.extent, origin="upper", interpolation="nearest",
                      **kwargs)
    ax.set_aspect("equal")
    ax.set_xlabel("metres east")
    ax.set_ylabel("metres north")
    ax.grid(False)
    return image


def hillshade(ax, window: Window, dem: np.ndarray, alpha: float = 1.0, cmap="gray"):
    values = window.clip(dem).astype(float)
    values = np.where(values >= 9999.0, np.nan, values)
    fill = np.nanmedian(values) if np.isfinite(values).any() else 0.0
    shade = LightSource(azdeg=315, altdeg=45).hillshade(np.nan_to_num(values, nan=fill), vert_exag=3.0,
                                                          dx=window.dx, dy=window.dy)
    return show_raster(ax, window, _placed(window, shade), cmap=cmap, alpha=alpha, vmin=0, vmax=1)


def _placed(window: Window, clipped: np.ndarray) -> np.ndarray:
    """A clipped array as a full-size raster whose window holds it, for show_raster."""
    full = np.full((window.r1 + 1, window.c1 + 1), np.nan)
    full[window.r0:, window.c0:] = clipped
    return full


def cell_outlines(ax, window: Window, rows, cols, **kwargs):
    """Rectangles round cells, to show their true shape."""
    from matplotlib.patches import Rectangle
    for r, c in zip(np.atleast_1d(rows), np.atleast_1d(cols)):
        x, y = window.xy(r, c)
        ax.add_patch(Rectangle((x - 0.5 * window.dx, y - 0.5 * window.dy), window.dx, window.dy, fill=False,
                               **kwargs))


def scale_bar(ax, length: float, label: str | None = None):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x, y = x0 + 0.05 * (x1 - x0), y0 + 0.05 * (y1 - y0)
    ax.plot([x, x + length], [y, y], color="k", lw=2, solid_capstyle="butt")
    ax.text(x + 0.5 * length, y + 0.015 * (y1 - y0), label or f"{length:g} m", ha="center", va="bottom", fontsize=7)


# --- Cross sections --------------------------------------------------------------------------------------------------


def new_stations(size: int, spacing: float) -> np.ndarray:
    return (np.arange(size) - size // 2) * spacing


def legacy_stations(side1: np.ndarray, side2: np.ndarray, spacing: float, side_one: int = 1):
    """A legacy cross section's two sides, which each run out from the stream cell, as one profile from left to
    right: stations and elevations."""
    stations = np.concatenate([-side_one * spacing * np.arange(side2.size)[:0:-1],
                               side_one * spacing * np.arange(side1.size)])
    values = np.concatenate([side2[:0:-1], side1])
    if side_one < 0:
        stations, values = stations[::-1], values[::-1]
    return stations, values


def between_banks(ax, left: float, right: float, spacing: float, curves=(), margin: float = 1.5, pad: float = 0.15):
    """Show the cross section from margin spacings left of the left bank to as far right of the right bank (left
    and right are distances from the stream cell), with the y limits fitted to the curves there."""
    lo, hi = -left - margin * spacing, right + margin * spacing
    ax.set_xlim(lo, hi)
    values = []
    for x, y in curves:
        x, y = np.asarray(x, float), np.asarray(y, float)
        inside = (x >= lo) & (x <= hi) & np.isfinite(y) & (y < 9000)
        values.append(y[inside])
        # the curve where it crosses the window's edges
        for edge in (lo, hi):
            if x.size > 1 and x.min() <= edge <= x.max():
                values.append(np.interp([edge], x, y))
    values = np.concatenate(values) if values else np.array([])
    if values.size:
        y0, y1 = values.min(), values.max()
        span = max(y1 - y0, 0.5)
        ax.set_ylim(y0 - pad * span, y1 + pad * span)
    return lo, hi


def mark_banks(ax, left: float, right: float, color=BANK, label: str | None = "banks", **kwargs):
    for k, x in enumerate((-left, right)):
        ax.axvline(x, color=color, lw=1.0, ls=kwargs.pop("ls", "--") if k == 0 else "--",
                   label=label if k == 0 else None, **kwargs)


def water(ax, stations, elevations, wse, color=WATER, alpha=0.35, label=None):
    """Shade the water under a water surface, from the stream cell out to where the ground rises above it."""
    x = np.asarray(stations, float)
    y = np.asarray(elevations, float)
    fine = np.linspace(x.min(), x.max(), 4000)
    ground = np.interp(fine, x, y)
    center = np.argmin(np.abs(fine))
    wet = np.zeros(fine.size, bool)
    for step in (1, -1):
        j = center
        while 0 <= j < fine.size and ground[j] < wse:
            wet[j] = True
            j += step
    ax.fill_between(fine, ground, wse, where=wet, color=color, alpha=alpha, lw=0, label=label)


def cdf(ax, values, color, label, **kwargs):
    values = np.sort(np.asarray(values, float)[np.isfinite(values)])
    if values.size == 0:
        return
    ax.plot(values, np.arange(1, values.size + 1) / values.size, color=color, label=label, **kwargs)


def note(ax, text: str, loc: str = "upper left", **kwargs):
    x, ha = (0.02, "left") if "left" in loc else (0.98, "right")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=kwargs.pop("fontsize", 7.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9), **kwargs)
