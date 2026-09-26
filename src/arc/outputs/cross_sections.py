"""The cross-section file (XS_Out_File): each stream cell's cross section, as legacy HydraulicData's
save_cross_section_file wrote it.

The file is tab-separated, with a row per cross section. Its profiles and Manning's n run out from the stream cell
on each side, the two sides as legacy ARC numbered them, and are written as numpy prints arrays, as legacy wrote
them with np.array2string(..., precision=6, floatmode='fixed').
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from arc import LOG

XS_EXPORT_COLUMNS = ['COMID', 'Row', 'Col', 'XS1_Profile', 'Ordinate_Dist', 'Manning_N_Raster1', 'XS2_Profile',
                     'Manning_N_Raster2', 'r1', 'c1', 'r2', 'c2']


class CrossSectionRecord(NamedTuple):
    """One row of the cross-section file. The profiles and n run out from the stream cell, which starts both, and
    (r1, c1) and (r2, c2) are the cells at the end of each side."""
    comid: int
    row: int
    col: int
    xs1_profile: np.ndarray
    ordinate_dist: float
    manning_n_raster1: np.ndarray
    xs2_profile: np.ndarray
    manning_n_raster2: np.ndarray
    r1: int
    c1: int
    r2: int
    c2: int


def format_array(values: np.ndarray) -> str:
    """The text np.array2string(values, precision=6, max_line_width=np.inf, threshold=np.inf, floatmode='fixed')
    gives, as legacy ARC wrote its cross sections, built directly for a 1-D float array without exponents.

    np.array2string took most of legacy's time writing the file. Every number is rounded to 6 decimals and padded
    on the left to the widest whole part, as numpy does. Arrays numpy would print with exponents (values of 1e8 or
    more, or a smallest non-zero value below 0.0001 or under a thousandth of the largest), or without any finite
    values, go to np.array2string itself.
    """
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.floating) or values.size == 0:
        return _array2string(values)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size == 0:
        return _array2string(values)
    magnitudes = np.abs(finite_values[finite_values != 0])
    if magnitudes.size:
        largest, smallest = magnitudes.max(), magnitudes.min()
        with np.errstate(over='ignore'):
            if largest >= 10.0 ** min(8, np.finfo(values.dtype).precision) or smallest < 0.0001 \
                    or largest / smallest > 1000.0:
                return _array2string(values)
    texts = ['%.6f' % x for x in values.tolist()]
    whole = max(len(text) - 7 for text, is_finite in zip(texts, finite) if is_finite)  # the part before '.'
    width = whole + 7
    parts = []
    for text, is_finite in zip(texts, finite):
        if not is_finite:
            text = 'nan' if text == 'nan' else ('-inf' if text.startswith('-') else 'inf')
        parts.append(text.rjust(width))
    return '[' + ' '.join(parts) + ']'


def _array2string(values: np.ndarray) -> str:
    return np.array2string(values, precision=6, max_line_width=np.inf, threshold=np.inf, floatmode='fixed')


def cross_section_dataframe(records: list[CrossSectionRecord]) -> pd.DataFrame:
    """The cross-section file's table, with the arrays as text."""
    records = [record for record in records if record is not None]
    df = pd.DataFrame([record._asdict() for record in records])
    if df.empty:
        return pd.DataFrame(columns=XS_EXPORT_COLUMNS)
    df.columns = XS_EXPORT_COLUMNS
    for col in df.columns:
        df[col] = df[col].apply(lambda x: format_array(x) if isinstance(x, np.ndarray) else x)
    return df


def write_cross_sections(records: list[CrossSectionRecord], path: os.PathLike) -> pd.DataFrame:
    """Write the cross-section file, tab-separated, and return its table."""
    df = cross_section_dataframe(records)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep='\t')
    LOG.info('Finished writing ' + str(path))
    return df
