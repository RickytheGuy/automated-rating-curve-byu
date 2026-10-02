"""Writing ARC's tables (the VDT and AP databases, the curve files and the representative cross sections) as CSV or
Parquet.

Parquet is written with brotli at level 5, and dictionary encoding only on the columns where under 30% of the values
differ. pyarrow's default gives every column a dictionary until it passes 1 MB, about 131,000 distinct numbers, which
a 1° tile's VDT never reaches. A column whose values mostly differ (water surface elevations, top widths, most
discharges) then stores every value plus an index for every row. Benchmarked on the N14W089 tile, four more tiles'
VDTs and the 57 FIM sites (2026-10-01), these settings make the VDT 8–25% smaller than brotli 8 with pyarrow's
dictionaries, ARC's old setting, and two to six times faster to write. Brotli's levels 7 and up pay a setup cost
for every column, which dominates on small files. The files hold the same columns and values either way.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DICTIONARY_FRACTION = 0.3  # a column is dictionary-encoded when fewer than this fraction of its values differ
BROTLI_LEVEL = 5


def write_table(df: pd.DataFrame, path: os.PathLike) -> None:
    """Write a table as CSV or, for a .parquet path, Parquet (see the notes above)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith('.parquet'):
        repeated = [col for col in df.columns if df[col].nunique() < DICTIONARY_FRACTION * len(df)]
        df.to_parquet(path, index=False, compression='brotli', compression_level=BROTLI_LEVEL,
                      use_dictionary=repeated)
    else:
        df.to_csv(path, index=False)
