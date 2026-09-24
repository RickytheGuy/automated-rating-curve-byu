from dataclasses import dataclass

import numpy as np

@dataclass
class XSection:
    elevations: np.ndarray
    mannings_n: np.ndarray
    ordinate_distance: float