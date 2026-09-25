from dataclasses import dataclass, field

import numpy as np

@dataclass(eq=False, repr=False)
class XSection:
    elevations: np.ndarray
    mannings_n: np.ndarray
    ordinate_distance: float

    # The following fields are derived later
    left_bank_distance: float = field(init=False, default=-1.0)
    right_bank_distance: float = field(init=False, default=-1.0)

