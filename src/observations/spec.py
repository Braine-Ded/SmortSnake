from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationSpec:
    """Shape of the arrays an encoder produces: (11,) for features, (C, H, W) for planes."""

    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))
