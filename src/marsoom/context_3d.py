import numpy as np
from dataclasses import dataclass

@dataclass
class Context3D:
    world2projT: np.ndarray
    camera_positon: np.ndarray

