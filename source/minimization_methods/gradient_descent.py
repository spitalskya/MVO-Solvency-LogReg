from typing import Callable
import numpy as np
from scipy.optimize import OptimizeResult


def optimalStep(obj_fun: Callable[[np.ndarray], float],
                grad: Callable[[np.ndarray], np.ndarray],
                x_0: np.ndarray) -> OptimizeResult:
    pass


def constantStep(obj_fun: Callable[[np.ndarray], float],
                 grad: Callable[[np.ndarray], np.ndarray],
                 x_0: np.ndarray) -> OptimizeResult:
    pass
