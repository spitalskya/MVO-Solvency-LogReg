from typing import Callable
import numpy as np
from scipy.optimize import OptimizeResult


def backtracking(obj_fun: Callable[[np.ndarray], float],
                 grad: Callable[[np.ndarray], np.ndarray],
                 x_0: np.ndarray, s: np.ndarray) -> OptimizeResult:
    pass


def bisection(obj_fun: Callable[[np.ndarray], float],
              grad: Callable[[np.ndarray], np.ndarray],
              x_0: np.ndarray, s: np.ndarray) -> OptimizeResult:
    pass
