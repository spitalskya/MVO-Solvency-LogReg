from typing import Callable
import numpy as np
from scipy.optimize import OptimizeResult


def BFGS(obj_fun: Callable[[np.ndarray], float],
         grad: Callable[[np.ndarray], np.ndarray],
         x_0: np.ndarray, step: str) -> OptimizeResult:
     pass


def DFP(obj_fun: Callable[[np.ndarray], float],
        grad: Callable[[np.ndarray], np.ndarray],
        x_0: np.ndarray, step: str) -> OptimizeResult:
     pass


if __name__ == "__main__":
    pass
