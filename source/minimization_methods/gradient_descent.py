from typing import Callable
import numpy as np
from scipy.optimize import OptimizeResult


def optimalStep(obj_fun: Callable[[np.ndarray], float],
                grad: Callable[[np.ndarray], np.ndarray],
                x_0: np.ndarray, options={}) -> OptimizeResult:
    """_summary_

    Args:
        obj_fun (Callable[[np.ndarray], float]): _description_
        grad (Callable[[np.ndarray], np.ndarray]): _description_
        x_0 (np.ndarray): _description_

    Returns:
        OptimizeResult: _description_
    """
    pass
    


def constantStep(obj_fun: Callable[[np.ndarray], float],
                 grad: Callable[[np.ndarray], np.ndarray],
                 x_0: np.ndarray, options= {}) -> OptimizeResult:
    """_summary_

    Args:
        obj_fun (Callable[[np.ndarray], float]): _description_
        grad (Callable[[np.ndarray], np.ndarray]): _description_
        x_0 (np.ndarray): _description_

    Returns:
        OptimizeResult: _description_
    """
    tol:float = options.get("tol", 1e-6)
    maxiter:int = options.get("maxiter", 1000)
    gamma = ...
    x_k = x_0
    while True:
        x_k1 = x_k - gamma*grad(x_k)
        
    pass
