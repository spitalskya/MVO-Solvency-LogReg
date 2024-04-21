from typing import Callable
import numpy as np
from scipy.optimize import OptimizeResult, minimize


def BFGS(obj_fun: Callable[[np.ndarray], float],
         grad: Callable[[np.ndarray], np.ndarray],
         x_0: np.ndarray, step: str) -> OptimizeResult:
     pass


def DFP(obj_fun: Callable[[np.ndarray], float],
        grad: Callable[[np.ndarray], np.ndarray],
        x_0: np.ndarray, step: str) -> OptimizeResult:
    # TODO: Docstring here

    maxiter = 1000
    tol = 1e-3
    x = x_0
    g = grad(x_0)

    H = np.identity(x_0.shape[0])

    for it in range(1, maxiter + 1):

        # compute direction
        s = -H @ g

        # find optimal step size
        step_len = minimize(lambda step_size: obj_fun(x + step_size * s), np.zeros(x_0.shape)).x[0]

        # compute next x
        x = x + step_len * s

        # compute gradient, end if less than tol
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break

        # compute another Hessian
        s_k = step_len * s
        y_k = grad(x) - g
        z_k = H @ y_k

        H = H + (np.outer(s_k, s_k) / np.inner(s_k, y_k)) - (np.outer(z_k, z_k) / np.inner(y_k, z_k))

    if np.linalg.norm(g) < tol:
        msg, success = "Optimization successful", True
    else:
        msg, success = "Optimization not successful", False

    # TODO: you need nfev and njev
    return OptimizeResult(x=x, success=success, message=msg,
                          nit=it)


if __name__ == "__main__":
    pass
