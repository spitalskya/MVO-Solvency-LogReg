from typing import Callable
import numpy as np
from scipy.optimize import OptimizeResult, minimize, approx_fprime
from typing import Literal, Any
from source.minimization_methods.minimization_in_direction import bisection, backtracking


def BFGS(obj_fun: Callable[[np.ndarray], float],
         grad: Callable[[np.ndarray], np.ndarray],
         x_0: np.ndarray, step: str) -> OptimizeResult:
    pass


def DFP(obj_fun: Callable[[np.ndarray], float],
        grad: Callable[[np.ndarray], np.ndarray],
        x_0: np.ndarray, step: Literal["optimal", "suboptimal"],
        args: tuple[Any]=(),
        callback: Callable[[np.ndarray], None] = None, **kwargs) -> OptimizeResult:
    """
    Minimizes the objective function using the DFP method.
    :param callback:
    :param obj_fun: Objective function.
    :param grad: Gradient of the objective function.
    :param x_0: Optimization starting point.
    :param step:
    :return:
    """

    step_optimizer: Callable
    if step == 'optimal':
        step_optimizer = bisection
    elif step == 'suboptimal':
        step_optimizer = backtracking

    if grad is None:
        def grad(x: np.ndarray, *args) -> np.ndarray:
            return approx_fprime(x, obj_fun, *args)

    g = grad(x_0, *args)

    H = np.identity(x_0.shape[0])
    x = x_0

    maxiter = kwargs.get("maxiter", 10000)
    tol = kwargs.get("tol", 1e-1)

    nfev: int = 0
    njev: int = 1
    it = 0
    trajectory: list[np.ndarray] = [x]

    for it in range(1, maxiter + 1):
        # TODO: replace optimizers
        # compute direction
        s = -H @ g

        # find optimal step size
        step_len = minimize(lambda step_size: obj_fun(x + step_size * s), np.zeros(1)).x[0]
        if step_len == 0:
            step_len = 10e-5

        # compute next x
        x_plus = x + step_len * s
        nfev += 1
        g_plus = grad(x_plus, *args)
        njev += 1
        trajectory.append(x_plus.copy())

        # callback
        if callback:
            callback(x_plus)

        # compute another Hessian
        p_k = x_plus - x
        y_k = g_plus - g
        H += (np.outer(p_k, p_k) / np.inner(p_k, y_k)) - ((H @ np.outer(y_k, y_k) @ H) / (y_k @ H @ y_k))

        # assign next x, g
        x = x_plus
        g = g_plus

        # compute gradient, end if less than tol
        if np.linalg.norm(g_plus) < tol:
            break

    if np.linalg.norm(g) < tol:
        msg, success = "Optimization successful", True
    else:
        msg, success = "Optimization not successful", False

    # TODO: you need nfev and njev
    return OptimizeResult(x=x, success=success, message=msg,
                          nit=it, nfev=nfev, njev=njev, trajectory=trajectory)
