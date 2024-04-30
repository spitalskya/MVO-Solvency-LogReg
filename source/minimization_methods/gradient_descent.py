from typing import Callable
import numpy as np
from scipy.optimize import OptimizeResult, approx_fprime
from minimization_methods.minimization_in_direction import bisection

def optimalStep(obj_fun: Callable[[np.ndarray], float],
                grad: Callable[[np.ndarray], np.ndarray],
                x_0: np.ndarray, args: tuple[float], callback: Callable=None, options={}) -> OptimizeResult:
    """_summary_
    Args:
        obj_fun : callable f(x, *args)
            Objective function to be minimized.

        grad : callable f(x, *args)
            Gradient of objective function.

        x_0 : array-like
            Initial guess.

        args : tuple, optional
            Extra arguments passed to the `obj_fun` and `grad`.

        callback : callable f(x), optional
            Function called after each iteration.

        options (dict, optional): 
            maxiter : int
                Maximum number of iterations to perform.
            tol : float
                Tolerance for termination.
            
    Raises:
        ValueError: function called without initial guess

    Returns:
        OptimizeResult
        The optimization result represented as a OptimizeResult object.
        Important attributes are: x the solution,
        success a Boolean flag indicating if the optimizer exited successfully,
        message which describes the cause of the termination.
    """
    if x_0 is None:
        raise ValueError("Must provide initial guess `x0`!")
    if grad is None:
        grad = lambda x: approx_fprime(x, obj_fun, args=args)

    maxiter: int = options.get("maxiter", 10_000)
    tol: float = options.get("tol", 1e-6)
    x: np.ndarray = np.array(x_0, dtype=np.float64)
    it: int
    for it in range(1, maxiter + 1):
        grad_value: np.ndarray = grad(x, *args)
        if np.linalg.norm(grad_value) < tol:
            break
        stepsize:float = bisection(obj_fun, grad, x).x

        x -= stepsize * grad_value

        if callback is not None:
            callback(x)

    success: bool = np.linalg.norm(grad_value) < tol
    msg: str
    if success:
        msg = "Optimization successful"
    else:
        msg = "Optimization failed"

    return OptimizeResult(x=x, success=success, message=msg, nit=it, njev = it)
    


def constantStep(obj_fun: Callable[[np.ndarray], float],
                 grad: Callable[[np.ndarray], np.ndarray],
                 x_0: np.ndarray, args: tuple[float], callback: Callable=None,  options= {}) -> OptimizeResult:
    """_summary_
    Args:
    obj_fun : callable f(x, *args)
        Objective function to be minimized.

    grad : callable f(x, *args)
        Gradient of objective function.

    x0 : array-like
        Initial guess.

    args : tuple, optional
        Extra arguments passed to the `obj_fun` and `grad`.

    callback : callable f(x), optional
        Function called after each iteration.

    options : dict, optional
        A diactionary with solver options.
            maxiter : int
                Maximum number of iterations to perform.
            tol : float
                Tolerance for termination.
            stepsize : float
                guess for step search.  
    Returns
    res : OptimizeResult
        The optimization result represented as a OptimizeResult object.
        Important attributes are: x the solution,
        success a Boolean flag indicating if the optimizer exited successfully,
        message which describes the cause of the termination.
        See OptimizeResult for a description of other attributes.
    """
    if x_0 is None:
        raise ValueError("Must provide initial guess `x_0`!")
    if grad is None:
        grad = lambda x: approx_fprime(x, obj_fun, args=args)

    maxiter: int = options.get("maxiter", 1000)
    tol: float = options.get("tol", 1e-6)

    x: np.ndarray = np.array(x_0, dtype=np.float64)
    stepsize: float = options.get("stepsize")

    it: int
    for it in range(1, maxiter+1):
        grad_value: np.ndarray = grad(x, *args)

        if np.linalg.norm(grad_value) < tol:
            break

        x -= stepsize * grad

        if callback is not None:
            callback(x)

    success: bool =  np.linalg.norm(grad_value) < tol

    msg: str
    if success:
        msg = "Optimization successful"
    else:
        msg = "Optimization failed"

    return OptimizeResult(x=x, success=success, message=msg, grad_value=grad_value, callback=callback, nit=it, njev=it)

