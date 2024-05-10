from typing import Callable, Optional
import numpy as np
from scipy.optimize import OptimizeResult, approx_fprime
from minimization_methods.minimization_in_direction import bisection

def optimalStep(obj_fun: Callable[[np.ndarray], float],
                grad: Optional[Callable[[np.ndarray], np.ndarray]],
                x_0: np.ndarray, args: tuple[float]=(), callback: Callable=None, **kwargs) -> OptimizeResult:
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
        def grad(x: np.ndarray, *args) -> np.ndarray:
            return approx_fprime(x, obj_fun, *args)
    maxiter: int = kwargs.get("maxiter", 1000)
    tol: float = kwargs.get("tol", 1e-3)
    x: np.ndarray = np.array(x_0, dtype=np.float64)
    trajectory: list[np.ndarray] = [x.copy()]

    it: int
    njev_bi: int = 0
    nit_bi: int = 0
    for it in range(1, maxiter + 1):
        grad_value: np.ndarray = grad(x, *args)
        if np.linalg.norm(grad_value) < tol:
            break

        stepsizeInfo: OptimizeResult = bisection(obj_fun=obj_fun, grad=grad, x_0=x, args=args, s=-grad_value)
        stepsize: float = stepsizeInfo.x
        nit_bi += stepsizeInfo.nit
        njev_bi += stepsizeInfo.njev

        x -= stepsize * grad_value
        trajectory.append(x.copy())
        if callback is not None:
            callback(x)

    success: bool = np.linalg.norm(grad_value) < tol
    msg: str
    if success:
        msg = "Optimization successful"
    else:
        msg = "Optimization failed"

    return OptimizeResult(x=x, success=success, 
                          message=msg, nit=it + nit_bi, 
                          njev=it + njev_bi, trajectory=trajectory)
    


def constantStep(obj_fun: Callable[[np.ndarray], float],
                 grad: Optional[Callable[[np.ndarray], np.ndarray]],
                 x_0: np.ndarray, args: tuple[float] = (), callback: Callable=None,  **kwargs) -> OptimizeResult:
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
        def grad(x: np.ndarray, *args) -> np.ndarray:
            return approx_fprime(x, obj_fun, *args)

    maxiter: int = kwargs.get("maxiter", 1000)
    tol: float = kwargs.get("tol", 1e-3)

    x: np.ndarray = np.array(x_0, dtype=np.float64)
    stepsize: float = kwargs.get("stepsize", 1e-1)
    trajectory: list[np.ndarray] = [x.copy()]

    it: int
    for it in range(1, maxiter+1):
        grad_value: np.ndarray = grad(x, *args)

        if np.linalg.norm(grad_value) < tol:
            break
        x -= stepsize * grad_value
        trajectory.append(x.copy())

        if callback is not None:
            callback(x)

    success: bool =  np.linalg.norm(grad_value) < tol

    msg: str
    if success:
        msg = "Optimization successful"
    else:
        msg = "Optimization failed"

    return OptimizeResult(x=x, success=success, message=msg,
                          grad_value=grad_value, 
                          nit=it, njev=it, trajectory=trajectory)

