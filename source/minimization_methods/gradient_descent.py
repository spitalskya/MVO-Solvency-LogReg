from typing import Callable
import numpy as np
from scipy.optimize import OptimizeResult
from minimization_in_direction import backtracking, bisection

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
            max_stepsize : numeric
                Initial guess for step search.

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

    maxiter = options.get("maxiter", 10_000)
    tol = options.get("tol", 1e-6)
    max_stepsize = options.get("stepsize", 1)
    x = np.array(x_0)
    for it in range(1, maxiter + 1):
        grad_value = grad(x, *args)
        s = -grad
        if np.linalg.norm(grad) < tol:
            break
        dfun = lambda lam : grad(x + lam*s, *args) @ s
        stepsize = bisection(dfun, bounds=(0, max_stepsize))['x']

        x -= stepsize * grad_value

        if callback is not None:
            callback(x)

    success = np.linalg.norm(grad(x, *args)) < tol

    if success:
        msg = "Optimization successful"
    else:
        msg = "Optimization failed"

    return OptimizeResult(x=x, success=success, message=msg, nit=it, njev = it)
    


def constantStep(obj_fun: Callable[[np.ndarray], float],
                 grad: Callable[[np.ndarray], np.ndarray],
                 x_0: np.ndarray, args: tuple[float], callback: Callable =None,  options= {}) -> OptimizeResult:
    """
    Minimization method

    Parameters
    ----------
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
            max_stepsize : numeric
                Initial guess for step search.
            maxiter_step_search: int
                Maximum number of iteration to perform when searching
                for step size.
            

    **kwargs : dict, optional
        Other parameters passed to `backtrack`. Will be ignored.

    Raises
    ------
    ValueError
        if `x0` is not provided.
    AssertionError
        If `alpha` is not from range (0, 1/2] or `delta` is not from range (0, 1).

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a OptimizeResult object.
        Important attributes are: x the solution,
        success a Boolean flag indicating if the optimizer exited successfully,
        message which describes the cause of the termination.
        See OptimizeResult for a description of other attributes.
    """
    if x_0 is None:
        raise ValueError("Must provide initial guess `x_0`!")

    maxiter = options.get("maxiter", 1000)
    tol = options.get("tol", 1e-6)

    x = np.array(x_0)

    grad_value = grad(x, *args)
    stepsize = backtracking(obj_fun=obj_fun, grad=grad, x_0 = x_0, s = -grad_value, args=args)
    for it in range(1, maxiter+1):
        
        if np.linalg.norm(grad) < tol:
            break

        x -= stepsize * grad

        if callback is not None:
            callback(x)
        grad_value = grad(x, *args)

    success =  np.linalg.norm(grad_value) < tol

    if success:
        msg = "Optimization successful"
    else:
        msg = "Optimization failed"

    return OptimizeResult(x=x, success=success, message=msg, jac=grad_value, callback = callback, nit=it)

