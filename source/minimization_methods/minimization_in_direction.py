from typing import Callable
import numpy as np
from scipy.optimize import OptimizeResult, approx_fprime


def backtracking(obj_fun: Callable[[np.ndarray], float],
                 x_0: np.ndarray, s: np.ndarray,
                 grad: Callable[[np.ndarray], np.ndarray]=None,
                 alpha: float=0.1, delta: float=0.5,
                 callback: Callable=None, maxiter: int = 1000, args: tuple = (), **kwargs) -> OptimizeResult:
    """
    Minimization method

    Parameters:
    -----------
        fun : callable f(x, *args)
            objective function
        
        dfun : callable f(x, *args)
            derivative of objective function
                If not provided, uses approx_fprime

        alpha : float
            Parameter for auxiliary function
        
        delta : float
            Parameter of reduction

        x0 : float
            starting point
        
        args : tuple, optional
            Extra arguments passed to the objective function
        
        callback : callable f(x), optional
            Function called after each iteration
        
        maxiter : int
            Maximum number of iterations to perform
        
        **kvargs : dict, optional
            Other parameters passed to 'backtracking'
    
    Raises
    ------
    ValueError
        if 'x_0' is not provided

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a OptimizeResult object.
        Important attributes are: x the solution,
        See OptimizeResult for a description of other attributes.
    """

    if x_0 is None:
        raise ValueError("Initial guess 'x0' must be provided.")
       
    if grad is None:
        grad = lambda x: approx_fprime(x, obj_fun, args=args)
    
    x: np.ndarray[float] = np.array(x_0, dtype=float)
    it: int = 0
    fun0: float = obj_fun(np.zeros(len(x_0)), *args)
    grad0: np.ndarray[float] = grad(np.zeros(len(x_0)), *args)
    
    while obj_fun(x, *args) >= fun0 + alpha * np.dot(x, grad0):
        if it == maxiter:
            break

        if callback is not None:
            callback(x)
        
        x *= (grad(x) @ s) * delta
        it += 1
    
    return OptimizeResult(x=x, nit=it)


def f1(x, a=1):
    A = np.diag((1, a))
    h = np.array((a, a**2))
    return 0.5 * x@A@x - x@h

def df1(x, a=1):
    A = np.diag((1, a))
    h = np.array((a, a**2))
    return A@x - h

print(backtracking(f1, np.array([100, 100]), np.array([1, 1]), df1))
    


def bisection(obj_fun: Callable[[np.ndarray], float],
              grad: Callable[[np.ndarray], np.ndarray],
              x_0: np.ndarray, s: np.ndarray) -> OptimizeResult:
    pass
