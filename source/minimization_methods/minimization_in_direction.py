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
    
    lam: float = 1
    it: int = 0
    fun0: float = obj_fun(x_0, *args)
    direction_der_x_0: float = np.dot(grad(x_0), s)

    
    while obj_fun(x_0 + lam * s, *args) >= fun0 + alpha * lam * direction_der_x_0:
        if it == maxiter:
            break

        if callback is not None:
            callback(lam)
        
        lam *= delta
        it += 1
    
    return OptimizeResult(x=lam, nit=it)
    
def bisection(obj_fun: Callable[[np.ndarray], float],
              grad: Callable[[np.ndarray], np.ndarray],
              x_0: np.ndarray, s: np.ndarray) -> OptimizeResult:
    pass
