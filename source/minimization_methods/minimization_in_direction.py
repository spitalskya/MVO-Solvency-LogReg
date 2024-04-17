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
        obj_fun : callable f(x, *args)
            objective function
        
        grad : callable f(x, *args)
            derivative of objective function
                If not provided, uses approx_fprime

        alpha : float
            Parameter for auxiliary function
        
        delta : float
            Parameter of reduction

        x_0 : np.ndarray
            Starting point
        
        s: np.ndarray
            Direction vector
        
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
              x_0: np.ndarray[float], s: np.ndarray[float], args: tuple=(),
              grad: Callable[[np.ndarray], np.ndarray]=None,
              callback: Callable=None, options:dict={}, **kwargs) -> OptimizeResult:
    """

    Args:
        obj_fun: Callable[[np.ndarray], float]
            Objective function

        grad: Callable[[np.ndarray], np.ndarray] 
            Derivation of objective function
                If not provided, uses approx_fprime

        x_0: np.ndarray
            Starting point

        s: np.ndarray
            Direction
        args: tuple, optional
            Extra arguments passed to the objective function.

        callback: Callable, optional
            Function called after each iteration.

        maxiter: int, optional
            Maximal number of iterations to perform
        
        options : dict, optional
            A diactionary with solver options.
                maxiter : int
                    Maximum number of iterations to perform.
                tol : float
                    Tolerance for termination
        
        **kwargs: dict, optional
            Other parameters passed to bisection
    
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
    
    direction_der: float = np.dot(grad(x_0, *args), s)

    #getting bounds a, b
    if direction_der < 0:
        x: np.ndarray[float]= x_0
        k: int = 1
        while True:
            x += k * s
            if grad(x, *args) > 0:
                a: np.ndarray[float] = x - s
                b: np.ndarray[float] = x
                break
            k += 1
    
    elif direction_der > 0:
        x: np.ndarray[float] = x_0
        k: int = 1
        while True:
            x -= k * s
            if grad(x, *args) < 0:
                a: np.ndarray[float] = x
                b: np.ndarray[float] = x + s
                break
            k += 1
    tol: float = options.get("tol", 1e-6)
    maxiter: int = options.get("maxiter", 1000)
    midpoint: float = (a+b) / 2
    
    it: int
    for it in range(1, maxiter+1):
        value: float = np.dot(grad(midpoint, *args), s)
        if value < 0:
            a = midpoint
        elif value >= 0:
            b = midpoint
        
        midpoint = (a+b) / 2

        if callback is not None:
            callback(midpoint)
        
        if np.linalg.norm(gradJ(midpoint)) < tol:
            break
    
    success: bool = np.linalg.norm(gradJ(midpoint)) < tol

    msg: str
    if success:
        msg = "Optimatization successful"
    else:
        msg = "Optimatization failed"
    
    return OptimizeResult(x=(a+b)/2, success=success, message=msg,
                          nit=it, tol=tol, interval=(a, b))
    

    

