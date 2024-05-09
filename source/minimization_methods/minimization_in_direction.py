from typing import Callable, Optional
import numpy as np
from scipy.optimize import OptimizeResult, approx_fprime


def backtracking(obj_fun: Callable[[np.ndarray], float],
                 x_0: np.ndarray, s: np.ndarray,
                 grad: Optional[Callable[[np.ndarray], np.ndarray]]=None,
                 alpha: float=0.1, delta: float=0.5,
                 callback: Optional[Callable]=None, args: tuple = (), **kwargs) -> OptimizeResult:
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
        def grad(x: np.ndarray) -> np.ndarray:
            return approx_fprime(x, obj_fun, *args)
    
    lam: float = 1
    it: int = 0
    maxiter: int = kwargs.get("maxiter", 1000)
    fun0: float = obj_fun(x_0, *args)
    direction_der_x_0: float = np.dot(grad(x_0), s)
    success: bool

    
    while obj_fun(x_0 + lam * s, *args) >= fun0 + alpha * lam * direction_der_x_0:
        if it == maxiter:
            success = False
            break

        if callback is not None:
            callback(lam)
        
        lam *= delta
        it += 1
    
    if lam == 0:
        success = False
    else:
        success = True
    
    return OptimizeResult(x=lam, nit=it, nfev=it+1, njev=1, success=success)
    
def bisection(obj_fun: Callable[[np.ndarray], float],
              x_0: np.ndarray[float], s: np.ndarray[float], args: tuple=(),
              grad: Optional[Callable[[np.ndarray], np.ndarray]]=None,
              callback: Optional[Callable]=None, **kwargs) -> OptimizeResult:
    """
    Minimization method

    Parameters:
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
        
        **kwargs: dict, optional
            Other parameters passed to bisection
                maxiter : int
                    Maximum number of iterations to perform.
                tol : float
                    Tolerance for termination
    
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
        def grad(x: np.ndarray, *args) -> np.ndarray:
            return approx_fprime(x, obj_fun, *args)
    
    def fun(lam: float, *args):
        return grad(x_0 + lam * s, *args) @ s
    
    njev: int = 0

    dir_derivative_0: float = np.dot(grad(x_0, *args), s)

    #getting bounds a, b
    if dir_derivative_0 < 0:
        x: np.ndarray[float] = x_0.copy()
        k: int = 1
        while True:
            x += k * s
            dir_derivative: float = np.dot(grad(x, *args), s)
            if dir_derivative > 0:
                a: np.ndarray[float] = x - (k/2) * s
                b: np.ndarray[float] = x
                break
            elif dir_derivative == 0:
                print('found minimum')
                #TODO
            k *= 2
            njev += 1
    
    elif dir_derivative_0 > 0:
        x: np.ndarray[float] = x_0.copy()
        k: int = 1
        while True:
            x -= k * s
            dir_derivative: float = np.dot(grad(x, *args), s)
            if dir_derivative < 0:
                a: np.ndarray[float] = x
                b: np.ndarray[float] = x + (k/2) * s
                break
            elif dir_derivative == 0:
                print('found minimum')
                #TODO
            k *= 2
            njev += 1
    else:
        print('found minimum')
        #TODO
    
    print(a, b)

    tol: float = kwargs.get("tol", 1e-6)
    maxiter: int = kwargs.get("maxiter", 1000)
    midpoint: float = (a+b) / 2

    def fun(lam: float, *args):
        return grad(x_0 + lam * s, *args) @ s
    
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
        
        if np.linalg.norm(b-a) < tol:
            break

        njev += 1
    
    success: bool = np.linalg.norm(b-a) < tol

    msg: str
    if success:
        msg = "Optimatization successful"
    else:
        msg = "Optimatization failed"
    print('##############')
    res: float = np.linalg.norm(midpoint - x_0) / np.linalg.norm(s)
    print('##############')
    
    return OptimizeResult(x=(a+b)/2, success=success, message=msg, 
                          nit=it, tol=tol, interval=(a, b), njev=njev, nfev=0)


def f1(x, a=1):
    A = np.diag((1, a))
    h = np.array((a, a**2))
    return 0.5 * x@A@x - x@h

def df1(x, a=1):
    A = np.diag((1, a))
    h = np.array((a, a**2))
    return A@x - h

print(bisection(obj_fun=f1, x_0=np.array([-2, -2]), s=np.array([5, 5]), grad=df1))
