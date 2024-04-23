from typing import Callable
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from minimization_methods.minimization_in_direction import bisection, backtracking


def BFGS(obj_fun: Callable[[np.ndarray], float],
         grad: Callable[[np.ndarray], np.ndarray],
         x0: np.ndarray, step: str) -> OptimizeResult:
    # ! document
    """_summary_

    Args:
        obj_fun (Callable[[np.ndarray], float]): _description_
        grad (Callable[[np.ndarray], np.ndarray]): _description_
        x0 (np.ndarray): _description_
        step (str): _description_
    Returns:
        OptimizeResult: _description_
    """
    
    if step == "optimal":
        step_optimizer = bisection
    elif step == "suboptimal":
        step_optimizer = backtracking
    
    g = grad(x0)
    H = np.identity(x0.shape[0])
    x = x0
    
    maxiter = 1000
    tol = 1e-3
    nfev = 0
    njev = 1
    
    for it in range(1, maxiter + 1):
        s = -H @ g
        
        # TODO 
        """
        step_optimizer_result = step_optimizer(obj_fun, grad, x0, s
        nfev += step_optimizer_result.nfev
        njev += step_optimizer_result.njev
        lam = step_optimizer_result.x[0]
        """
        lam = (minimize(lambda lam: obj_fun(x + lam * s), 0).x)[0]
        if (lam == 0): break        # should not happen?
        
        x_plus = x + lam * s
        g_plus = grad(x_plus)
        
        y = g_plus - g
        p = x_plus - x
        
        # TODO callback
        
        if np.linalg.norm(g_plus) < tol:
            break
        
        # H+ calculation
        yp_outer: np.ndarray = np.outer(y, p)
        denominator: float = p @ y
        H += (1 + (y @ H @ y)/denominator) * ((np.outer(p, p)) / denominator) - (H @ yp_outer + yp_outer @ H) / denominator
        
        x = x_plus
        g = g_plus
        
    if np.linalg.norm(g_plus) < tol:
        msg, success = "Optimization successful", True
    else:
        msg, success = "Optimization not successful", False
    
    return OptimizeResult(x=x_plus, success=success, message=msg,
                          nit=it, nfev=nfev, njev=njev+it)



def DFP(obj_fun: Callable[[np.ndarray], float],
        grad: Callable[[np.ndarray], np.ndarray],
        x_0: np.ndarray, step: str) -> OptimizeResult:
     pass


def main() -> None:
    def generate_f3(n, ret_params=False):
        """
        Generate a quartic function f3(x) of the form
        0.25 * x^T * A * x + 0.5 * x^T * G * x + h^T * x.

        Parameters
        ----------
        n : int
            The dimension of the quadratic function.
        ret_params : bool, optional
            Whether to return additional parameters used to define the function.
            If True, returns the quartic function and its gradient..
            If False, returns also matrices A, G and vector h.

        Returns
        -------
        If `ret_params` is False, returns the quartic function Q(x) as
        a callable function and its gradient as callable function.

        If `ret_params` is True, returns a tuple containing:
            - The quartic function f3(x) as a callable function.
            - Gradient of f3(x) as callable function
            - A : numpy.ndarray
                A random n x n positive definite matrix.
            - G : numpy.ndarray
                A random n x n positive definite matrix.
            - h : numpy.ndarray
                A vector of length n.
        """
        def generate_random_positive_definite(n):
            S = np.random.randint(-100, 100, (n, n))
            
            eps = 1
            return S.T @ S + eps * np.identity(n)
            
        A = generate_random_positive_definite(n)    
        
        G = generate_random_positive_definite(n)

        h = np.random.randn(n)

        def f3(x):
            return 0.25 * (x@A@x)**2 + 0.5 * (x@G@x) + h@x

        def df3(x):
            return (A @ x) * (x @ A @ x) + G @ x + h

        if ret_params:
            return f3, df3, A, G, h

        return f3, df3
    
    f1, df1 = generate_f3(4)
    
    # simple test of methods
    x = np.zeros(4)
    print(BFGS(f1, df1, x, "suboptimal").x)
    # print(DFP(f1, df1, x, "suboptimal").x)
    print(minimize(f1, x).x)

if __name__ == "__main__":
    main()
