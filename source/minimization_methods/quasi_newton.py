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

    maxiter = kwargs.get("maxiter", 1000)
    tol = kwargs.get("tol", 1e-1)

    nfev: int = 0
    njev: int = 1
    trajectory: list[np.ndarray] = [x]

    for it in range(1, maxiter + 1):
        # TODO: increment nfev, njev and replace optimizers
        # compute direction
        s = -H @ g

        # find optimal step size
        step_len = minimize(lambda step_size: obj_fun(x + step_size * s), np.zeros(1)).x[0]
        if step_len == 0:
            step_len = 10e-10

        # compute next x
        x_plus = x + step_len * s
        g_plus = grad(x_plus)
        trajectory.append(x_plus.copy())

        # callback
        if callback:
            callback(x_plus)

        # compute gradient, end if less than tol
        if np.linalg.norm(g_plus) < tol:
            break

        # compute another Hessian
        p_k = x_plus - x
        y_k = g_plus - g
        H += (np.outer(p_k, p_k) / np.inner(p_k, y_k)) - ((H@np.outer(y_k, y_k)@H) / (y_k@H@y_k))
        # s_k = step_len * s
        # y_k = g_plus - g
        # z_k = H @ y_k
        #
        # H = H + (np.outer(s_k, s_k) / np.inner(s_k, y_k)) - (np.outer(z_k, z_k) / np.inner(y_k, z_k))

        # assign next x, g
        x = x_plus
        g = g_plus
        
    if np.linalg.norm(g_plus) < tol:
        msg, success = "Optimization successful", True
    else:
        msg, success = "Optimization not successful", False

    # TODO: you need nfev and njev
    return OptimizeResult(x=x, success=success, message=msg,
                          nit=it, nfev=nfev, njev=njev)


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
            return 0.25 * (x @ A @ x) ** 2 + 0.5 * (x @ G @ x) + h @ x

        def df3(x):
            return (A @ x) * (x @ A @ x) + G @ x + h

        if ret_params:
            return f3, df3, A, G, h

        return f3, df3

    f1, df1 = generate_f3(4)

    # simple test of methods
    x = np.zeros(4)
    print(DFP(f1, df1, x, "suboptimal"))
    # print(DFP(f1, df1, x, "suboptimal", ()))
    print(minimize(f1, x).x)


if __name__ == "__main__":
    main()

