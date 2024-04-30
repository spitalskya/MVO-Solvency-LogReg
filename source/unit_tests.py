import numpy as np 
from minimization_methods.gradient_descent import optimalStep, constantStep
from minimization_methods.quasi_newton import DFP, BFGS

def f1(x, a=1):
    A = np.diag((1, a))
    h = np.array((a, a**2))
    return 0.5 * x@A@x - x@h

def df1(x, a=1):
    A = np.diag((1, a))
    h = np.array((a, a**2))
    return A@x - h

def f2(x):
    A = np.array([[11, 9],[9,10]])
    h = np.array((200, 190))
    return 0.5 * x@A@x - x@h

def df2(x):
    A = np.array([[11, 9],[9,10]])
    h = np.array((200, 190))
    return A@x - h

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
        - The quartic function Q(x) as a callable function.
        - Gradient of Q(x) as callable function
        - A : numpy.ndarray
            A random n x n positive definite matrix.
        - G : numpy.ndarray
            A random n x n positive definite matrix.
        - h : numpy.ndarray
            A vector of length n.
    """
    A = np.random.randn(n, n)
    A = A@A.T + np.eye(n)

    G = np.random.randn(n, n)
    G = G@G.T + np.eye(n)

    h = np.random.randn(n)

    def f3(x):
        return 0.25 * (x@A@x)**2 + 0.5 * (x@G@x) + h@x

    def df3(x):
        return 0.5 * (x@A@x) *(A@x) + G@x + h

    if ret_params:
        return f3, df3, A, G, h

    return f3, df3

f3, df3 = generate_f3(4)
x3 = np.zeros(4)
x = np.zeros(2)
print(optimalStep(f3, df3, x, ()))
print(optimalStep(f1, df1, x, (5,)))
print(optimalStep(f2, df2, x, ()))
print(constantStep(f3, df3, x, (),{"stepsize": 1e-3}))
print(constantStep(f1, df1, x, (5,),{"stepsize": 1e-1}))
print(constantStep(f2, df2, x, (),{"stepsize": 1e-1}))