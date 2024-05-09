import numpy as np
from matplotlib.pyplot import Axes
from scipy.optimize import OptimizeResult
from typing import Optional, Callable
from visualizer import Visualizer
import pandas as pd
from scipy.optimize import approx_fprime, minimize
from minimization_methods.quasi_newton import BFGS, DFP
from minimization_methods.gradient_descent import optimalStep, constantStep
from warnings import warn


class LogisticRegression:
    """
    Handles fitting logistic regression on given data via numerous optimization methods.
    """
    _solution: OptimizeResult | None
    _prediction_function: Callable[[np.ndarray[np.ndarray]], np.ndarray[float]] | None

    fitted: bool
    coefficients: np.ndarray[float]

    def __init__(self) -> None:
        """
        Creates unfitted instance.
        """
        self._solution = None
        self.fitted = False

    def fit(self,
            u: np.ndarray[np.ndarray],
            v: np.ndarray[float],
            method: str | None = None,
            step_selection: str | None = None) -> None:
        """
        Fits the regression, i.e., finds optimal coefficients
        for a sigmoid function which describes given data the best. Stores the function.
        Parameters
        ----------
        u:
        v
        method
        step_selection
        Returns
        -------
        """
        if method is None or step_selection is None:
            raise ValueError("Method and step selection must be set")

        if self.fitted:
            warn("Overriding already fitted instance!")

        def objective_function(x: np.ndarray) -> float:
            return np.sum((1 - v) * np.dot(u, x) + np.log(1 + np.exp(-np.dot(u, x))))

        def gradient(x: np.ndarray) -> np.ndarray[float]:
            return np.dot(u.T, (1 - v - (1 / (1 + np.exp(np.dot(u, x))))))

        x0 = np.array([0] * u.shape[1])
        if method == "BFGS":
            self._solution = BFGS(obj_fun=objective_function, grad=gradient, x0=x0, step=step_selection)
        elif method == "DFP":
            self._solution = DFP(obj_fun=objective_function, grad=gradient, x_0=x0, step=step_selection)
        elif method == "Grad-Opt":
            self._solution = optimalStep(obj_fun=objective_function, grad=gradient, x_0=x0)
        elif method == "Grad-Const":
            self._solution = constantStep(obj_fun=objective_function, grad=gradient, x_0=x0)
        else:
            self._solution = minimize(objective_function, x0, jac=gradient)

        self.coefficients = self._solution.x

        def sigmoid(u: np.ndarray) -> float:
            return 1 / (1 + np.exp(-np.dot(self.coefficients, u)))

        self._prediction_function = sigmoid

        self.fitted = True

    def get_result(self) -> OptimizeResult:
        return self._solution

    def predict(self, u: np.ndarray[np.ndarray[int]]) -> list[float]:
        if not self.fitted:
            raise ValueError("Can't use predict on a non-fitted model!")

        result: list = []
        for i in u:
            result.append(self._prediction_function(i))
        return result
    
    def visualize(self, ax: Axes): #Roboooooo
        # Visualizer.visualize()
        pass

def main() -> None:
    df = pd.read_csv("data/credit_risk_train.csv")
    df = df.to_numpy()

    v: np.ndarray = df.T[0] #tu vytiahnes vektor v z dát
    u: np.ndarray[np.ndarray[int]] = df.T[1:].T     # toto sa chce robit v classe

    ones = np.ones((u.shape[0], 1))
    u = np.hstack((ones, u)) #horizontal stack jednotiek, brutalna funkcia do rodiny, podporujem ju

    '''hore sa pridava do matice U vektor jednotiek, to sa asi chce diat niekde inde'''
    test: LogisticRegression = LogisticRegression()
    test.fit(u=u, v=v,
                   method="DFP", step_selection="optimal")
    print(test.coefficients)

    df2 = pd.read_csv("data/credit_risk_test.csv")

    df2 = df2.to_numpy()
    v = df2.T[0]
    u = df2.T[1:].T
    ones = np.ones((u.shape[0], 1))
    u = np.hstack((ones, u))
    
    '''tu tak isto, asi sa to chce diat v classe'''
    predicted: np.ndarray[int] = np.rint(test.predict(u=u))
    res = []
    for i in range(len(v)):
        if v[i] == predicted[i]:
            res.append(1)
    '''kontrola, ci prediktnute v a povodne je dobre, asi tiez nie tu?'''

    print(len(res)/ len(v)) #percento spravnosti

if __name__ == "__main__":
    main()