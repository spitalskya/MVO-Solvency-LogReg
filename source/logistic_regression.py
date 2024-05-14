from typing import Callable, Literal
from warnings import warn

import numpy as np
import pandas as pd
from matplotlib.pyplot import Axes
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize
from sklearn.metrics import precision_score
from visualizer import Visualizer
import matplotlib.pyplot as plt
from minimization_methods.gradient_descent import optimalStep, constantStep
from minimization_methods.quasi_newton import BFGS, DFP
# FIXME: consider suppressing or solving RuntimeWarning: overflow encountered in exp
#   return np.dot(u.T, (1 - v - (1 / (1 + np.exp(np.dot(u, x))))))


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
            step_selection: Literal["optimal", "suboptimal"] | None = None) -> None:
        """
        Fits the regression, i.e., finds optimal coefficients
        for a sigmoid function which describes given data the best. Stores the function.
        Parameters
        ----------
        u: matrix with data to learn from
        v: vector with classification values corresponding to given u matrix.
        method: which optimization method to use to acquire sigmoid function
        step_selection: which method to use to find the step for a method.
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

        ones = np.ones((u.shape[0], 1))
        u = np.hstack((ones, u))

        x0 = np.array([0] * u.shape[1])
        if method == "BFGS":
            self._solution = BFGS(obj_fun=objective_function, grad=gradient, x0=x0, step=step_selection)
        elif method == "DFP":
            self._solution = DFP(obj_fun=objective_function, grad=gradient, x_0=x0, step=step_selection)
        elif method == "Grad-Opt":
            self._solution = optimalStep(obj_fun=objective_function, grad=gradient, x_0=x0)
        elif method == "Grad-Const":
            self._solution = constantStep(obj_fun=objective_function, grad=gradient, x_0=x0, stepsize=1e-8)
        else:
            raise ValueError("Wrong method")

        self.coefficients = self._solution.x

        def sigmoid(u_for_pred: np.ndarray[np.ndarray]) -> np.ndarray[float]:
            return 1 / (1 + np.exp(-np.inner(self.coefficients, u_for_pred)))

        self._prediction_function = sigmoid

        self.fitted = True

    def get_result(self) -> OptimizeResult:
        """
        Returns the OptimizeResult from method optimization running.
        Returns
        -------
        OptimizeResult with all the data.
        """
        return self._solution

    def predict_proba(self, u: np.ndarray[np.ndarray[int]]) -> np.ndarray[float]:
        """
        Returns predictions probabilities from given u.
        Parameters
        ----------
        u: matrix from which to make predictions.

        Returns
        -------
        array of predicted probabilities
        """
        if not self.fitted:
            raise ValueError("Can't use predict on a non-fitted model!")
        ones = np.ones((u.shape[0], 1))
        u = np.hstack((ones, u))

        # noinspection PyTypeChecker
        result = self._prediction_function(u)

        # TODO: delete comment after review; the precision scores do not match
        # result: list = []
        # for i in u:
        #     result.append(self._prediction_function(i))

        return result

    def predict(self, u: np.ndarray[np.ndarray[int]]) -> np.ndarray[float]:
        """
        Returns predicted values with probabilities rounded.
        Parameters
        ----------
        u: matrix from which to make predictions.
        Returns
        -------
        array of predicted values i.e. array with 0-oes or 1-s.
        """
        return np.rint(self.predict_proba(u))
    
    def visualize(self, ax: Axes) -> None:
        # FIXME: review the visualizations
        """
        Visualizes the trajectory.
        Parameters
        ----------
        ax: Matplotlib axes object to plot to.
        Returns
        -------
        """
        Visualizer.visualize(ax, self._solution.trajectory, self._solution.x)


def main() -> None:
    train = pd.read_csv("data/credit_risk_train.csv")
    u_train, v_train = train.drop("Creditability", axis="columns").to_numpy(), train["Creditability"].to_numpy()

    log_reg = LogisticRegression()

    log_reg.fit(u=u_train, v=v_train, method="DFP", step_selection="suboptimal")
    print(log_reg.coefficients)

    test = pd.read_csv("data/credit_risk_test.csv")
    u_test = test.drop("Creditability", axis="columns").to_numpy()
    v_real, v_pred = test["Creditability"].to_numpy(), log_reg.predict(u_test)

    print(precision_score(v_real, v_pred))

    # FIXME: does not much our precision score
    predicted: np.ndarray[int] = np.rint(log_reg.predict(u=u_test))
    res = []
    for i in range(len(v_real)):
        if v_real[i] == predicted[i]:
            res.append(1)
    print(len(res) / len(v_real))

    figure, ax = plt.subplots(1, 1)
    log_reg.visualize(ax)
    plt.savefig("trajectory_viz.png")


if __name__ == "__main__":
    main()
