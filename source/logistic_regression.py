import numpy as np
from matplotlib.pyplot import Axes
from scipy.optimize import OptimizeResult
from visualizer import Visualizer
import pandas as pd
from scipy.optimize import approx_fprime, minimize
from minimization_methods.minimization_in_direction import backtracking

class LogisticRegression:
    
    
    _u: np.ndarray[np.ndarray[int]]
    _v: np.ndarray[int]
    
    _solution: OptimizeResult | None
    
    
    def __init__(self, u: np.ndarray[np.ndarray[int]], v: np.ndarray[int]) -> None:
        self._u = u
        self._v = v
    
    def objective_function(self, x: np.ndarray) -> float:
        return np.sum((1 - self._v) * np.dot(self._u, x) + np.log(1 + np.exp(-np.dot(self._u, x))))

    def gradient(self, x: np.ndarray) -> np.ndarray[float]:
        result: np.ndarray[float] = np.dot(self._u.T, (1 - self._v - (1 / (1 + np.exp(np.dot(self._u, x))))))
        return result

    def fit(self):
        self.x = minimize(self.objective_function, np.array([0,0,0,0]), jac=self.gradient).x

    
    def get_result(self, method: str=None, step_selection: str=None) -> OptimizeResult:
        ...
    
    def predict(self, u: np.ndarray[np.ndarray[int]]) -> float:
        result: list = []
        for i in u:
            result.append(1 / (1 + np.exp(-np.dot(self.x, i))))
        return result
    
    def visualize(self, ax: Axes):
        # Visualizer.visualize()
        pass

df = pd.read_csv("C:/Users/antal/Desktop/matfyz/Metódy voľnej optimalizácie/MVO-Solvency-LogReg/source/data/credit_risk_train.csv")
df = df.to_numpy()

v: np.ndarray = df.T[0]
u: np.ndarray[np.ndarray[int]] = df.T[1:].T

ones = np.ones((u.shape[0], 1))
u = np.hstack((ones, u))
test: LogisticRegression = LogisticRegression(u=u, v=v)
print(test.fit())

df2 = pd.read_csv("C:/Users/antal/Desktop/matfyz/Metódy voľnej optimalizácie/MVO-Solvency-LogReg/source/data/credit_risk_test.csv")

df2 = df2.to_numpy()
v = df2.T[0]
u = df2.T[1:].T
ones = np.ones((u.shape[0], 1))
u = np.hstack((ones, u))

predicted: np.ndarray[int] = np.rint(test.predict(u=u))
res = []
for i in range(len(v)):
    if v[i] == predicted[i]:
        res.append(1)

print(len(res)/ len(v))