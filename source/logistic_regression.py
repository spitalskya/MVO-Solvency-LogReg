import numpy as np
from matplotlib.pyplot import Axes
from scipy.optimize import OptimizeResult
from visualizer import Visualizer
import pandas as pd
from scipy.optimize import approx_fprime, minimize
from minimization_methods.quasi_newton import BFGS, DFP
from minimization_methods.gradient_descent import optimalStep, constantStep

class LogisticRegression:
    
    
    _u: np.ndarray[np.ndarray[int]]
    _v: np.ndarray[int]
    _x0: np.ndarray[int]
    _x_min: np.ndarray[float]
    
    _solution: OptimizeResult | None
    
    
    def __init__(self, u: np.ndarray[np.ndarray[int]], v: np.ndarray[int]) -> None:
        self._u = u
        self._v = v
        self._x0 = np.array([0,0,0,0]) #toto sa vsade pouziva tak isto, takze asi cool mat tu ako class premenenu
    
    def objective_function(self, x: np.ndarray) -> float:
        return np.sum((1 - self._v) * np.dot(self._u, x) + np.log(1 + np.exp(-np.dot(self._u, x))))

    def gradient(self, x: np.ndarray) -> np.ndarray[float]:
        result: np.ndarray[float] = np.dot(self._u.T, (1 - self._v - (1 / (1 + np.exp(np.dot(self._u, x))))))
        return result
    
    """dvoch funkcii hore sa nechytat, funguju :D"""

    def fit(self, method: str=None, step_selection: str=None) -> None: #neviem ci fit je dobry nazov?? asi nie :/ kedze sa len hlada x minimum 
        if method is None or step_selection is None:
            raise ValueError("Method and step selection must be set")
        
        if method == "BFGS":
            self._x_min = BFGS(self.objective_function, self.gradient, x_0=self._x0, step=step_selection).x
        elif method == "DFP":
            self._x_min = DFP(self.objective_function, self.gradient, self._x0, step_selection).x
        elif method == "Grad-Opt":
            self._x_min = optimalStep(self.objective_function, self.gradient, self._x0).x
        elif method == "Grad-Const":
            self._x_min = constantStep(self.objective_function, self.gradient, self._x0).x
        else:
            self._x_min = minimize(self.objective_function, self._x0, jac=self.gradient).x

    
    def get_result(self, method: str=None, step_selection: str=None) -> OptimizeResult:
        ...
    
    def sigmoid(self, u: np.ndarray) -> float:
        return 1 / (1 + np.exp(-np.dot(self._x_min, u)))
    
    def predict(self, u: np.ndarray[np.ndarray[int]]) -> list[float]: #toto prosim nejak mudrejsie 
        result: list = []
        for i in u:
            result.append(self.sigmoid(i))
        return result
    
    def visualize(self, ax: Axes): #Roboooooo
        # Visualizer.visualize()
        pass

df = pd.read_csv("C:/Users/antal/Desktop/matfyz/Metódy voľnej optimalizácie/MVO-Solvency-LogReg/source/data/credit_risk_train.csv")
df = df.to_numpy()

v: np.ndarray = df.T[0] #tu vytiahnes vektor v z dát
u: np.ndarray[np.ndarray[int]] = df.T[1:].T

ones = np.ones((u.shape[0], 1))
u = np.hstack((ones, u)) #horizontal stack jednotiek, brutalna funkcia do rodiny, podporujem ju 

'''hore sa pridava do matice U vektor jednotiek, to sa asi chce diat niekde inde'''
test: LogisticRegression = LogisticRegression(u=u, v=v)
print(test.fit())

df2 = pd.read_csv("C:/Users/antal/Desktop/matfyz/Metódy voľnej optimalizácie/MVO-Solvency-LogReg/source/data/credit_risk_test.csv")

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