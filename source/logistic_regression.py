import numpy as np
from matplotlib.pyplot import Axes
from scipy.optimize import OptimizeResult
from visualizer import Visualizer


class LogisticRegression:
    
    
    _u: np.ndarray[np.ndarray[int]]
    _v: np.ndarray[int]
    
    _solution: OptimizeResult | None
    
    
    def __init__(self, u: np.ndarray, v: np.ndarray) -> None:
        self._u = u
        self._v = v
    
    def objective_function(self, x: np.ndarray) -> float:
        return np.sum((1 - self._v) * np.dot(x, self._u) + np.log(1 + np.exp(-np.dot(x, self._u))))
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        matrix: np.ndarray[np.ndarray[int]] = self._u
        ones_vector: np.ndarray[1] = np.ones((matrix.shape[0], 1))
        matrix = np.hstack((ones_vector, matrix))
        return np.sum(np.outer(matrix.T, (1 - self._v - (1 / 1 + np.exp(np.dot(x, self._u))))))
    
    
    def get_result() -> OptimizeResult:
        pass
    
    def visualize(self, ax: Axes):
        # Visualizer.visualize()
        pass

