import numpy as np
from matplotlib.pyplot import Axes
from scipy.optimize import OptimizeResult
from visualizer import Visualizer


class LogisticRegression:
    
    
    _u: np.ndarray
    _v: np.ndarray
    
    _solution: OptimizeResult | None
    
    
    def __init__(u: np.ndarray, v: np.ndarray) -> None:
        pass
    
    
    def objective_function(x: np.ndarray) -> float:
        pass
    
    
    def gradient(x: np.ndarray) -> np.ndarray:
        pass
    
    
    def get_result() -> OptimizeResult:
        pass
    
    
    def visualize(self, ax: Axes):
        # Visualizer.visualize()
        pass
