import matplotlib.pyplot as plt
import numpy as np

class Visualizer:

    @staticmethod
    def visualize(ax: plt.Axes, j_k: np.ndarray, j_opt: float) -> None:
        num_iterations = len(j_k)
        iterations = list(range(1, num_iterations + 1))
        result = np.zeros(num_iterations)
        for i in range(num_iterations):
            result[i] = np.linalg.norm(j_k[i] - j_opt)
        
        ax.plot(iterations, result, marker='o', linewidth=2)
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('|J(x_k) - J*|')
        ax.set_title('Convergence graph')
        ax.grid(True)
