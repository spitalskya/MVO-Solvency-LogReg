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

if __name__ == "__main__":
    def obj_fun(x):
        return (x - 1) ** 2  

    def grad_fun(x):
        return 2 * (x**2 - 1)  
    
    def bisection(obj_fun, grad_fun, x_k, a=0, b=1, tol=1e-6, max_iter=100):
        for i in range(max_iter):
            x = (a + b) / 2
            grad_value = grad_fun(x_k)
            grad_value_x = grad_fun(x_k - x * grad_value)
            if grad_value_x >= 0:
                b = x
            else:
                a = x
            if abs(grad_value_x) < tol:
                break
        return x
    def gradient_descent_optimalstep_by_bisection(obj_fun, grad_fun, x_0, j_opt, num_iterations=10):
        j_k_opt = []
        x_k = x_0
        for i in range(num_iterations):
            grad_value = grad_fun(x_k)
            stepsize = bisection(obj_fun, grad_fun, x_k)
            x_k -= stepsize * grad_value
            j_k_opt.append(obj_fun(x_k))
        return np.array(j_k_opt)  # Convert to numpy array
    
    def gradient_descent_constantstep(obj_fun, grad_fun, x_0, j_opt, stepsize, num_iterations=10):
        j_k_const = []
        x_k = x_0
        for i in range(num_iterations):
            grad_value = grad_fun(x_k)
            x_k -= stepsize * grad_value
            j_k_const.append(obj_fun(x_k))
        return np.array(j_k_const)  # Convert to numpy array
    #################################################

    x_0 = 0  
    j_opt = obj_fun(5)  

    num_iterations = 10
    stepsize = .1
    j_k_opt = gradient_descent_optimalstep_by_bisection(obj_fun, grad_fun, x_0, j_opt, num_iterations)
    j_k_const = gradient_descent_constantstep(obj_fun, grad_fun, x_0, j_opt, stepsize, num_iterations)
    # from minimization_methods.gradient_descent import optimalStep, constantStep
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    # j_k_opt = optimalStep(obj_fun, grad_fun, 0, (), options={"maxiter": 10})
    # j_k_const = constantStep(obj_fun, grad_fun, 0, (), options={"maxiter": 10})
    # j_opt = obj_fun(1)

    Visualizer.visualize(ax, j_k_opt, j_opt)
    Visualizer.visualize(ax, j_k_const, j_opt)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('|J(x_k) - J*|')
    ax.set_title('Convergence of quasinewtonmethods')
    ax.legend(['Optimal Step', 'Constant Step'])
    ax.grid(True)
    plt.show()
