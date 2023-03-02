'''This gradient_descent.py file calculates
gradient descent and stochastic gradient descent
for given function values'''
import numpy as np


def min_least_square(y, y_hat, n):
    minls = np.sum(np.square(y-y_hat))/n
    return minls


def grad_desc(n=1000, a=4.0, b=2.0, sigma=0.05,
              x_start=-5, x_end=5, learn_rate=0.01):
    if learn_rate <= 0:
        raise ValueError("Requires positive learning rate.")
    # Initializing return arrays
    loss = []
    slope = []
    intrcpt = []
    x = np.linspace(x_start, x_end, n)
    y = a * x + b
    epsilon = np.random.normal(0, sigma, n)
    y_true = y + epsilon
    # Value of a will be slope
    a = 0
    # Value of b will be intercept
    b = 0
    for i in range(n):
        y_hat = a * x + b
        loss.append(min_least_square(y_true, y_hat, n))
        # Calculate slope and intercept
        a -= learn_rate * (np.sum(2 * x * (-y_true + a * x + b)) / n)
        b -= learn_rate * (np.sum(-2 * y_true + 2 * a * x + 2 * b) / n)
        # Filling arrays
        slope.append(a)
        intrcpt.append(b)

    return loss, slope, intrcpt


def stoch_grad_desc(n=1000, a=4.0, b=2.0, sigma=0.05,
                    x_start=-5, x_end=5, learn_rate=0.01):
    if learn_rate <= 0:
        raise ValueError("Requires positive learning rate.")
    # Initializing return arrays
    loss = []
    slope = []
    intrcpt = []
    x = np.linspace(x_start, x_end, n)
    y = a * x + b
    epsilon = np.random.normal(0, sigma, n)
    y_true = y + epsilon
    # Value of a will be slope
    a = 0
    # Value of b will be intercept
    b = 0
    for i in range(n):
        y_hat = a * x + b
        index = np.random.randint(0, n)
        loss.append(min_least_square(y_true, y_hat, n))
        # Calculate slope and intercept
        a -= learn_rate * (2 * x[index] * (-y_true[index] + a * x[index] + b))
        b -= learn_rate * (-2 * y_true[index] + 2 * a * x[index] + 2 * b)
        # Filling arrays
        slope.append(a)
        intrcpt.append(b)

    return loss, slope, intrcpt
