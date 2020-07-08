import numpy as np


def exponential_random_variable(lambda_param, size):
    uniform_random_variable = np.random.uniform(low=0.0, high=1.0, size=size)
    return (-1 / lambda_param) * np.log(1 - uniform_random_variable)


