"""
Module for establishing a pacing function for data-driven curriculum learning.
Used by the CurriculumSampler class to determine the upper limit of the sampling difficulty.
"""

from typing import Callable
import numpy as np

def get_pacing_fn(
    pacing_fn_name: str,
    total_steps: int,
    start_percent: float,
    end_percent: float,
    starting_difficulty: float = 0.2,
    max_difficulty: float = 1.0,
    growth_rate_c: float = 10,
) -> Callable[[int], float]:
    """
    Modified from: https://github.com/google-research/understanding-curricula/blob/main/utils/utils.py

    Args:
        * pacing_fn_name (str): The name of the pacing function to use.
        * total_steps (int): The total number of steps in the training process.
        * start_percent (float): The percentage of steps from the total number of steps that
            have been taken before we begin increasing the data difficulty
        * end_percent (float): The percentage of steps from the total number of steps that
            have been taken after which we stop increasing the data difficulty.

        * starting_difficulty (float): The starting difficulty of the dataset as a percentile of
            the dataset's difficulty. A value of 0.2 means that initially, we sample from the
            bottom 20% difficult examples.
        * max_difficulty (float): The maximum difficulty of the dataset as a percentile of
            the dataset's difficulty. A value of 1.0 means that the maximum difficulty we
            can sample is the maximum difficulty in the dataset.

    Returns:
        * (callable): A function that takes in the current step and returns the number of
            data points to use.

    """

    assert (
        start_percent < end_percent
    ), f"For the Pacing Fn: start_percent ({start_percent}) must be less than end_percent ({end_percent})"

    step_start = start_percent * total_steps
    step_end = end_percent * total_steps

    num_steps = int(step_end - step_start)

    if pacing_fn_name == "linear":
        rate = (max_difficulty - starting_difficulty) / (num_steps)

        def _linear_function(step: int):
            if step < step_start:
                return starting_difficulty

            step_diff = step - step_start

            return float(
                min(rate * step_diff + starting_difficulty, max_difficulty)
            )

        return _linear_function

    elif pacing_fn_name == "quad":
        rate = (max_difficulty - starting_difficulty) / (num_steps) ** 2

        def _quad_function(step):
            if step < step_start:
                return starting_difficulty

            step_diff = step - step_start

            return float(
                min(
                    rate * step_diff ** 2 + starting_difficulty, max_difficulty
                )
            )

        return _quad_function

    elif pacing_fn_name == "root":
        rate = (max_difficulty - starting_difficulty) / (num_steps) ** 0.5

        def _root_function(step):
            if step < step_start:
                return starting_difficulty

            step_diff = step - step_start

            return float(
                min(
                    rate * step_diff ** 0.5 + starting_difficulty,
                    max_difficulty,
                )
            )

        return _root_function

    elif pacing_fn_name == "step":

        def _step_function(step):
            if step < step_end:
                return starting_difficulty
            else:
                return max_difficulty

        return _step_function

    elif pacing_fn_name == "exp":
        import numpy as np

        c = 10
        tilde_b = starting_difficulty
        tilde_a = num_steps
        rate = (max_difficulty - tilde_b) / (np.exp(c) - 1)
        constant = c / tilde_a

        def _exp_function(step):
            if step < step_start:
                return starting_difficulty

            step_diff = step - step_start

            return float(
                min(
                    rate * (np.exp(step_diff * constant) - 1) + tilde_b,
                    max_difficulty,
                )
            )

        return _exp_function

    elif pacing_fn_name == "log":
        import numpy as np

        #c = 10
        c = growth_rate_c
        tilde_b = starting_difficulty
        tilde_a = num_steps
        ec = np.exp(-c)
        N_b = max_difficulty - tilde_b

        def _log_function(step):
            if step < step_start:
                return starting_difficulty

            step_diff = step - step_start

            return min(
                N_b * (1 + (1.0 / c) * np.log(step_diff / tilde_a + ec))
                + tilde_b,
                max_difficulty,
            )

        return _log_function


    else:
        # If no pacing function is specified, set the hardest difficulty from the beginning.
        return lambda step: 1.0


"""
class LogPacingFn:
    def __init__(self,
                 total_steps: int,
                 start_percent: float,
                 end_percent: float,
                 starting_difficulty: float = 0.2,
                 max_difficulty: float = 1.0):
        self.total_steps = total_steps
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.starting_difficulty = starting_difficulty
        self.max_difficulty = max_difficulty

        self.step_start = start_percent * total_steps
        self.step_end = end_percent * total_steps
        self.num_steps = int(self.step_end - self.step_start)
        self.c = 10
        self.tilde_b = starting_difficulty
        self.tilde_a = self.num_steps
        self.ec = np.exp(self.c*-1)
        self.N_b = max_difficulty - self.tilde_b

    def __call__(self, step: int):


"""







"""

Code from chatGPT to convert pacing functions to a graph


import numpy as np
import matplotlib.pyplot as plt

# Define the pacing functions
def linear_pacing(step, d_start, d_max, N, x_start):
    rate = (d_max - d_start) / N
    return min(rate * (step - x_start) + d_start, d_max)

def quadratic_pacing(step, d_start, d_max, N, x_start):
    rate = (d_max - d_start) / N**2
    return min(rate * (step - x_start)**2 + d_start, d_max)

def root_pacing(step, d_start, d_max, N, x_start):
    rate = (d_max - d_start) / np.sqrt(N)
    return min(rate * np.sqrt(step - x_start) + d_start, d_max)

def step_pacing(step, d_start, d_max, x_end):
    return d_start if step < x_end else d_max

def exponential_pacing(step, d_start, d_max, N, x_start):
    c = 10
    rate = (d_max - d_start) / (np.exp(c) - 1)
    constant = c / N
    return min(rate * (np.exp((step - x_start) * constant) - 1) + d_start, d_max)

def logarithmic_pacing(step, d_start, d_max, N, x_start):
    c = 10
    ec = np.exp(-c)
    Nb = d_max - d_start
    return min(Nb * (1 + (1.0 / c) * np.log((step - x_start) / N + ec)) + d_start, d_max)

# Define the parameters
d_start = 0
d_max = 1
N = 100
x_start = 0
x_end = 50
steps = np.arange(0, N + 20, 1)  # Steps from 0 to N + 20 for visualization

# Generate the pacing function values
linear_values = [linear_pacing(step, d_start, d_max, N, x_start) for step in steps]
quadratic_values = [quadratic_pacing(step, d_start, d_max, N, x_start) for step in steps]
root_values = [root_pacing(step, d_start, d_max, N, x_start) for step in steps]
step_values = [step_pacing(step, d_start, d_max, x_end) for step in steps]
exponential_values = [exponential_pacing(step, d_start, d_max, N, x_start) for step in steps]
logarithmic_values = [logarithmic_pacing(step, d_start, d_max, N, x_start) for step in steps]

# Plot the pacing functions
plt.figure(figsize=(10, 6))
plt.plot(steps, linear_values, label='Linear', linestyle='--')
plt.plot(steps, quadratic_values, label='Quadratic', linestyle='-.')
plt.plot(steps, root_values, label='Root', linestyle=':')
plt.plot(steps, step_values, label='Step', linestyle='-')
plt.plot(steps, exponential_values, label='Exponential', linestyle='-')
plt.plot(steps, logarithmic_values, label='Logarithmic', linestyle='-')

plt.xlabel('Step')
plt.ylabel('Difficulty')
plt.title('Pacing Functions')
plt.legend()
plt.grid(True)
plt.show()


"""