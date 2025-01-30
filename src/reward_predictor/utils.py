import functools
import numpy as np

def function_wrapper(func, *args, **kwargs):
    """
    Wraps a function with its arguments, returning a callable that can be used in multiprocessing.

    Args:
        func (callable): The function to wrap.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        callable: A function pointer that takes no arguments and calls the original function with provided arguments.
    """
    return functools.partial(func, *args, **kwargs)

def corrcoef(dist_a, dist_b):
    """Returns a scalar between 1.0 and -1.0. 0.0 is no correlation. 1.0 is perfect correlation"""
    dist_a = np.copy(dist_a)  # Prevent np.corrcoef from blowing up on data with 0 variance
    dist_b = np.copy(dist_b)
    dist_a[0] += 1e-12
    dist_b[0] += 1e-12
    return np.corrcoef(dist_a, dist_b)[0, 1]
