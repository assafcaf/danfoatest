import functools
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