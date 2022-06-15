"""Implements common tools of loader functions, including a decorator to
ensure correct loader function signature.
"""


def add_loader(*, TargetCls: type, omit_self: bool = True):
    """This decorator should be used to specify loader methods.

    Args:
        TargetCls: The return type of the load function. This is stored as an
            attribute of the decorated function.
        omit_self (bool, optional): If True (default), the decorated method
            will not be supplied with the ``self`` object instance, thus being
            equivalent to a class method.
    """

    def load_func_decorator(func):
        """This decorator sets the load function's ``TargetCls`` attribute."""

        def load_func(instance, *args, **kwargs):
            """Calls the load function, either with or without ``self``"""
            load_func.__doc__ = func.__doc__
            if omit_self:
                # class method
                return func(*args, **kwargs)
            # regular method
            return func(instance, *args, **kwargs)

        # Set the target class as function attribute
        load_func.TargetCls = TargetCls

        # Carry over the docstring of the decorated function
        load_func.__doc__ = func.__doc__

        return load_func

    return load_func_decorator
