"""Implements common tools of loader functions, including a decorator to
ensure correct loader function signature.
"""

def add_loader(*, TargetCls, omit_self: bool=True):
    """This decorator should be used to specify loader functions.
    
    Args:
        TargetCls: Description
        omit_self (bool, optional): If True (default), the decorated method
            will not be supplied with the `self` object instance
    """
    def load_func_decorator(func):
        """This decorator sets the load function's `TargetCls` attribute."""
        def load_func(instance, *args, **kwargs):
            """Calls the load function, either as with or without `self`."""
            if omit_self:
                return func(*args, **kwargs)
            # not as static method
            return func(instance, *args, **kwargs)

        # Set the target class as function attribute 
        load_func.TargetCls = TargetCls
        return load_func
    return load_func_decorator
