import functools


def enable_result_transforms(func):
    """Decorator that tries to use the object provided using a kwarg called
    'electrode_transformator' to transform the return values of an import
    function. It is intended to be used to transform electrode numbers and
    locations, i.e. for use in roll-along-measurement schemes.

    The transformator object must have a function .transform, which takes three
    parameters: data, electrode, topography and returns three correspondingly
    transformed objects.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_transformator = kwargs.pop('electrode_transformator', None)
        data, electrodes, topography = func(*args, **kwargs)
        if func_transformator is not None:
            data_transformed, electrodes_transformed, \
                topography_transformed = func_transformator.transform(
                    data, electrodes, topography
                )
            return data_transformed, electrodes_transformed, \
                topography_transformed
        else:
            return data, electrodes, topography
    return wrapper


