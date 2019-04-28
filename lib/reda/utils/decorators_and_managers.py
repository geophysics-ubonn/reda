"""Utility class for context managers and decorators
"""


def append_doc_of(fun):
    def decorator(f):
        f.__doc__ += fun.__doc__
        return f

    return decorator


def prepend_doc_of(fun):
    def decorator(f):
        f.__doc__ = fun.__doc__ + f.__doc__
        return f

    return decorator


class LogDataChanges():
    """Context manager that observes the DataFrame of a data container for
    changes in the number of rows.

    Examples
    --------

    >>> from reda.testing.containers import ERTContainer
    >>> from reda.containers.ERT import LogDataChanges
    >>> with LogDataChanges(ERTContainer):
    ...     # now change the data
    ...     ERTContainer.data.loc[0, "r"] = 22
    ...     ERTContainer.data.query("r < 10", inplace=True)
    >>> # ERTContainer.print_log()
    2... - root - INFO - Data change from 22 to 21

    """

    def __init__(
            self,
            container,
            filter_action='default',
            filter_query="",
    ):
        self.container = container
        self.logger = container.logger
        self.filter_action = filter_action
        self.data_size_before = None
        self.data_size_after = None
        self.filter_query = filter_query

    def __enter__(self):
        if self.container.data is None:
            self.data_size_before = 0
        else:
            self.data_size_before = self.container.data.shape[0]
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        # make sure we do not execute more code if an exception was raised in
        # the context
        # See:
        # https://docs.python.org/3/reference/datamodel.html#object.__exit__
        if(
            exc_type is not None or
            exc_value is not None or
            traceback is not None):
            return
        self.data_size_after = self.container.data.shape[0]
        self.logger.info(
            'Data change from {0} to {1}'.format(
                self.data_size_before,
                self.data_size_after,
            ),
            extra={
                'filter_action': self.filter_action,
                'df_size_before': self.data_size_before,
                'df_size_after': self.data_size_after,
                'filter_query': self.filter_query,
            },
        )
