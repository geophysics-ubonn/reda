from pandas.api.types import is_datetime64_any_dtype


def ensure_dateteim64_is_in_ns(dataframe):
    """
    Find all columns that are of type datetime64 and convert them to [ns]
    resolution.
    This prevents nasty import/export errors.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe to check and fix

    """
    for column in dataframe.columns:
        if is_datetime64_any_dtype(dataframe[column]):
            dataframe[column] = dataframe[column].astype('datetime64[ns]')
