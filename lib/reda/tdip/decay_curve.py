class DecayCurveObj():
    """Helper class to construct object out of pd.DataFrame. Used to put a sub
    DataFrame in a DataFrame.
    """

    def __init__(self, df):
        self.df = df
