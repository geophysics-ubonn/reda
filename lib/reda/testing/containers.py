"""Dummy data containers for testing purposes."""

import pandas as pd
import numpy as np
import reda

ERTContainer = reda.ERT()
df = pd.DataFrame(columns=list("ABMNR"))
df.A = np.arange(1, 23)
df.B = df.A + 1
df.M = df.A + 2
df.N = df.B + 2
np.random.seed(0)
df.R = np.random.randn(len(df.R))
ERTContainer.df = df
