# -*- coding: utf-8 -*-
"""
Plotting a pseudosection
========================

In this example we show how to plot different forms of pseudosections.

.. warning:: This is just a dummy example.
"""

################################################################################
# First we import reda and create a dummy ERT container

import reda
import matplotlib.pyplot as plt
import numpy as np

################################################################################
# Find reciprocals
ert = reda.ERT()
ert.data = reda.utils.norrec.get_test_df()
ert.compute_reciprocal_errors()

import pandas as pd
df = pd.DataFrame()

################################################################################
# Now we make a random figure

x = np.linspace(-1, 2, 100)
y = np.exp(x)

plt.figure()
plt.plot(x, y)
plt.xlabel('$x$')
plt.ylabel('$\exp(x)$')

plt.figure()
plt.plot(x, -np.exp(-x))
plt.xlabel('$x$')
plt.ylabel('$-\exp(-x)$')

plt.show()
