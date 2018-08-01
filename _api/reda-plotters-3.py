from reda.plotters.plots2d import plot_pseudodepths
# define a few measurements
import numpy as np
configs = np.array((
    (4, 7, 5, 6),
    (3, 8, 5, 6),
    (2, 9, 5, 6),
    (1, 10, 5, 6),
))
# plot
fig, axes = plot_pseudodepths(configs, nr_electrodes=10, spacing=1,
                              ctypes=['schlumberger', ])