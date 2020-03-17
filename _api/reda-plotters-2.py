from reda.plotters.plots2d import plot_pseudodepths
# define a few measurements
import numpy as np
configs = np.array((
    (1, 2, 4, 3),
    (1, 2, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 5, 4),
    (2, 3, 6, 5),
    (3, 4, 6, 5),
))
# plot
fig, axes = plot_pseudodepths(configs, nr_electrodes=6, spacing=1,
                              ctypes=['dd', ])