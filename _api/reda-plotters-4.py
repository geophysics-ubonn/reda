import numpy as np
configs = np.array((
    (1, 2, 4, 3),
    (1, 2, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 5, 4),
    (2, 3, 6, 5),
    (3, 4, 6, 5),
))
measurements = np.random.random(configs.shape[0])
import pandas as pd
df = pd.DataFrame(configs, columns=['A', 'B', 'M', 'N'])
df['measurements'] = measurements

from reda.plotters.pseudoplots import plot_pseudosection_type2
fig, ax, cb = plot_pseudosection_type2(
   dataobj=df,
   column='measurements',
)