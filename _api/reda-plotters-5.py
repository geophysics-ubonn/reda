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
measurements2 = np.random.random(configs.shape[0])

import pandas as pd
df = pd.DataFrame(configs, columns=['A', 'B', 'M', 'N'])
df['measurements'] = measurements
df['measurements2'] = measurements2

from reda.plotters.pseudoplots import plot_pseudosection_type2

fig, axes = plt.subplots(1, 2)

plot_pseudosection_type2(
    df,
    column='measurements',
    ax=axes[0],
    cblabel='this label',
    xlabel='xlabel',
    ylabel='ylabel',
)
plot_pseudosection_type2(
    df,
    column='measurements2',
    ax=axes[1],
    cblabel='measurement 2',
    xlabel='xlabel',
    ylabel='ylabel',
)
fig.tight_layout()