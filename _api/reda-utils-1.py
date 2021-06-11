import matplotlib as mpl
import matplotlib.pyplot as plt
from reda.utils.mpl import mpl_get_cb_bound_next_to_plot
fig, ax = plt.subplots()
fig.subplots_adjust(right=0.8)
plt_obj = ax.plot([1, 2, 3], [1, 2, 3], '.-')
cb_pos = mpl_get_cb_bound_next_to_plot(ax)
ax1 = fig.add_axes(cb_pos, frame_on=True)
cmap = mpl.cm.jet_r
norm = mpl.colors.Normalize(vmin=float(23), vmax=float(33))
cb1 = mpl.colorbar.ColorbarBase(
    ax1,
    cmap=cmap,
    norm=norm,
    orientation='vertical'
)
cb1.locator = mpl.ticker.FixedLocator([23, 28, 33])
cb1.update_ticks()
# cb1.ax.artists.remove(cb1.outline)
