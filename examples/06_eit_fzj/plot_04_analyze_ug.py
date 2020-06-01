#!/usr/bin/env python
"""
EIT-FZJ: Analyze DC voltages at electrodes during injections
------------------------------------------------------------

"""
import matplotlib.pylab as plt
import reda.importers.eit_fzj as eit_fzj
import IPython

IPython

adc_data = eit_fzj.get_adc_data('data_eit_fzj_2013_ug/eit_data_mnu0.mat')

frequencies = list(adc_data.swaplevel(0, 2).groupby('frequency').groups.keys())

fig, axes = plt.subplots(
    len(frequencies), 1, figsize=(20 / 2.54, 100 / 2.54),
    sharex=True, sharey=True)

for nr, frequency in enumerate(frequencies):
    ax = axes[nr]
    im = ax.imshow(
        adc_data.swaplevel(
            0, 1, axis=1
        ).swaplevel(0, 2, axis=0)['Ug3_1'].loc[frequency].values)
    ax.set_aspect('auto')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'Ug [V]')

for ax in axes[:]:
    ax.set_ylabel('injection number')

axes[-1].set_xlabel('ADC Channel')

fig.tight_layout()
# fig.savefig('plot_ug3-1.pdf', dpi=300)
plt.show(fig)
