#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tempfile
import zipfile
import pkg_resources

import reda

import matplotlib.pylab as plt


def prepare_4d_data():
    """Unzip included 4D data into a temp directory.

    Returns
    -------
    tempdir : string
        Temporary directory
    """
    outdir = tempfile.mkdtemp()
    filename = pkg_resources.resource_filename(
        'reda.testing', 'data/seit_test_data.zip'
    )
    mzip = zipfile.ZipFile(filename)
    mzip.extractall(outdir)
    return outdir

def load_data_into_container(datadir):
    seit = reda.sEIT()
    for nr in range(0, 4):
        seit.import_crtomo(
            directory=datadir + '/modV_0{}_noisy/'.format(nr),
            timestep=nr
        )
    seit.compute_K_analytical(spacing=1)
    return seit


def test_load_4d_data():
    datadir = prepare_4d_data()
    seit = load_data_into_container(datadir)


def test_check_4d_data():
    datadir = prepare_4d_data()
    seit = load_data_into_container(datadir)
    assert seit.nr_frequencies == 15
    assert seit.nr_timesteps == 4


def test_histogram_plotting():
    datadir = prepare_4d_data()
    seit = load_data_into_container(datadir)

    # frequencies into subplots
    name, figs = seit.plot_histograms('rho_a', 'frequency')
    # expect four figures, one for each timestep
    assert len(figs) == 4
    fig = figs[sorted(figs.keys())[0]]
    assert len(fig.axes) == 16
    # the last subplot should be hidden because we have only 15 frequencies
    assert fig.axes[-1].get_visible() is False
    # close the figures
    [plt.close(fig) for fig in figs.values()]

    # run the other trafos once
    name, figs = seit.plot_histograms('rho_a', 'frequency', log10=True)
    [plt.close(fig) for fig in figs.values()]
    name, figs = seit.plot_histograms('rho_a', 'frequency', lin_and_log10=True)
    [plt.close(fig) for fig in figs.values()]

    name, figs = seit.plot_histograms('rho_a', 'timestep')
    # expect 15 frequency plot, with four subplots each
    assert len(figs) == 15
    # get first figure
    fig = figs[sorted(figs.keys())[0]]
    assert len(fig.axes) == 4

    [plt.close(fig) for fig in figs.values()]
    # delete three frequency
    seit.data = seit.data.query('frequency != 0.001')
    seit.data = seit.data.query('frequency != 1.0')
    seit.data = seit.data.query('frequency != 1000.0')

    name, figs = seit.plot_histograms('rho_a', 'frequency')
    assert len(figs) == 4
    [plt.close(fig) for fig in figs.values()]

    name, figs = seit.plot_histograms('rho_a', 'timestep')
    # expect 15 frequency plot, with four subplots each
    assert len(figs) == 12
    [plt.close(fig) for fig in figs.values()]

    # take only the first timestep
    seit.data = seit.data.query('timestep == 1 and frequency == 1')
    name, figs = seit.plot_histograms('rho_a', 'timestep')
