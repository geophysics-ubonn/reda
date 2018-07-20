# -*- coding: utf8 -*-
"""

"""
import os
from urllib import request
import zipfile

import pandas as pd
# if this is set to a valid directory path, try to fetch data from here
# The path could be a local copy of the data repository
use_local_data_repository = None

repository_url = ''.join((
    'https://raw.githubusercontent.com/geophysics-ubonn/'
    'reda_examples_mw/master/'
))
inventory_filename = 'inventory.dat'


def download_data(identifier, outdir):
    """Download data from a separate data repository for testing.

    Parameters
    ----------
    identifier: string
        The identifier used to find the data set
    outdir: string
        unzip the data in this directory
    """
    # determine target
    if use_local_data_repository is not None:
        url_base = 'file:' + request.pathname2url(
            use_local_data_repository + os.sep)
    else:
        url_base = repository_url

    print('url_base: {}'.format(url_base))
    url = url_base + inventory_filename
    # download inventory file
    filename, headers =request.urlretrieve(url)

    df = pd.read_csv(
        filename,
        delim_whitespace=True,
        comment='#',
        header=None,
        names=['identifier', 'rel_path'],
    )

    # find relative path to data file
    rel_path_query = df.query('identifier == "{}"'.format(identifier))
    if rel_path_query.shape[0] == 0:
        raise Exception('identifier not found')
    rel_path = rel_path_query['rel_path'].values[0]

    # download the file
    url = url_base + rel_path
    print('data url: {}'.format(url))
    filename, headers =request.urlretrieve(url)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    zip_obj = zipfile.ZipFile(filename)
    zip_obj.extractall(outdir)
