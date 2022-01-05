#!/usr/bin/env python
"""
https://docs.h5py.org/en/stable/quick.html#quick
"""
import reda


if __name__ == '__main__':
    ert = reda.ERT()
    ert.tsert_summary('data.h5')
    ert.import_tsert('data.h5')

    # obj = format_tsert('data.h5')
    # f = obj._open_file('r')
    # obj.summary()
    # # metadata = obj.add_metadata()
    # # metadata = obj.load_metadata()
    # # print(metadata)

    # data = obj.load_all_timesteps('base')
