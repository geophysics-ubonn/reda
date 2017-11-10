#!/usr/bin/env python
import IPython
IPython
import sys
import IPython.core.ultratb as ultratb
sys.excepthook = ultratb.VerboseTB(
    call_pdb=True,
)

import pandas as pd
pd.set_option('display.width', 1000)

import edf
container = edf.ERT()
container.import_syscal_dat('data_normal.txt')
container.import_syscal_dat('data_reciprocal.txt', reciprocals=48)

# container.print_log()

container.query('R < 0.5')
# container.print_data_journal()

container.query('R > -0.5')
# container.print_data_journal()
# container.print_log()

import IPython
IPython.embed()
