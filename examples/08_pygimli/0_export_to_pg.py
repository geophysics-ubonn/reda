#!/usr/bin/env python
import reda
ert = reda.ERT()
ert.import_syscal_bin(
    'raw_data/p1.1_nor_dd/data.bin', spacing=2.5, check_meas_nums=False
)
pg_scheme = ert.export_to_pygimli_scheme()
