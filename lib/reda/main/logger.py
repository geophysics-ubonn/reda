"""
set up the logger for EDF
"""
import logging


def setup_logger():
    logging.basicConfig(
        # filename='logfile.log',
        level=logging.DEBUG,
    )
