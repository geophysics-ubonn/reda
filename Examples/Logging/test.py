#!/usr/bin/env python


import queue
log_queue = queue.Queue()
log_list = []
import logging
import logging.handlers


class ListHandler(logging.Handler):  # Inherit from logging.Handler
    def __init__(self, log_list):
        # run the regular Handler __init__
        logging.Handler.__init__(self)
        # Our custom argument
        self.log_list = log_list

    def emit(self, record):
        # record.message is the log message
        self.log_list.append(record)


logger = logging.getLogger(__name__)
# logger.addHandler(logging.handlers.QueueHandler(log_queue))
handler = ListHandler(log_list)
handler.setFormatter('%(asctime)s %(message)s')
logger.addHandler(handler)
logger.error('Blup')
# log_queue.qsize()

import IPython
IPython.embed()
