"""
Set up some logging facilities for reda

We differentiate multiple logging types/targets

1) general logging (using the python logging module)
2) a data processing log (which also uses the logging module), available via
   the container-inherent .logger object
"""
import logging
import datetime


def setup_logger():
    logging.basicConfig(
        # filename='logfile.log',
        level=logging.DEBUG,
    )


class ListHandler(logging.Handler):  # Inherit from logging.Handler
    def __init__(self, log_list):
        # run the regular Handler __init__
        logging.Handler.__init__(self)
        # Our custom argument
        self.log_list = log_list

    def emit(self, record):
        # record.message is the log message
        self.log_list.append(record)


class LoggingClass(object):
    """Set up logging facilities for the containers
    """

    def setup_logger(self):
        """Setup a logger
        """
        self.log_list = []
        handler = ListHandler(self.log_list)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.addHandler(handler)

        logger.setLevel(logging.INFO)

        self.handler = handler
        self.logger = logger

    def print_log(self):
        for record in self.log_list:
            print(self.handler.format(record))

    def print_data_journal(self):
        print('')
        print('--- Data Journal Start ---')
        print('{0}'.format(datetime.datetime.now()))
        for record in self.log_list:
            if hasattr(record, 'filter_action'):
                # print(record)
                if record.filter_action == 'import':
                    print('Data was imported from file {0} '.format(
                        record.filter_query) + '({0} data points)'.format(
                            record.df_size_after - record.df_size_before))
                if record.filter_action == 'filter':
                    print(
                        'A filter was applied with query "{0}".'.format(
                            record.filter_query
                        ) +
                        ' In total {0} records were removed'.format(
                            -record.df_size_after + record.df_size_before
                        )
                    )
        print('--- Data Journal End ---')
        print('')
