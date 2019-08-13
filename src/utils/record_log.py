from src.utils.configs import cfg
import os
import time
import logging


class RecordLog(object):
    def __init__(self, file_name = 'log.txt'):

        build_time = '-'.join(time.asctime(time.localtime(time.time())).strip().split(' ')[1:-1])
        build_time = '-'.join(build_time.split(':'))
        # log_file_name = str(cfg.skip_window)+'_'+str(cfg.max_epoch)+'_'\
        #                 +str(cfg.train_batch_size)+'_'+build_time

        log_file_name = cfg.log_name + '_'+build_time

        path = os.path.join(cfg.log_dir or cfg.standby_log_dir, log_file_name+"_"+file_name)
        date_format = '%m/%d/%Y %I:%M:%S %p'
        format_str = '%(asctime)s %(message)s'
        level = logging.DEBUG

        logging.basicConfig(filename = path,
                            datefmt = date_format,
                            format = format_str,
                            level = level)
        logging.info('model parameters:--------------------')
        logging.info(cfg.log_name)

    def add(self, content = '-'*30, is_print = True):
        if is_print:
            print(content)
        logging.info(content)

    def done(self):
        self.add('Done')


_logger = RecordLog()