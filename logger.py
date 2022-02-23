from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from datetime import datetime
import os

class LOG_LEVELS:
    #highest priority means it will be logged
    DEBUG       = 1 
    INFO        = 2
    WARN        = 3
    CRITICAL    = 4
    ERROR       = 5

    @staticmethod
    def get_name(level):
        if level == 1:
            return 'DEBUG'
        elif level == 2:
            return 'INFO'
        elif level == 3:
            return 'WARN'
        elif level == 4:
            return 'CRITICAL'
        elif level == 5:
            return 'ERROR'
        else:
            return 'UNKNOWN'
        
class MyLogger():
    def __init__(self, name, log_file_path, use_stdout=False, overwrite=True, log_level=LOG_LEVELS.DEBUG):
        self.name = name
        self.log_file_path = log_file_path
        self.stdout = use_stdout
        self.level = log_level
        #only zero rank process can truncate the file
        if overwrite and rank_zero_only.rank == 0:
            #truncate the file
            with open(self.log_file_path, 'w') as log_file:
                pass
    
    @rank_zero_only
    def _log_msg(self, msg, level):
        if level >= self.level:
            present_time = datetime.now()
            msg_str = '%s [%s] %s' % (present_time.strftime('%m/%d/%Y %I:%M:%S %p'), LOG_LEVELS.get_name(level), msg)
            if self.stdout:
                print(msg_str)
            with open(os.path.abspath(self.log_file_path), 'a+') as log_file:
                log_file.write(msg_str+"\n")

    @rank_zero_only
    def info(self, msg):
        self._log_msg(msg, LOG_LEVELS.INFO)

    @rank_zero_only
    def critical(self, msg):
        self._log_msg(msg, LOG_LEVELS.CRITICAL)
        
    @rank_zero_only
    def debug(self, msg):
        self._log_msg(msg, LOG_LEVELS.DEBUG)
    
    @rank_zero_only
    def warn(self, msg):
        self._log_msg(msg, LOG_LEVELS.WARN)

    @rank_zero_only
    def error(self, msg):
        self._log_msg(msg, LOG_LEVELS.ERROR)

def logger_wrapper(logger, logger_name):
    if isinstance(logger, MyLogger):
        return logger
    base_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    return MyLogger(logger_name, os.path.join(log_dir, "%s.log"%logger_name), use_stdout=True, overwrite=True)