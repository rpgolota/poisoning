import logging

class PoisonLogger:
    
    GENERAL_LOGGER = logging.getLogger('PoisonLog')
    GENERAL_LOGGER.setLevel(logging.DEBUG)
    GENERAL_HANDLER = logging.StreamHandler()
    FILE_HANDLER = logging.FileHandler('poison.log', mode='w')    
    
    GENERAL_FMT = ('[%(asctime)s.%(msecs)03d][%(name)s][%(levelname)-8s][%(filename)s][%(funcName)s][%(lineno)d] : %(message)s', '%Y/%m/%d][%H:%M:%S')
    GENERAL_FOMATTER = logging.Formatter(*GENERAL_FMT)
    
    GENERAL_HANDLER.setFormatter(GENERAL_FOMATTER)
    FILE_HANDLER.setFormatter(GENERAL_FOMATTER)
    
    GENERAL_LOGGER.addHandler(GENERAL_HANDLER)
    GENERAL_LOGGER.addHandler(FILE_HANDLER)

    debug = GENERAL_LOGGER.debug
    info = GENERAL_LOGGER.info
    warn = GENERAL_LOGGER.warn
    error = GENERAL_LOGGER.error
    critical = GENERAL_LOGGER.critical
    exception = GENERAL_LOGGER.exception
    
    def disable_debug():
        PoisonLogger.debug = PoisonLogger.nothing
    
    def enable_debug():
        PoisonLogger.debug = PoisonLogger.GENERAL_LOGGER.debug
    
    def nothing(*args, **kwargs):
        pass