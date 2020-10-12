MP_STATUS_CHECK_INTERVAL = 5.0

class ExceptionWrapper(object):
    def __init__(self, exc):
        self._ext = 'ExceptionWrapper {}'.format(exc)

from . import worker, load_local