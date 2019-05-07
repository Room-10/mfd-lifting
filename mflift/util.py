
import sys, os

class cached_property(object):
    def __init__(self, fun):
        self._fun = fun

    def __get__(self, obj, _):
        setattr(obj, self._fun.__name__, self._fun(obj))
        return getattr(obj, self._fun.__name__)

def suppress_stdout():
    sys.stdout = open(os.devnull, 'w')

def restore_stdout():
    sys.stdout = sys.__stdout__
