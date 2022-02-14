from time import time
from functools import wraps


def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        print("func:{}\n took: {} sec".format(f.__name__, te - ts))
        return result

    return wrap
