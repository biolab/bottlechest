"Test that no slow functions creep in where they don't belong."

import numpy as np
import bottlechest as bn


def arrayaxis(dtypes=bn.dtypes):
    "Iterator that yield arrays and axis to use for unit testing."
    ss = {}
    ss[1] = {'size':  4, 'shapes': [(4,)]}
    ss[2] = {'size':  6, 'shapes': [(2,3)]}
    for ndim in ss:
        size = ss[ndim]['size']
        shapes = ss[ndim]['shapes']
        for dtype in dtypes:
            a = np.arange(size, dtype=dtype)
            if not issubclass(a.dtype.type, np.inexact):
                for shape in shapes:
                    a = a.reshape(shape)
                    for axis in list(range(-a.ndim, a.ndim)) + [None]:
                        yield a.copy(), axis

def fast_checker(selector, mode='func'):
    for arr, axis in arrayaxis():
        if mode == 'func':
            func, a = selector(arr, axis)
        elif mode == 'move':
            if axis is not None:
                func, a = selector(arr, axis)
            else:
                func = np.sum
        elif mode == 'replace':
            func = selector(arr)
        else:
            raise ValueError("`mode` value not recognized.")
        if 'slow' in func.__name__:
            raise AssertionError("slow version of func unexpectedly called.")

# Functions -----------------------------------------------------------------

def test_median_selector():
    "Test median_selector."
    fast_checker(bn.func.median_selector)

def test_nanmedian_selector():
    "Test nanmedian_selector."
    fast_checker(bn.func.nanmedian_selector)

def test_nansum_selector():
    "Test nansum_selector."
    fast_checker(bn.func.nansum_selector)

def test_nanmin_selector():
    "Test nanmin_selector."
    fast_checker(bn.func.nanmin_selector)

def test_nanmax_selector():
    "Test nanmax_selector."
    fast_checker(bn.func.nanmax_selector)

def test_nanmean_selector():
    "Test nanmean_selector."
    fast_checker(bn.func.nanmean_selector)

def test_nanstd_selector():
    "Test nanstd_selector."
    fast_checker(bn.func.nanstd_selector)

def test_nanargmin_selector():
    "Test nanargmin_selector."
    fast_checker(bn.func.nanargmin_selector)

def test_nanargmax_selector():
    "Test nanargmax_selector."
    fast_checker(bn.func.nanargmax_selector)

def test_nanvar_selector():
    "Test nanvar_selector."
    fast_checker(bn.func.nanvar_selector)

def test_rankdata_selector():
    "Test rankdata_selector."
    fast_checker(bn.func.rankdata_selector)

def test_nanrankdata_selector():
    "Test nanrankdata_selector."
    fast_checker(bn.func.nanrankdata_selector)

def test_ss_selector():
    "Test ss_selector."
    fast_checker(bn.func.ss_selector)

def test_replace_selector():
    "Test replace_selector."
    fast_checker(bn.func.replace_selector, mode='replace')

def test_anynan_selector():
    "Test anynan_selector."
    fast_checker(bn.func.anynan_selector)

