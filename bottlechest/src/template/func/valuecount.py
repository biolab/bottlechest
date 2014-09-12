"nanequal template"

from copy import deepcopy
import bottlechest as bn

__all__ = ["valuecount"]

FLOAT_DTYPES = [x for x in bn.dtypes if 'float' in x]
INT_DTYPES = [x for x in bn.dtypes if 'int' in x]

# Float dtypes (not axis=None) ----------------------------------------------

floats = {}
floats['dtypes'] = FLOAT_DTYPES + INT_DTYPES
floats['axisNone'] = True
floats['force_output_dtype'] = 'bool'
floats['reuse_non_nan_func'] = False

floats['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=2] a):
    '''Count the occurrences of each value'''
"""

loop = {}

loop[2] = """\
    if nINDEX0  != 2:
        raise ValueError("valuecount expects an array with shape (2, N)")

    cdef Py_ssize_t src
    for src in range(1, nINDEX1):
        if a[0, src] == a[0, src-1]:
            break
    else:
        return a

    cdef Py_ssize_t dst = src - 1
    for src in range(src, nINDEX1):
        if a[0, src] != a[0, src]:
            break
        if a[0, src] == a[0, dst]:
            a[1, dst] += a[1, src]
        else:
            dst += 1
            a[0, dst] = a[0, src]
            a[1, dst] = a[1, src]
    return a[:, :dst + 1]
"""

floats['loop'] = loop


# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "valuecount"
slow['signature'] = "arr"
slow['func'] = "bn.slow.valuecount(arr)"

# Template ------------------------------------------------------------------

valuecount = {}
valuecount['name'] = 'valuecount'
valuecount['is_reducing_function'] = False
valuecount['cdef_output'] = False
valuecount['slow'] = slow
valuecount['templates'] = {}
valuecount['templates']['float'] = floats
valuecount['pyx_file'] = 'func/%sbit/valuecount.pyx'

valuecount['main'] = '''"valuecount auto-generated from template"

def valuecount(arr):
    """
    Count number of occurrences of each value in array.

    It does so in-place, on a 2-d array of shape (2, N); the first row
    contains values and the second contains weights (1's, if unweighted).
    The array should be sorted by the first rows (e.g. a.sort(axis=1))
    The function "compresses" the table by summing the consecutive values
    in the second row that have the same value in the first row.

    Parameters
    ----------
    x : an array of shape (2, N)

    Returns
    -------
    out : the number of unique values

    Raises
    ------
    ValueError
        If the input is not 2-dimensional

    """
    func, a = valuecount_selector(arr)
    return func(a)

def valuecount_selector(arr):
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)

    cdef int ndim = PyArray_NDIM(a)
    cdef int dtype = PyArray_TYPE(a)
    cdef int i
    cdef tuple key = (ndim, dtype, None)
    try:
        func = valuecount_dict[key]
        return func, a,
    except KeyError:
        pass

    try:
        func = valuecount_slow_dict[None]
    except KeyError:
        tup = (str(ndim), str(a.dtype))
        raise TypeError("Unsupported ndim/dtype (%s/%s)." % tup)
    return func, a
'''
