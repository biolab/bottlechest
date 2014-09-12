"countnans template"

from copy import deepcopy
import bottlechest as bn

__all__ = ["countnans"]

FLOAT_DTYPES = [x for x in bn.dtypes if 'float' in x]
INT_DTYPES = [x for x in bn.dtypes if 'int' in x]

# Float dtypes (not axis=None) ----------------------------------------------

floats = {}
floats['dtypes'] = FLOAT_DTYPES
floats['axisNone'] = False
floats['force_output_dtype'] = "float64"
floats['reuse_non_nan_func'] = False

floats['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a,
                              np.ndarray[np.float64_t, ndim=1] weights=None):
    "Count of nans in NDIMd array with dtype=DTYPE along axis=AXIS."
    cdef np.DTYPE_t ai
"""

loop = {}
loop[2] = """
    cdef float fcnt
    cdef int count
    if weights is not None:
        for iINDEX0 in range(nINDEX0):
            fcnt = 0
            for iINDEX1 in range(nINDEX1):
                ai = a[INDEXALL]
                if ai != ai:
                    fcnt += weights[iINDEX1]
            y[INDEXPOP] = fcnt
    else:
        for iINDEX0 in range(nINDEX0):
            count = 0
            for iINDEX1 in range(nINDEX1):
                ai = a[INDEXALL]
                if ai != ai:
                    count += 1
            y[INDEXPOP] = count
    return y"""

floats['loop'] = loop

# Float dtypes (axis=None) --------------------------------------------------

floats_None = deepcopy(floats)
floats_None['axisNone'] = True

loop = {}
loop[1] = """
    if weights is not None:
        if weights.ndim != 1 or weights.shape[0] != nINDEX0:
             raise ValueError("shape of weights does not match the data")
        fcnt = 0
        for iINDEX0 in range(nINDEX0):
            ai = a[INDEXALL]
            if ai != ai:
                fcnt += weights[iINDEX0]
        return fcnt
    else:
        count = 0
        for iINDEX0 in range(nINDEX0):
            ai = a[INDEXALL]
            if ai != ai:
                count += 1
        return count"""

loop[2] = """
    if weights is not None:
        if weights.shape[0] != (nINDEX0, nINDEX1):
             raise ValueError("shape of weights does not match the data")
        fcnt = 0
        for iINDEX0 in range(nINDEX0):
            for iINDEX1 in range(nINDEX1):
                ai = a[INDEXALL]
                if ai != ai:
                    fcnt += weights[INDEXALL]
        return fcnt
    else:
        count = 0
        for iINDEX0 in range(nINDEX0):
            for iINDEX1 in range(nINDEX1):
                ai = a[INDEXALL]
                if ai != ai:
                    count += 1
        return count"""

floats_None['loop'] = loop

# Int dtypes (not axis=None) ------------------------------------------------


# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "countnans"
slow['signature'] = "arr, weights"
slow['func'] = "bn.slow.countnans(arr, weights, axis=AXIS)"

# Template ------------------------------------------------------------------

countnans = {}
countnans['name'] = 'countnans'
countnans['is_reducing_function'] = True
countnans['cdef_output'] = True
countnans['slow'] = slow
countnans['templates'] = {}
countnans['templates']['float'] = floats
countnans['templates']['float_None'] = floats_None
countnans['pyx_file'] = 'func/%sbit/countnans.pyx'

countnans['main'] = '''"countnans auto-generated from template"

def countnans(arr, weights=None, axis=None):
    """
    Count the undefined elements along given axis.

    Parameters
    ----------
    arr : array_like
        Array in which nans are to be counted. If `arr` is not an array, a
        conversion is attempted.
    weights : {array-like, None}
        Array containing the weights along the specified axis.
        Note: weights, if present, must be of type `float64`
    axis : {int, None}, optional
        Axis along which the nans are counted. The default (axis=None) is to
        count them across the entire array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        The output array is an array of float64 if data is weighted, or of int,
        if it is not.

    Examples
    --------
    >>> bn.countnans(1)
    0.0
    >>> bn.countnans([1])
    0.0
    >>> bn.countnans([1, np.nan])
    1.0
    >>> a = np.array([[1, 4], [1, np.nan]])
    >>> bn.countnans(a)
    1.0
    >>> bn.countnans(a, axis=0)
    array([ 0, 1])

    """
    func, arr, weights = countnans_selector(arr, weights, axis)
    return func(arr, weights)

def countnans_selector(arr, weights, axis):
    """
    Return countnans function and array that matches `arr` and `axis`.

    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.countnans() is in checking that `axis` is within range, converting `arr`
    into an array (if it is not already an array), and selecting the function
    to use to count the nans.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    weights : {array-like, None}
        Array containing the weights along the specified axis.
        Note: weights, if present, must be of type `float64`
    axis : {int, None}
        Axis along which the nans are to be counted.

    Returns
    -------
    func : function
        The countnans function that matches the number of dimensions and dtype
        of the input array and the axis along which you wish to count the nans.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.
    weights : {array-like, None}
        If the input array `weights` is not a ndarray, then `weights` will
        contain the result of converting `weights` into a ndarray.

    """
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)
    if weights is not None and type(weights) is not np.ndarray:
        weights = np.array(weights, copy=False)
    cdef int ndim = PyArray_NDIM(a)
    cdef int dtype = PyArray_TYPE(a)
    if (axis is not None) and (axis < 0):
        axis += ndim
    cdef tuple key = (ndim, dtype, axis)
    try:
        if weights is not None and weights.ndim > 1:
            raise KeyError # fast function supports only 1-dimensional weights
        func = countnans_dict[key]
    except KeyError:
        if axis is not None:
            if (axis < 0) or (axis >= ndim):
                raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = countnans_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a, weights
'''
