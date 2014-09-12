"nanequal template"

from copy import deepcopy
import bottlechest as bn

__all__ = ["nanequal"]

FLOAT_DTYPES = [x for x in bn.dtypes if 'float' in x]
INT_DTYPES = [x for x in bn.dtypes if 'int' in x]

# Float dtypes (not axis=None) ----------------------------------------------

floats = {}
floats['dtypes'] = FLOAT_DTYPES
floats['axisNone'] = False
floats['force_output_dtype'] = 'bool'
floats['reuse_non_nan_func'] = False

floats['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a,
                              np.ndarray[np.DTYPE_t, ndim=NDIM] b):
    "Check whether two arrays are equal, ignoring NaNs, in NDIMd array with dtype=DTYPE along axis=AXIS."
    cdef int f = 1
    cdef np.DTYPE_t ai
"""

loop = {}
loop[2] = """\
    for iINDEX0 in range(nINDEX0):
        f = 1
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            bi = b[INDEXALL]
            if ai != bi and ai == ai and bi == bi:
                y[INDEXPOP] = 0
                f = 0
                break
        if f == 1:
            y[INDEXPOP] = 1
    return y
"""

floats['loop'] = loop

# Float dtypes (axis=None) --------------------------------------------------

floats_None = deepcopy(floats)
floats_None['axisNone'] = True

loop = {}
loop[1] = """\
    for iINDEX0 in range(nINDEX0):
        ai = a[INDEXALL]
        bi = b[INDEXALL]
        if ai != bi and ai == ai and bi == bi:
            return np.bool_(False)
    return np.bool_(True)
"""
loop[2] = """\
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            bi = b[INDEXALL]
            if ai != bi and ai == ai and bi == bi:
                return np.bool_(False)
    return np.bool_(True)
"""

floats_None['loop'] = loop

# Int dtypes (not axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['dtypes'] = INT_DTYPES 

loop = {}
loop[2] = """\
    for iINDEX0 in range(nINDEX0):
        f = 1
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            bi = b[INDEXALL]
            if ai != bi:
                y[INDEXPOP] = 0
                f = 0
                break
        if f == 1:
            y[INDEXPOP] = 1
    return y
"""

ints['loop'] = loop


# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "nanequal"
slow['signature'] = "arr1, arr2"
slow['func'] = "bn.slow.nanequal(arr1, arr2, axis=AXIS)"

# Template ------------------------------------------------------------------

nanequal = {}
nanequal['name'] = 'nanequal'
nanequal['is_reducing_function'] = True
nanequal['cdef_output'] = True
nanequal['slow'] = slow
nanequal['templates'] = {}
nanequal['templates']['float'] = floats
nanequal['templates']['float_None'] = floats_None
nanequal['templates']['int'] = ints
nanequal['pyx_file'] = 'func/%sbit/nanequal.pyx'

nanequal['main'] = '''"nanequal auto-generated from template"

def nanequal(arr1, arr2, axis=None):
    """
    Test whether two array are equal along a given axis, ignoring NaNs.

    Returns single boolean unless `axis` is not ``None``.

    Parameters
    ----------
    arr1 : array_like
        First input array. If `arr` is not an array, a conversion is attempted.
    arr2 : array_like
        Second input array
    axis : {int, None}, optional
        Axis along which arrays are compared. The default (`axis` = ``None``)
        is to compare flattened arrays. `axis` may be
        negative, in which case it counts from the last to the first axis.

    Returns
    -------
    y : bool or ndarray
        A new boolean or `ndarray` is returned.

    See also
    --------
    bottlechest.nancmp: Compare two arrays, ignoring NaNs

    Examples -- TODO: PROVIDE EXAMPLES!
    --------
    >>> bn.nanequal(1)
    False
    >>> bn.nanequal(np.nan)
    True
    >>> bn.nanequal([1, np.nan])
    True
    >>> a = np.array([[1, 4], [1, np.nan]])
    >>> bn.nanequal(a)
    True
    >>> bn.nanequal(a, axis=0)
    array([False,  True], dtype=bool)    

    """
    func, arr1, arr2 = nanequal_selector(arr1, arr2, axis)
    return func(arr1, arr2)

def nanequal_selector(arr1, arr2, axis):
    """
    Return nanequal function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in bn.nanequal()
    is in checking that `axis` is within range, converting `arr` into an
    array (if it is not already an array), and selecting the function to use.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr1 : array_like
        First input array. If `arr` is not an array, a conversion is attempted.
    arr2 : array_like
        Second input array
    axis : {int, None}, optional
        Axis along which arrays are compared. The default (`axis` = ``None``)
        is to compare flattened arrays. `axis` may be
        negative, in which case it counts from the last to the first axis.
    
    Returns
    -------
    func : function
        The nanequal function that matches the number of dimensions and
        dtype of the input array and the axis.
    a1 : ndarray
        If the input array `arr1` is not a ndarray, then `a1` will contain the
        result of converting `arr1` into a ndarray.
    a2 : ndarray
        Equivalent for arr2.

    Examples   TODO: PROVIDE EXAMPLES
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])
    
    Obtain the function needed to determine if there are any NaN in `arr`:

    >>> func, a = bn.func.nanequal_selector(arr, axis=0)
    >>> func
    <function nanequal_1d_float64_axisNone>
    
    Use the returned function and array to determine if there are any
    NaNs:
    
    >>> func(a)
    False

    """
    cdef np.ndarray a1, a2
    if type(arr1) is np.ndarray:
        a1 = arr1
    else:    
        a1 = np.array(arr1, copy=False)
    if type(arr2) is np.ndarray:
        a2 = arr2
    else:
        a2 = np.array(arr2, copy=False)
    cdef int ndim = PyArray_NDIM(a1)
    cdef int ndim2 = PyArray_NDIM(a2)
    if ndim != ndim2:
        raise ValueError("arrays have different dimensions, %i != %i" %
                         (ndim, ndim2))
    cdef int dtype = PyArray_TYPE(a1)
    cdef np.npy_intp *dim1, *dim2
    cdef int i
    cdef tuple key = (ndim, dtype, axis)
    if dtype == PyArray_TYPE(a2):
        dim1 = PyArray_DIMS(a1)
        dim2 = PyArray_DIMS(a2)
        for i in range(ndim):
            if dim1[i] != dim2[i]:
                raise ValueError("shape mismatch");
        if (axis is not None) and (axis < 0):
            axis += ndim
        try:
            func = nanequal_dict[key]
            return func, a1, a2
        except KeyError:
            pass

    if axis is not None:
        if (axis < 0) or (axis >= ndim):
            raise ValueError("axis(=%d) out of bounds" % axis)
    try:
        func = nanequal_slow_dict[axis]
    except KeyError:
        tup = (str(ndim), str(a1.dtype), str(axis))
        raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a1, a2
'''   
