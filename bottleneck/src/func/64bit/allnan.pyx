# 64 bit version
"allnan auto-generated from template"

def allnan(arr, axis=None):
    """
    Test whether all array elements along a given axis are NaN.

    Returns single boolean unless `axis` is not ``None``.

    Note that allnan([]) is True to match np.isnan([]).all().

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which NaNs are searched.  The default (`axis` = ``None``)
        is to search for NaNs over a flattened input array. `axis` may be
        negative, in which case it counts from the last to the first axis.

    Returns
    -------
    y : bool or ndarray
        A new boolean or `ndarray` is returned.

    See also
    --------
    bottleneck.anynan: Test if any array element along given axis is NaN

    Examples
    --------
    >>> bn.allnan(1)
    False
    >>> bn.allnan(np.nan)
    True
    >>> bn.allnan([1, np.nan])
    False
    >>> a = np.array([[1, np.nan], [1, np.nan]])
    >>> bn.allnan(a)
    False
    >>> bn.allnan(a, axis=0)
    array([False,  True], dtype=bool)

    An empty array returns True:

    >>> bn.allnan([])
    True

    which is similar to:

    >>> all([])
    True
    >>> np.isnan([]).all()
    True

    """
    func, arr = allnan_selector(arr, axis)
    return func(arr)

def allnan_selector(arr, axis):
    """
    Return allnan function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in bn.allnan()
    is in checking that `axis` is within range, converting `arr` into an
    array (if it is not already an array), and selecting the function to use.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}
        Axis along which NaNs are searched.
    
    Returns
    -------
    func : function
        The allnan function that matches the number of dimensions and
        dtype of the input array and the axis.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])
    
    Obtain the function needed to determine if `arr` contains all NaNs:

    >>> func, a = bn.func.allnan_selector(arr, axis=0)
    >>> func
    <function allnan_1d_float64_axisNone>
    
    Use the returned function and array to determine is all elements are
    NaN:
    
    >>> func(a)
    False

    """
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:    
        a = np.array(arr, copy=False)
    cdef int ndim = PyArray_NDIM(a)
    cdef int dtype = PyArray_TYPE(a)
    if (axis is not None) and (axis < 0):
        axis += ndim
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = allnan_dict[key]
    except KeyError:
        if axis is not None:
            if (axis < 0) or (axis >= ndim):
                raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = allnan_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_2d_int8_axis0(np.ndarray[np.int8_t, ndim=2] a):
    "Check for all NaNs in 2d array with dtype=int8 along axis=0."
    cdef int f = 1
    cdef np.int8_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] y = PyArray_EMPTY(1, dims,
		NPY_BOOL, 0)
    if n0 == 0:
        f = 1
    else:
        f = 0
    for i1 in range(n1):
        y[i1] = f
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_2d_int8_axis1(np.ndarray[np.int8_t, ndim=2] a):
    "Check for all NaNs in 2d array with dtype=int8 along axis=1."
    cdef int f = 1
    cdef np.int8_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] y = PyArray_EMPTY(1, dims,
		NPY_BOOL, 0)
    if n1 == 0:
        f = 1
    else:
        f = 0
    for i0 in range(n0):
        y[i0] = f
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "Check for all NaNs in 2d array with dtype=int32 along axis=0."
    cdef int f = 1
    cdef np.int32_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] y = PyArray_EMPTY(1, dims,
		NPY_BOOL, 0)
    if n0 == 0:
        f = 1
    else:
        f = 0
    for i1 in range(n1):
        y[i1] = f
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a):
    "Check for all NaNs in 2d array with dtype=int32 along axis=1."
    cdef int f = 1
    cdef np.int32_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] y = PyArray_EMPTY(1, dims,
		NPY_BOOL, 0)
    if n1 == 0:
        f = 1
    else:
        f = 0
    for i0 in range(n0):
        y[i0] = f
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a):
    "Check for all NaNs in 2d array with dtype=int64 along axis=0."
    cdef int f = 1
    cdef np.int64_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] y = PyArray_EMPTY(1, dims,
		NPY_BOOL, 0)
    if n0 == 0:
        f = 1
    else:
        f = 0
    for i1 in range(n1):
        y[i1] = f
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a):
    "Check for all NaNs in 2d array with dtype=int64 along axis=1."
    cdef int f = 1
    cdef np.int64_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] y = PyArray_EMPTY(1, dims,
		NPY_BOOL, 0)
    if n1 == 0:
        f = 1
    else:
        f = 0
    for i0 in range(n0):
        y[i0] = f
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_1d_float32_axisNone(np.ndarray[np.float32_t, ndim=1] a):
    "Check for all NaNs in 1d array with dtype=float32 along axis=None."
    cdef int f = 1
    cdef np.float32_t ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    for i0 in range(n0):
        ai = a[i0]
        if ai == ai:
            return np.bool_(False)
    return np.bool_(True)

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_1d_float64_axisNone(np.ndarray[np.float64_t, ndim=1] a):
    "Check for all NaNs in 1d array with dtype=float64 along axis=None."
    cdef int f = 1
    cdef np.float64_t ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    for i0 in range(n0):
        ai = a[i0]
        if ai == ai:
            return np.bool_(False)
    return np.bool_(True)

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_2d_float32_axisNone(np.ndarray[np.float32_t, ndim=2] a):
    "Check for all NaNs in 2d array with dtype=float32 along axis=None."
    cdef int f = 1
    cdef np.float32_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai == ai:
                return np.bool_(False)
    return np.bool_(True)

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a):
    "Check for all NaNs in 2d array with dtype=float64 along axis=None."
    cdef int f = 1
    cdef np.float64_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai == ai:
                return np.bool_(False)
    return np.bool_(True)

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a):
    "Check for all NaNs in 2d array with dtype=float32 along axis=0."
    cdef int f = 1
    cdef np.float32_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] y = PyArray_EMPTY(1, dims,
		NPY_BOOL, 0)
    for i1 in range(n1):
        f = 1
        for i0 in range(n0):
            ai = a[i0, i1]
            if ai == ai:
                y[i1] = 0
                f = 0
                break
        if f == 1:
            y[i1] = 1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a):
    "Check for all NaNs in 2d array with dtype=float32 along axis=1."
    cdef int f = 1
    cdef np.float32_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] y = PyArray_EMPTY(1, dims,
		NPY_BOOL, 0)
    for i0 in range(n0):
        f = 1
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai == ai:
                y[i0] = 0
                f = 0
                break
        if f == 1:
            y[i0] = 1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "Check for all NaNs in 2d array with dtype=float64 along axis=0."
    cdef int f = 1
    cdef np.float64_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] y = PyArray_EMPTY(1, dims,
		NPY_BOOL, 0)
    for i1 in range(n1):
        f = 1
        for i0 in range(n0):
            ai = a[i0, i1]
            if ai == ai:
                y[i1] = 0
                f = 0
                break
        if f == 1:
            y[i1] = 1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "Check for all NaNs in 2d array with dtype=float64 along axis=1."
    cdef int f = 1
    cdef np.float64_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] y = PyArray_EMPTY(1, dims,
		NPY_BOOL, 0)
    for i0 in range(n0):
        f = 1
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai == ai:
                y[i0] = 0
                f = 0
                break
        if f == 1:
            y[i0] = 1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_1d_int8_axisNone(np.ndarray[np.int8_t, ndim=1] a):
    "Check for all NaNs in 1d array with dtype=int8 along axis=None."
    cdef int f = 1
    cdef np.int8_t ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    if n0 == 0:
        return np.bool_(True)
    else:
        return np.bool_(False)

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_1d_int32_axisNone(np.ndarray[np.int32_t, ndim=1] a):
    "Check for all NaNs in 1d array with dtype=int32 along axis=None."
    cdef int f = 1
    cdef np.int32_t ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    if n0 == 0:
        return np.bool_(True)
    else:
        return np.bool_(False)

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_1d_int64_axisNone(np.ndarray[np.int64_t, ndim=1] a):
    "Check for all NaNs in 1d array with dtype=int64 along axis=None."
    cdef int f = 1
    cdef np.int64_t ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    if n0 == 0:
        return np.bool_(True)
    else:
        return np.bool_(False)

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_2d_int8_axisNone(np.ndarray[np.int8_t, ndim=2] a):
    "Check for all NaNs in 2d array with dtype=int8 along axis=None."
    cdef int f = 1
    cdef np.int8_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    if n0* n1 == 0:
        return np.bool_(True)
    else:
        return np.bool_(False)

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_2d_int32_axisNone(np.ndarray[np.int32_t, ndim=2] a):
    "Check for all NaNs in 2d array with dtype=int32 along axis=None."
    cdef int f = 1
    cdef np.int32_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    if n0* n1 == 0:
        return np.bool_(True)
    else:
        return np.bool_(False)

@cython.boundscheck(False)
@cython.wraparound(False)
def allnan_2d_int64_axisNone(np.ndarray[np.int64_t, ndim=2] a):
    "Check for all NaNs in 2d array with dtype=int64 along axis=None."
    cdef int f = 1
    cdef np.int64_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    if n0* n1 == 0:
        return np.bool_(True)
    else:
        return np.bool_(False)

cdef dict allnan_dict = {}
allnan_dict[(2, NPY_int8, 0)] = allnan_2d_int8_axis0
allnan_dict[(2, NPY_int8, 1)] = allnan_2d_int8_axis1
allnan_dict[(2, NPY_int32, 0)] = allnan_2d_int32_axis0
allnan_dict[(2, NPY_int32, 1)] = allnan_2d_int32_axis1
allnan_dict[(2, NPY_int64, 0)] = allnan_2d_int64_axis0
allnan_dict[(2, NPY_int64, 1)] = allnan_2d_int64_axis1
allnan_dict[(1, NPY_float32, 0)] = allnan_1d_float32_axisNone
allnan_dict[(1, NPY_float32, None)] = allnan_1d_float32_axisNone
allnan_dict[(1, NPY_float64, 0)] = allnan_1d_float64_axisNone
allnan_dict[(1, NPY_float64, None)] = allnan_1d_float64_axisNone
allnan_dict[(2, NPY_float32, None)] = allnan_2d_float32_axisNone
allnan_dict[(2, NPY_float64, None)] = allnan_2d_float64_axisNone
allnan_dict[(2, NPY_float32, 0)] = allnan_2d_float32_axis0
allnan_dict[(2, NPY_float32, 1)] = allnan_2d_float32_axis1
allnan_dict[(2, NPY_float64, 0)] = allnan_2d_float64_axis0
allnan_dict[(2, NPY_float64, 1)] = allnan_2d_float64_axis1
allnan_dict[(1, NPY_int8, 0)] = allnan_1d_int8_axisNone
allnan_dict[(1, NPY_int8, None)] = allnan_1d_int8_axisNone
allnan_dict[(1, NPY_int32, 0)] = allnan_1d_int32_axisNone
allnan_dict[(1, NPY_int32, None)] = allnan_1d_int32_axisNone
allnan_dict[(1, NPY_int64, 0)] = allnan_1d_int64_axisNone
allnan_dict[(1, NPY_int64, None)] = allnan_1d_int64_axisNone
allnan_dict[(2, NPY_int8, None)] = allnan_2d_int8_axisNone
allnan_dict[(2, NPY_int32, None)] = allnan_2d_int32_axisNone
allnan_dict[(2, NPY_int64, None)] = allnan_2d_int64_axisNone

def allnan_slow_axis0(arr):
    "Unaccelerated (slow) allnan along axis 0."
    return bn.slow.allnan(arr, axis=0)

def allnan_slow_axis1(arr):
    "Unaccelerated (slow) allnan along axis 1."
    return bn.slow.allnan(arr, axis=1)

def allnan_slow_axis2(arr):
    "Unaccelerated (slow) allnan along axis 2."
    return bn.slow.allnan(arr, axis=2)

def allnan_slow_axis3(arr):
    "Unaccelerated (slow) allnan along axis 3."
    return bn.slow.allnan(arr, axis=3)

def allnan_slow_axis4(arr):
    "Unaccelerated (slow) allnan along axis 4."
    return bn.slow.allnan(arr, axis=4)

def allnan_slow_axis5(arr):
    "Unaccelerated (slow) allnan along axis 5."
    return bn.slow.allnan(arr, axis=5)

def allnan_slow_axis6(arr):
    "Unaccelerated (slow) allnan along axis 6."
    return bn.slow.allnan(arr, axis=6)

def allnan_slow_axis7(arr):
    "Unaccelerated (slow) allnan along axis 7."
    return bn.slow.allnan(arr, axis=7)

def allnan_slow_axis8(arr):
    "Unaccelerated (slow) allnan along axis 8."
    return bn.slow.allnan(arr, axis=8)

def allnan_slow_axis9(arr):
    "Unaccelerated (slow) allnan along axis 9."
    return bn.slow.allnan(arr, axis=9)

def allnan_slow_axis10(arr):
    "Unaccelerated (slow) allnan along axis 10."
    return bn.slow.allnan(arr, axis=10)

def allnan_slow_axis11(arr):
    "Unaccelerated (slow) allnan along axis 11."
    return bn.slow.allnan(arr, axis=11)

def allnan_slow_axis12(arr):
    "Unaccelerated (slow) allnan along axis 12."
    return bn.slow.allnan(arr, axis=12)

def allnan_slow_axis13(arr):
    "Unaccelerated (slow) allnan along axis 13."
    return bn.slow.allnan(arr, axis=13)

def allnan_slow_axis14(arr):
    "Unaccelerated (slow) allnan along axis 14."
    return bn.slow.allnan(arr, axis=14)

def allnan_slow_axis15(arr):
    "Unaccelerated (slow) allnan along axis 15."
    return bn.slow.allnan(arr, axis=15)

def allnan_slow_axis16(arr):
    "Unaccelerated (slow) allnan along axis 16."
    return bn.slow.allnan(arr, axis=16)

def allnan_slow_axis17(arr):
    "Unaccelerated (slow) allnan along axis 17."
    return bn.slow.allnan(arr, axis=17)

def allnan_slow_axis18(arr):
    "Unaccelerated (slow) allnan along axis 18."
    return bn.slow.allnan(arr, axis=18)

def allnan_slow_axis19(arr):
    "Unaccelerated (slow) allnan along axis 19."
    return bn.slow.allnan(arr, axis=19)

def allnan_slow_axis20(arr):
    "Unaccelerated (slow) allnan along axis 20."
    return bn.slow.allnan(arr, axis=20)

def allnan_slow_axis21(arr):
    "Unaccelerated (slow) allnan along axis 21."
    return bn.slow.allnan(arr, axis=21)

def allnan_slow_axis22(arr):
    "Unaccelerated (slow) allnan along axis 22."
    return bn.slow.allnan(arr, axis=22)

def allnan_slow_axis23(arr):
    "Unaccelerated (slow) allnan along axis 23."
    return bn.slow.allnan(arr, axis=23)

def allnan_slow_axis24(arr):
    "Unaccelerated (slow) allnan along axis 24."
    return bn.slow.allnan(arr, axis=24)

def allnan_slow_axis25(arr):
    "Unaccelerated (slow) allnan along axis 25."
    return bn.slow.allnan(arr, axis=25)

def allnan_slow_axis26(arr):
    "Unaccelerated (slow) allnan along axis 26."
    return bn.slow.allnan(arr, axis=26)

def allnan_slow_axis27(arr):
    "Unaccelerated (slow) allnan along axis 27."
    return bn.slow.allnan(arr, axis=27)

def allnan_slow_axis28(arr):
    "Unaccelerated (slow) allnan along axis 28."
    return bn.slow.allnan(arr, axis=28)

def allnan_slow_axis29(arr):
    "Unaccelerated (slow) allnan along axis 29."
    return bn.slow.allnan(arr, axis=29)

def allnan_slow_axis30(arr):
    "Unaccelerated (slow) allnan along axis 30."
    return bn.slow.allnan(arr, axis=30)

def allnan_slow_axis31(arr):
    "Unaccelerated (slow) allnan along axis 31."
    return bn.slow.allnan(arr, axis=31)

def allnan_slow_axis32(arr):
    "Unaccelerated (slow) allnan along axis 32."
    return bn.slow.allnan(arr, axis=32)

def allnan_slow_axisNone(arr):
    "Unaccelerated (slow) allnan along axis None."
    return bn.slow.allnan(arr, axis=None)


cdef dict allnan_slow_dict = {}
allnan_slow_dict[0] = allnan_slow_axis0
allnan_slow_dict[1] = allnan_slow_axis1
allnan_slow_dict[2] = allnan_slow_axis2
allnan_slow_dict[3] = allnan_slow_axis3
allnan_slow_dict[4] = allnan_slow_axis4
allnan_slow_dict[5] = allnan_slow_axis5
allnan_slow_dict[6] = allnan_slow_axis6
allnan_slow_dict[7] = allnan_slow_axis7
allnan_slow_dict[8] = allnan_slow_axis8
allnan_slow_dict[9] = allnan_slow_axis9
allnan_slow_dict[10] = allnan_slow_axis10
allnan_slow_dict[11] = allnan_slow_axis11
allnan_slow_dict[12] = allnan_slow_axis12
allnan_slow_dict[13] = allnan_slow_axis13
allnan_slow_dict[14] = allnan_slow_axis14
allnan_slow_dict[15] = allnan_slow_axis15
allnan_slow_dict[16] = allnan_slow_axis16
allnan_slow_dict[17] = allnan_slow_axis17
allnan_slow_dict[18] = allnan_slow_axis18
allnan_slow_dict[19] = allnan_slow_axis19
allnan_slow_dict[20] = allnan_slow_axis20
allnan_slow_dict[21] = allnan_slow_axis21
allnan_slow_dict[22] = allnan_slow_axis22
allnan_slow_dict[23] = allnan_slow_axis23
allnan_slow_dict[24] = allnan_slow_axis24
allnan_slow_dict[25] = allnan_slow_axis25
allnan_slow_dict[26] = allnan_slow_axis26
allnan_slow_dict[27] = allnan_slow_axis27
allnan_slow_dict[28] = allnan_slow_axis28
allnan_slow_dict[29] = allnan_slow_axis29
allnan_slow_dict[30] = allnan_slow_axis30
allnan_slow_dict[31] = allnan_slow_axis31
allnan_slow_dict[32] = allnan_slow_axis32
allnan_slow_dict[None] = allnan_slow_axisNone