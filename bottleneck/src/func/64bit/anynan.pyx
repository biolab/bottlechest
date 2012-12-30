# 64 bit version
"anynan auto-generated from template"

def anynan(arr, axis=None):
    """
    Test whether any array element along a given axis is NaN.

    Returns single boolean unless `axis` is not ``None``.

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
    bottleneck.allnan: Test if all array elements along given axis are NaN

    Examples
    --------
    >>> bn.anynan(1)
    False
    >>> bn.anynan(np.nan)
    True
    >>> bn.anynan([1, np.nan])
    True
    >>> a = np.array([[1, 4], [1, np.nan]])
    >>> bn.anynan(a)
    True
    >>> bn.anynan(a, axis=0)
    array([False,  True], dtype=bool)    

    """
    func, arr = anynan_selector(arr, axis)
    return func(arr)

def anynan_selector(arr, axis):
    """
    Return anynan function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in bn.anynan()
    is in checking that `axis` is within range, converting `arr` into an
    array (if it is not already an array), and selecting the function to use.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}
        Axis along which NaNs are searched.
    
    Returns
    -------
    func : function
        The anynan function that matches the number of dimensions and
        dtype of the input array and the axis.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])
    
    Obtain the function needed to determine if there are any NaN in `arr`:

    >>> func, a = bn.func.anynan_selector(arr, axis=0)
    >>> func
    <function anynan_1d_float64_axisNone>
    
    Use the returned function and array to determine if there are any
    NaNs:
    
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
        func = anynan_dict[key]
    except KeyError:
        if axis is not None:
            if (axis < 0) or (axis >= ndim):
                raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = anynan_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_2d_int8_axis0(np.ndarray[np.int8_t, ndim=2] a):
    "Check for NaNs in 2d array with dtype=int8 along axis=0."
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
    for i1 in range(n1):
        y[i1] = 0
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_2d_int8_axis1(np.ndarray[np.int8_t, ndim=2] a):
    "Check for NaNs in 2d array with dtype=int8 along axis=1."
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
    for i0 in range(n0):
        y[i0] = 0
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "Check for NaNs in 2d array with dtype=int32 along axis=0."
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
    for i1 in range(n1):
        y[i1] = 0
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a):
    "Check for NaNs in 2d array with dtype=int32 along axis=1."
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
    for i0 in range(n0):
        y[i0] = 0
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a):
    "Check for NaNs in 2d array with dtype=int64 along axis=0."
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
    for i1 in range(n1):
        y[i1] = 0
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a):
    "Check for NaNs in 2d array with dtype=int64 along axis=1."
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
    for i0 in range(n0):
        y[i0] = 0
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_1d_float32_axisNone(np.ndarray[np.float32_t, ndim=1] a):
    "Check for NaNs in 1d array with dtype=float32 along axis=None."
    cdef int f = 1
    cdef np.float32_t ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    for i0 in range(n0):
        ai = a[i0]
        if ai != ai:
            return np.bool_(True)
    return np.bool_(False)

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_1d_float64_axisNone(np.ndarray[np.float64_t, ndim=1] a):
    "Check for NaNs in 1d array with dtype=float64 along axis=None."
    cdef int f = 1
    cdef np.float64_t ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    for i0 in range(n0):
        ai = a[i0]
        if ai != ai:
            return np.bool_(True)
    return np.bool_(False)

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_2d_float32_axisNone(np.ndarray[np.float32_t, ndim=2] a):
    "Check for NaNs in 2d array with dtype=float32 along axis=None."
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
            if ai != ai:
                return np.bool_(True)
    return np.bool_(False)

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a):
    "Check for NaNs in 2d array with dtype=float64 along axis=None."
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
            if ai != ai:
                return np.bool_(True)
    return np.bool_(False)

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a):
    "Check for NaNs in 2d array with dtype=float32 along axis=0."
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
            if ai != ai:
                y[i1] = 1
                f = 0
                break
        if f == 1:
            y[i1] = 0
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a):
    "Check for NaNs in 2d array with dtype=float32 along axis=1."
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
            if ai != ai:
                y[i0] = 1
                f = 0
                break
        if f == 1:
            y[i0] = 0
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "Check for NaNs in 2d array with dtype=float64 along axis=0."
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
            if ai != ai:
                y[i1] = 1
                f = 0
                break
        if f == 1:
            y[i1] = 0
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "Check for NaNs in 2d array with dtype=float64 along axis=1."
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
            if ai != ai:
                y[i0] = 1
                f = 0
                break
        if f == 1:
            y[i0] = 0
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_1d_int8_axisNone(np.ndarray[np.int8_t, ndim=1] a):
    "Check for NaNs in 1d array with dtype=int8 along axis=None."
    cdef int f = 1
    cdef np.int8_t ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    return np.bool_(False)

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_1d_int32_axisNone(np.ndarray[np.int32_t, ndim=1] a):
    "Check for NaNs in 1d array with dtype=int32 along axis=None."
    cdef int f = 1
    cdef np.int32_t ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    return np.bool_(False)

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_1d_int64_axisNone(np.ndarray[np.int64_t, ndim=1] a):
    "Check for NaNs in 1d array with dtype=int64 along axis=None."
    cdef int f = 1
    cdef np.int64_t ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    return np.bool_(False)

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_2d_int8_axisNone(np.ndarray[np.int8_t, ndim=2] a):
    "Check for NaNs in 2d array with dtype=int8 along axis=None."
    cdef int f = 1
    cdef np.int8_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    return np.bool_(False)

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_2d_int32_axisNone(np.ndarray[np.int32_t, ndim=2] a):
    "Check for NaNs in 2d array with dtype=int32 along axis=None."
    cdef int f = 1
    cdef np.int32_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    return np.bool_(False)

@cython.boundscheck(False)
@cython.wraparound(False)
def anynan_2d_int64_axisNone(np.ndarray[np.int64_t, ndim=2] a):
    "Check for NaNs in 2d array with dtype=int64 along axis=None."
    cdef int f = 1
    cdef np.int64_t ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    return np.bool_(False)

cdef dict anynan_dict = {}
anynan_dict[(2, NPY_int8, 0)] = anynan_2d_int8_axis0
anynan_dict[(2, NPY_int8, 1)] = anynan_2d_int8_axis1
anynan_dict[(2, NPY_int32, 0)] = anynan_2d_int32_axis0
anynan_dict[(2, NPY_int32, 1)] = anynan_2d_int32_axis1
anynan_dict[(2, NPY_int64, 0)] = anynan_2d_int64_axis0
anynan_dict[(2, NPY_int64, 1)] = anynan_2d_int64_axis1
anynan_dict[(1, NPY_float32, 0)] = anynan_1d_float32_axisNone
anynan_dict[(1, NPY_float32, None)] = anynan_1d_float32_axisNone
anynan_dict[(1, NPY_float64, 0)] = anynan_1d_float64_axisNone
anynan_dict[(1, NPY_float64, None)] = anynan_1d_float64_axisNone
anynan_dict[(2, NPY_float32, None)] = anynan_2d_float32_axisNone
anynan_dict[(2, NPY_float64, None)] = anynan_2d_float64_axisNone
anynan_dict[(2, NPY_float32, 0)] = anynan_2d_float32_axis0
anynan_dict[(2, NPY_float32, 1)] = anynan_2d_float32_axis1
anynan_dict[(2, NPY_float64, 0)] = anynan_2d_float64_axis0
anynan_dict[(2, NPY_float64, 1)] = anynan_2d_float64_axis1
anynan_dict[(1, NPY_int8, 0)] = anynan_1d_int8_axisNone
anynan_dict[(1, NPY_int8, None)] = anynan_1d_int8_axisNone
anynan_dict[(1, NPY_int32, 0)] = anynan_1d_int32_axisNone
anynan_dict[(1, NPY_int32, None)] = anynan_1d_int32_axisNone
anynan_dict[(1, NPY_int64, 0)] = anynan_1d_int64_axisNone
anynan_dict[(1, NPY_int64, None)] = anynan_1d_int64_axisNone
anynan_dict[(2, NPY_int8, None)] = anynan_2d_int8_axisNone
anynan_dict[(2, NPY_int32, None)] = anynan_2d_int32_axisNone
anynan_dict[(2, NPY_int64, None)] = anynan_2d_int64_axisNone

def anynan_slow_axis0(arr):
    "Unaccelerated (slow) anynan along axis 0."
    return bn.slow.anynan(arr, axis=0)

def anynan_slow_axis1(arr):
    "Unaccelerated (slow) anynan along axis 1."
    return bn.slow.anynan(arr, axis=1)

def anynan_slow_axis2(arr):
    "Unaccelerated (slow) anynan along axis 2."
    return bn.slow.anynan(arr, axis=2)

def anynan_slow_axis3(arr):
    "Unaccelerated (slow) anynan along axis 3."
    return bn.slow.anynan(arr, axis=3)

def anynan_slow_axis4(arr):
    "Unaccelerated (slow) anynan along axis 4."
    return bn.slow.anynan(arr, axis=4)

def anynan_slow_axis5(arr):
    "Unaccelerated (slow) anynan along axis 5."
    return bn.slow.anynan(arr, axis=5)

def anynan_slow_axis6(arr):
    "Unaccelerated (slow) anynan along axis 6."
    return bn.slow.anynan(arr, axis=6)

def anynan_slow_axis7(arr):
    "Unaccelerated (slow) anynan along axis 7."
    return bn.slow.anynan(arr, axis=7)

def anynan_slow_axis8(arr):
    "Unaccelerated (slow) anynan along axis 8."
    return bn.slow.anynan(arr, axis=8)

def anynan_slow_axis9(arr):
    "Unaccelerated (slow) anynan along axis 9."
    return bn.slow.anynan(arr, axis=9)

def anynan_slow_axis10(arr):
    "Unaccelerated (slow) anynan along axis 10."
    return bn.slow.anynan(arr, axis=10)

def anynan_slow_axis11(arr):
    "Unaccelerated (slow) anynan along axis 11."
    return bn.slow.anynan(arr, axis=11)

def anynan_slow_axis12(arr):
    "Unaccelerated (slow) anynan along axis 12."
    return bn.slow.anynan(arr, axis=12)

def anynan_slow_axis13(arr):
    "Unaccelerated (slow) anynan along axis 13."
    return bn.slow.anynan(arr, axis=13)

def anynan_slow_axis14(arr):
    "Unaccelerated (slow) anynan along axis 14."
    return bn.slow.anynan(arr, axis=14)

def anynan_slow_axis15(arr):
    "Unaccelerated (slow) anynan along axis 15."
    return bn.slow.anynan(arr, axis=15)

def anynan_slow_axis16(arr):
    "Unaccelerated (slow) anynan along axis 16."
    return bn.slow.anynan(arr, axis=16)

def anynan_slow_axis17(arr):
    "Unaccelerated (slow) anynan along axis 17."
    return bn.slow.anynan(arr, axis=17)

def anynan_slow_axis18(arr):
    "Unaccelerated (slow) anynan along axis 18."
    return bn.slow.anynan(arr, axis=18)

def anynan_slow_axis19(arr):
    "Unaccelerated (slow) anynan along axis 19."
    return bn.slow.anynan(arr, axis=19)

def anynan_slow_axis20(arr):
    "Unaccelerated (slow) anynan along axis 20."
    return bn.slow.anynan(arr, axis=20)

def anynan_slow_axis21(arr):
    "Unaccelerated (slow) anynan along axis 21."
    return bn.slow.anynan(arr, axis=21)

def anynan_slow_axis22(arr):
    "Unaccelerated (slow) anynan along axis 22."
    return bn.slow.anynan(arr, axis=22)

def anynan_slow_axis23(arr):
    "Unaccelerated (slow) anynan along axis 23."
    return bn.slow.anynan(arr, axis=23)

def anynan_slow_axis24(arr):
    "Unaccelerated (slow) anynan along axis 24."
    return bn.slow.anynan(arr, axis=24)

def anynan_slow_axis25(arr):
    "Unaccelerated (slow) anynan along axis 25."
    return bn.slow.anynan(arr, axis=25)

def anynan_slow_axis26(arr):
    "Unaccelerated (slow) anynan along axis 26."
    return bn.slow.anynan(arr, axis=26)

def anynan_slow_axis27(arr):
    "Unaccelerated (slow) anynan along axis 27."
    return bn.slow.anynan(arr, axis=27)

def anynan_slow_axis28(arr):
    "Unaccelerated (slow) anynan along axis 28."
    return bn.slow.anynan(arr, axis=28)

def anynan_slow_axis29(arr):
    "Unaccelerated (slow) anynan along axis 29."
    return bn.slow.anynan(arr, axis=29)

def anynan_slow_axis30(arr):
    "Unaccelerated (slow) anynan along axis 30."
    return bn.slow.anynan(arr, axis=30)

def anynan_slow_axis31(arr):
    "Unaccelerated (slow) anynan along axis 31."
    return bn.slow.anynan(arr, axis=31)

def anynan_slow_axis32(arr):
    "Unaccelerated (slow) anynan along axis 32."
    return bn.slow.anynan(arr, axis=32)

def anynan_slow_axisNone(arr):
    "Unaccelerated (slow) anynan along axis None."
    return bn.slow.anynan(arr, axis=None)


cdef dict anynan_slow_dict = {}
anynan_slow_dict[0] = anynan_slow_axis0
anynan_slow_dict[1] = anynan_slow_axis1
anynan_slow_dict[2] = anynan_slow_axis2
anynan_slow_dict[3] = anynan_slow_axis3
anynan_slow_dict[4] = anynan_slow_axis4
anynan_slow_dict[5] = anynan_slow_axis5
anynan_slow_dict[6] = anynan_slow_axis6
anynan_slow_dict[7] = anynan_slow_axis7
anynan_slow_dict[8] = anynan_slow_axis8
anynan_slow_dict[9] = anynan_slow_axis9
anynan_slow_dict[10] = anynan_slow_axis10
anynan_slow_dict[11] = anynan_slow_axis11
anynan_slow_dict[12] = anynan_slow_axis12
anynan_slow_dict[13] = anynan_slow_axis13
anynan_slow_dict[14] = anynan_slow_axis14
anynan_slow_dict[15] = anynan_slow_axis15
anynan_slow_dict[16] = anynan_slow_axis16
anynan_slow_dict[17] = anynan_slow_axis17
anynan_slow_dict[18] = anynan_slow_axis18
anynan_slow_dict[19] = anynan_slow_axis19
anynan_slow_dict[20] = anynan_slow_axis20
anynan_slow_dict[21] = anynan_slow_axis21
anynan_slow_dict[22] = anynan_slow_axis22
anynan_slow_dict[23] = anynan_slow_axis23
anynan_slow_dict[24] = anynan_slow_axis24
anynan_slow_dict[25] = anynan_slow_axis25
anynan_slow_dict[26] = anynan_slow_axis26
anynan_slow_dict[27] = anynan_slow_axis27
anynan_slow_dict[28] = anynan_slow_axis28
anynan_slow_dict[29] = anynan_slow_axis29
anynan_slow_dict[30] = anynan_slow_axis30
anynan_slow_dict[31] = anynan_slow_axis31
anynan_slow_dict[32] = anynan_slow_axis32
anynan_slow_dict[None] = anynan_slow_axisNone