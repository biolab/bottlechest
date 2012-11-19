# 64 bit version
"nanequal auto-generated from template"

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
    bottleneck.nancmp: Compare two arrays, ignoring NaNs

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

@cython.boundscheck(False)
@cython.wraparound(False)
def nanequal_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a,
                              np.ndarray[np.int32_t, ndim=2] b):
    "Check whether two arrays are equal, ignoring NaNs, in 2d array with dtype=int32 along axis=0."
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
        f = 1
        for i0 in range(n0):
            ai = a[i0, i1]
            bi = b[i0, i1]
            if ai != bi:
                y[i1] = 0
                f = 0
                break
        if f == 1:
            y[i1] = 1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanequal_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a,
                              np.ndarray[np.int32_t, ndim=2] b):
    "Check whether two arrays are equal, ignoring NaNs, in 2d array with dtype=int32 along axis=1."
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
        f = 1
        for i1 in range(n1):
            ai = a[i0, i1]
            bi = b[i0, i1]
            if ai != bi:
                y[i0] = 0
                f = 0
                break
        if f == 1:
            y[i0] = 1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanequal_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a,
                              np.ndarray[np.int64_t, ndim=2] b):
    "Check whether two arrays are equal, ignoring NaNs, in 2d array with dtype=int64 along axis=0."
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
        f = 1
        for i0 in range(n0):
            ai = a[i0, i1]
            bi = b[i0, i1]
            if ai != bi:
                y[i1] = 0
                f = 0
                break
        if f == 1:
            y[i1] = 1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanequal_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a,
                              np.ndarray[np.int64_t, ndim=2] b):
    "Check whether two arrays are equal, ignoring NaNs, in 2d array with dtype=int64 along axis=1."
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
        f = 1
        for i1 in range(n1):
            ai = a[i0, i1]
            bi = b[i0, i1]
            if ai != bi:
                y[i0] = 0
                f = 0
                break
        if f == 1:
            y[i0] = 1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanequal_1d_float32_axisNone(np.ndarray[np.float32_t, ndim=1] a,
                              np.ndarray[np.float32_t, ndim=1] b):
    "Check whether two arrays are equal, ignoring NaNs, in 1d array with dtype=float32 along axis=None."
    cdef int f = 1
    cdef np.float32_t ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    for i0 in range(n0):
        ai = a[i0]
        bi = b[i0]
        if ai != bi and ai == ai and bi == bi:
            return np.bool_(False)
    return np.bool_(True)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanequal_1d_float64_axisNone(np.ndarray[np.float64_t, ndim=1] a,
                              np.ndarray[np.float64_t, ndim=1] b):
    "Check whether two arrays are equal, ignoring NaNs, in 1d array with dtype=float64 along axis=None."
    cdef int f = 1
    cdef np.float64_t ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    for i0 in range(n0):
        ai = a[i0]
        bi = b[i0]
        if ai != bi and ai == ai and bi == bi:
            return np.bool_(False)
    return np.bool_(True)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanequal_2d_float32_axisNone(np.ndarray[np.float32_t, ndim=2] a,
                              np.ndarray[np.float32_t, ndim=2] b):
    "Check whether two arrays are equal, ignoring NaNs, in 2d array with dtype=float32 along axis=None."
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
            bi = b[i0, i1]
            if ai != bi and ai == ai and bi == bi:
                return np.bool_(False)
    return np.bool_(True)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanequal_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a,
                              np.ndarray[np.float64_t, ndim=2] b):
    "Check whether two arrays are equal, ignoring NaNs, in 2d array with dtype=float64 along axis=None."
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
            bi = b[i0, i1]
            if ai != bi and ai == ai and bi == bi:
                return np.bool_(False)
    return np.bool_(True)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanequal_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a,
                              np.ndarray[np.float32_t, ndim=2] b):
    "Check whether two arrays are equal, ignoring NaNs, in 2d array with dtype=float32 along axis=0."
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
            bi = b[i0, i1]
            if ai != bi and ai == ai and bi == bi:
                y[i1] = 0
                f = 0
                break
        if f == 1:
            y[i1] = 1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanequal_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a,
                              np.ndarray[np.float32_t, ndim=2] b):
    "Check whether two arrays are equal, ignoring NaNs, in 2d array with dtype=float32 along axis=1."
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
            bi = b[i0, i1]
            if ai != bi and ai == ai and bi == bi:
                y[i0] = 0
                f = 0
                break
        if f == 1:
            y[i0] = 1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanequal_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a,
                              np.ndarray[np.float64_t, ndim=2] b):
    "Check whether two arrays are equal, ignoring NaNs, in 2d array with dtype=float64 along axis=0."
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
            bi = b[i0, i1]
            if ai != bi and ai == ai and bi == bi:
                y[i1] = 0
                f = 0
                break
        if f == 1:
            y[i1] = 1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanequal_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a,
                              np.ndarray[np.float64_t, ndim=2] b):
    "Check whether two arrays are equal, ignoring NaNs, in 2d array with dtype=float64 along axis=1."
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
            bi = b[i0, i1]
            if ai != bi and ai == ai and bi == bi:
                y[i0] = 0
                f = 0
                break
        if f == 1:
            y[i0] = 1
    return y

cdef dict nanequal_dict = {}
nanequal_dict[(2, NPY_int32, 0)] = nanequal_2d_int32_axis0
nanequal_dict[(2, NPY_int32, 1)] = nanequal_2d_int32_axis1
nanequal_dict[(2, NPY_int64, 0)] = nanequal_2d_int64_axis0
nanequal_dict[(2, NPY_int64, 1)] = nanequal_2d_int64_axis1
nanequal_dict[(1, NPY_float32, 0)] = nanequal_1d_float32_axisNone
nanequal_dict[(1, NPY_float32, None)] = nanequal_1d_float32_axisNone
nanequal_dict[(1, NPY_float64, 0)] = nanequal_1d_float64_axisNone
nanequal_dict[(1, NPY_float64, None)] = nanequal_1d_float64_axisNone
nanequal_dict[(2, NPY_float32, None)] = nanequal_2d_float32_axisNone
nanequal_dict[(2, NPY_float64, None)] = nanequal_2d_float64_axisNone
nanequal_dict[(2, NPY_float32, 0)] = nanequal_2d_float32_axis0
nanequal_dict[(2, NPY_float32, 1)] = nanequal_2d_float32_axis1
nanequal_dict[(2, NPY_float64, 0)] = nanequal_2d_float64_axis0
nanequal_dict[(2, NPY_float64, 1)] = nanequal_2d_float64_axis1

def nanequal_slow_axis0(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 0."
    return bn.slow.nanequal(arr1, arr2, axis=0)

def nanequal_slow_axis1(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 1."
    return bn.slow.nanequal(arr1, arr2, axis=1)

def nanequal_slow_axis2(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 2."
    return bn.slow.nanequal(arr1, arr2, axis=2)

def nanequal_slow_axis3(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 3."
    return bn.slow.nanequal(arr1, arr2, axis=3)

def nanequal_slow_axis4(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 4."
    return bn.slow.nanequal(arr1, arr2, axis=4)

def nanequal_slow_axis5(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 5."
    return bn.slow.nanequal(arr1, arr2, axis=5)

def nanequal_slow_axis6(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 6."
    return bn.slow.nanequal(arr1, arr2, axis=6)

def nanequal_slow_axis7(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 7."
    return bn.slow.nanequal(arr1, arr2, axis=7)

def nanequal_slow_axis8(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 8."
    return bn.slow.nanequal(arr1, arr2, axis=8)

def nanequal_slow_axis9(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 9."
    return bn.slow.nanequal(arr1, arr2, axis=9)

def nanequal_slow_axis10(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 10."
    return bn.slow.nanequal(arr1, arr2, axis=10)

def nanequal_slow_axis11(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 11."
    return bn.slow.nanequal(arr1, arr2, axis=11)

def nanequal_slow_axis12(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 12."
    return bn.slow.nanequal(arr1, arr2, axis=12)

def nanequal_slow_axis13(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 13."
    return bn.slow.nanequal(arr1, arr2, axis=13)

def nanequal_slow_axis14(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 14."
    return bn.slow.nanequal(arr1, arr2, axis=14)

def nanequal_slow_axis15(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 15."
    return bn.slow.nanequal(arr1, arr2, axis=15)

def nanequal_slow_axis16(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 16."
    return bn.slow.nanequal(arr1, arr2, axis=16)

def nanequal_slow_axis17(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 17."
    return bn.slow.nanequal(arr1, arr2, axis=17)

def nanequal_slow_axis18(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 18."
    return bn.slow.nanequal(arr1, arr2, axis=18)

def nanequal_slow_axis19(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 19."
    return bn.slow.nanequal(arr1, arr2, axis=19)

def nanequal_slow_axis20(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 20."
    return bn.slow.nanequal(arr1, arr2, axis=20)

def nanequal_slow_axis21(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 21."
    return bn.slow.nanequal(arr1, arr2, axis=21)

def nanequal_slow_axis22(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 22."
    return bn.slow.nanequal(arr1, arr2, axis=22)

def nanequal_slow_axis23(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 23."
    return bn.slow.nanequal(arr1, arr2, axis=23)

def nanequal_slow_axis24(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 24."
    return bn.slow.nanequal(arr1, arr2, axis=24)

def nanequal_slow_axis25(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 25."
    return bn.slow.nanequal(arr1, arr2, axis=25)

def nanequal_slow_axis26(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 26."
    return bn.slow.nanequal(arr1, arr2, axis=26)

def nanequal_slow_axis27(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 27."
    return bn.slow.nanequal(arr1, arr2, axis=27)

def nanequal_slow_axis28(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 28."
    return bn.slow.nanequal(arr1, arr2, axis=28)

def nanequal_slow_axis29(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 29."
    return bn.slow.nanequal(arr1, arr2, axis=29)

def nanequal_slow_axis30(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 30."
    return bn.slow.nanequal(arr1, arr2, axis=30)

def nanequal_slow_axis31(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 31."
    return bn.slow.nanequal(arr1, arr2, axis=31)

def nanequal_slow_axis32(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis 32."
    return bn.slow.nanequal(arr1, arr2, axis=32)

def nanequal_slow_axisNone(arr1, arr2):
    "Unaccelerated (slow) nanequal along axis None."
    return bn.slow.nanequal(arr1, arr2, axis=None)


cdef dict nanequal_slow_dict = {}
nanequal_slow_dict[0] = nanequal_slow_axis0
nanequal_slow_dict[1] = nanequal_slow_axis1
nanequal_slow_dict[2] = nanequal_slow_axis2
nanequal_slow_dict[3] = nanequal_slow_axis3
nanequal_slow_dict[4] = nanequal_slow_axis4
nanequal_slow_dict[5] = nanequal_slow_axis5
nanequal_slow_dict[6] = nanequal_slow_axis6
nanequal_slow_dict[7] = nanequal_slow_axis7
nanequal_slow_dict[8] = nanequal_slow_axis8
nanequal_slow_dict[9] = nanequal_slow_axis9
nanequal_slow_dict[10] = nanequal_slow_axis10
nanequal_slow_dict[11] = nanequal_slow_axis11
nanequal_slow_dict[12] = nanequal_slow_axis12
nanequal_slow_dict[13] = nanequal_slow_axis13
nanequal_slow_dict[14] = nanequal_slow_axis14
nanequal_slow_dict[15] = nanequal_slow_axis15
nanequal_slow_dict[16] = nanequal_slow_axis16
nanequal_slow_dict[17] = nanequal_slow_axis17
nanequal_slow_dict[18] = nanequal_slow_axis18
nanequal_slow_dict[19] = nanequal_slow_axis19
nanequal_slow_dict[20] = nanequal_slow_axis20
nanequal_slow_dict[21] = nanequal_slow_axis21
nanequal_slow_dict[22] = nanequal_slow_axis22
nanequal_slow_dict[23] = nanequal_slow_axis23
nanequal_slow_dict[24] = nanequal_slow_axis24
nanequal_slow_dict[25] = nanequal_slow_axis25
nanequal_slow_dict[26] = nanequal_slow_axis26
nanequal_slow_dict[27] = nanequal_slow_axis27
nanequal_slow_dict[28] = nanequal_slow_axis28
nanequal_slow_dict[29] = nanequal_slow_axis29
nanequal_slow_dict[30] = nanequal_slow_axis30
nanequal_slow_dict[31] = nanequal_slow_axis31
nanequal_slow_dict[32] = nanequal_slow_axis32
nanequal_slow_dict[None] = nanequal_slow_axisNone