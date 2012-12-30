# 32 bit version
"nn auto-generated from template"

def nn(arr, arr0, int axis=1):
    """
    Distance of nearest neighbor (and its index) along specified axis.

    The Euclidian distance between `arr0` and its nearest neighbor in
    `arr` is returned along with the index of the nearest neighbor in
    `arr`.

    The squared distance used to determine the nearest neighbor of `arr0`
    is equivalent to np.sum((arr - arr0) ** 2), axis) where `arr` is 2d
    and `arr0` is 1d and `arr0` must be reshaped if `axis` is 1.

    If all distances are NaN then the distance returned is NaN and the
    index is zero.

    Parameters
    ----------
    arr : array_like
        A 2d array. If `arr` is not an array, a conversion is attempted.
    arr0 : array_like
        A 1d array. If `arr0` is not an array, a conversion is attempted.
    axis : int, optional
        Axis along which the distance is computed. The default (axis=1)
        is to compute the distance along rows.

    Returns
    -------
    dist : np.float64
        The Euclidian distance between `arr0` and the nearest neighbor
        in `arr`. If all distances are NaN then the distance returned
        is NaN.
    idx : int
        Index of nearest neighbor in `arr`. If all distances are NaN
        then the index returned is zero.

    See also
    --------
    bottleneck.ss: Sum of squares along specified axis.

    Notes
    -----
    A brute force algorithm is used to find the nearest neighbor.

    Depending on the shapes of `arr` and `arr0`, SciPy's cKDTree may
    be faster than bn.nn(). So benchmark if speed is important.
    
    The relative speed also depends on how many times you will use
    the same array `arr` to find nearest neighbors with different
    `arr0`. That is because it takes time to set up SciPy's cKDTree.

    Examples
    --------
    Create the input arrays:

    >>> arr = np.array([[1, 2], [3, 4]])
    >>> arr0 = np.array([2, 4])

    Find nearest neighbor of `arr0` in `arr` along axis 1:

    >>> dist, idx = bn.nn(arr, arr0, axis=1)
    >>> dist
    1.0
    >>> idx
    1

    Find nearest neighbor of `arr0` in `arr` along axis 0:

    >>> dist, idx = bn.nn(arr, arr0, axis=0)
    >>> dist
    0.0
    >>> idx
    1
    
    """
    func, a, a0 = nn_selector(arr, arr0, axis)
    return func(a, a0)

def nn_selector(arr, arr0, int axis):
    """
    Return nn function and arrays.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of dtype, and axis. A lot of the overhead in bn.nn() is in
    checking that `axis` is within range, converting `arr` into an array
    (if it is not already an array), and selecting the function to use to
    find the nearest neighbor.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using this function.

    Parameters
    ----------
    arr : array_like
        A 2d array. If `arr` is not an array, a conversion is attempted.
    arr0 : array_like
        A 1d array. If `arr0` is not an array, a conversion is attempted.
    axis : int, optional
        Axis along which the distance is computed. The default (axis=1)
        is to compute the distance along rows.

    Returns
    -------
    func : function
        The nn function that is appropriate to use with the given input.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.
    a0 : ndarray
        If the input array `arr0` is not a ndarray, then `a0` will contain the
        result of converting `arr0` into a ndarray.

    Examples
    --------
    Create the input arrays:

    >>> arr = np.array([[1, 2], [3, 4]])
    >>> arr0 = np.array([2, 4])
    
    Obtain the function needed to find the nearest neighbor of `arr0`
    in `arr0` along axis 0:

    >>> func, a, a0 = bn.func.nn_selector(arr, arr0, axis=0)
    >>> func
    <function nn_2d_int64_axis0>
    
    Use the returned function and arrays to determine the nearest
    neighbor:

    >>> dist, idx = func(a, a0)
    >>> dist
    0.0
    >>> idx
    1


    """
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:    
        a = np.array(arr, copy=False)
    cdef np.ndarray a0
    if type(arr0) is np.ndarray:
        a0 = arr0
    else:    
        a0 = np.array(arr0, copy=False)
    cdef int dtype = PyArray_TYPE(a)
    cdef int dtype0 = PyArray_TYPE(a0)
    if dtype != dtype0:
        raise ValueError("`arr` and `arr0` must be of the same dtype.")
    cdef int ndim = PyArray_NDIM(a)
    if ndim != 2:
        raise ValueError("`arr` must be 2d")
    cdef int ndim0 = PyArray_NDIM(a0)
    if ndim0 != 1:
        raise ValueError("`arr0` must be 1d")
    if axis < 0:
        axis += ndim    
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = nn_dict[key]
    except KeyError:
        if axis is not None:
            if (axis < 0) or (axis >= ndim):
                raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = nn_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a, a0

@cython.boundscheck(False)
@cython.wraparound(False)
def nn_2d_int8_axis0(np.ndarray[np.int8_t, ndim=2] a,
                              np.ndarray[np.int8_t, ndim=1] a0):
    "Nearest neighbor of 1d `a0` in 2d `a` with dtype=int8, axis=0."
    cdef:
        np.float64_t xsum = 0, d, xsummin=np.inf, dist
        Py_ssize_t imin = -1, n, a0size
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    a0size = PyArray_SIZE(a0)
    if n0 != a0size:
        raise ValueError("`a0` must match size of `a` along specified axis")
    for i1 in range(n1):
        xsum = 0
        for i0 in range(n0):
            d = a[i0, i1] - a0[i0]
            xsum += d * d
        if xsum < xsummin:
            xsummin = xsum
            imin = i1
    if imin == -1:
        dist = NAN
        imin = 0
    else:    
        dist = sqrt(xsummin)        
    return dist, imin

@cython.boundscheck(False)
@cython.wraparound(False)
def nn_2d_int8_axis1(np.ndarray[np.int8_t, ndim=2] a,
                              np.ndarray[np.int8_t, ndim=1] a0):
    "Nearest neighbor of 1d `a0` in 2d `a` with dtype=int8, axis=1."
    cdef:
        np.float64_t xsum = 0, d, xsummin=np.inf, dist
        Py_ssize_t imin = -1, n, a0size
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    a0size = PyArray_SIZE(a0)
    if n1 != a0size:
        raise ValueError("`a0` must match size of `a` along specified axis")
    for i0 in range(n0):
        xsum = 0
        for i1 in range(n1):
            d = a[i0, i1] - a0[i1]
            xsum += d * d
        if xsum < xsummin:
            xsummin = xsum
            imin = i0
    if imin == -1:
        dist = NAN
        imin = 0
    else:    
        dist = sqrt(xsummin)        
    return dist, imin

@cython.boundscheck(False)
@cython.wraparound(False)
def nn_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a,
                              np.ndarray[np.int32_t, ndim=1] a0):
    "Nearest neighbor of 1d `a0` in 2d `a` with dtype=int32, axis=0."
    cdef:
        np.float64_t xsum = 0, d, xsummin=np.inf, dist
        Py_ssize_t imin = -1, n, a0size
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    a0size = PyArray_SIZE(a0)
    if n0 != a0size:
        raise ValueError("`a0` must match size of `a` along specified axis")
    for i1 in range(n1):
        xsum = 0
        for i0 in range(n0):
            d = a[i0, i1] - a0[i0]
            xsum += d * d
        if xsum < xsummin:
            xsummin = xsum
            imin = i1
    if imin == -1:
        dist = NAN
        imin = 0
    else:    
        dist = sqrt(xsummin)        
    return dist, imin

@cython.boundscheck(False)
@cython.wraparound(False)
def nn_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a,
                              np.ndarray[np.int32_t, ndim=1] a0):
    "Nearest neighbor of 1d `a0` in 2d `a` with dtype=int32, axis=1."
    cdef:
        np.float64_t xsum = 0, d, xsummin=np.inf, dist
        Py_ssize_t imin = -1, n, a0size
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    a0size = PyArray_SIZE(a0)
    if n1 != a0size:
        raise ValueError("`a0` must match size of `a` along specified axis")
    for i0 in range(n0):
        xsum = 0
        for i1 in range(n1):
            d = a[i0, i1] - a0[i1]
            xsum += d * d
        if xsum < xsummin:
            xsummin = xsum
            imin = i0
    if imin == -1:
        dist = NAN
        imin = 0
    else:    
        dist = sqrt(xsummin)        
    return dist, imin

@cython.boundscheck(False)
@cython.wraparound(False)
def nn_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a,
                              np.ndarray[np.int64_t, ndim=1] a0):
    "Nearest neighbor of 1d `a0` in 2d `a` with dtype=int64, axis=0."
    cdef:
        np.float64_t xsum = 0, d, xsummin=np.inf, dist
        Py_ssize_t imin = -1, n, a0size
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    a0size = PyArray_SIZE(a0)
    if n0 != a0size:
        raise ValueError("`a0` must match size of `a` along specified axis")
    for i1 in range(n1):
        xsum = 0
        for i0 in range(n0):
            d = a[i0, i1] - a0[i0]
            xsum += d * d
        if xsum < xsummin:
            xsummin = xsum
            imin = i1
    if imin == -1:
        dist = NAN
        imin = 0
    else:    
        dist = sqrt(xsummin)        
    return dist, imin

@cython.boundscheck(False)
@cython.wraparound(False)
def nn_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a,
                              np.ndarray[np.int64_t, ndim=1] a0):
    "Nearest neighbor of 1d `a0` in 2d `a` with dtype=int64, axis=1."
    cdef:
        np.float64_t xsum = 0, d, xsummin=np.inf, dist
        Py_ssize_t imin = -1, n, a0size
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    a0size = PyArray_SIZE(a0)
    if n1 != a0size:
        raise ValueError("`a0` must match size of `a` along specified axis")
    for i0 in range(n0):
        xsum = 0
        for i1 in range(n1):
            d = a[i0, i1] - a0[i1]
            xsum += d * d
        if xsum < xsummin:
            xsummin = xsum
            imin = i0
    if imin == -1:
        dist = NAN
        imin = 0
    else:    
        dist = sqrt(xsummin)        
    return dist, imin

@cython.boundscheck(False)
@cython.wraparound(False)
def nn_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a,
                              np.ndarray[np.float32_t, ndim=1] a0):
    "Nearest neighbor of 1d `a0` in 2d `a` with dtype=float32, axis=0."
    cdef:
        np.float64_t xsum = 0, d, xsummin=np.inf, dist
        Py_ssize_t imin = -1, n, a0size
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    a0size = PyArray_SIZE(a0)
    if n0 != a0size:
        raise ValueError("`a0` must match size of `a` along specified axis")
    for i1 in range(n1):
        xsum = 0
        for i0 in range(n0):
            d = a[i0, i1] - a0[i0]
            xsum += d * d
        if xsum < xsummin:
            xsummin = xsum
            imin = i1
    if imin == -1:
        dist = NAN
        imin = 0
    else:    
        dist = sqrt(xsummin)        
    return dist, imin

@cython.boundscheck(False)
@cython.wraparound(False)
def nn_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a,
                              np.ndarray[np.float32_t, ndim=1] a0):
    "Nearest neighbor of 1d `a0` in 2d `a` with dtype=float32, axis=1."
    cdef:
        np.float64_t xsum = 0, d, xsummin=np.inf, dist
        Py_ssize_t imin = -1, n, a0size
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    a0size = PyArray_SIZE(a0)
    if n1 != a0size:
        raise ValueError("`a0` must match size of `a` along specified axis")
    for i0 in range(n0):
        xsum = 0
        for i1 in range(n1):
            d = a[i0, i1] - a0[i1]
            xsum += d * d
        if xsum < xsummin:
            xsummin = xsum
            imin = i0
    if imin == -1:
        dist = NAN
        imin = 0
    else:    
        dist = sqrt(xsummin)        
    return dist, imin

@cython.boundscheck(False)
@cython.wraparound(False)
def nn_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a,
                              np.ndarray[np.float64_t, ndim=1] a0):
    "Nearest neighbor of 1d `a0` in 2d `a` with dtype=float64, axis=0."
    cdef:
        np.float64_t xsum = 0, d, xsummin=np.inf, dist
        Py_ssize_t imin = -1, n, a0size
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    a0size = PyArray_SIZE(a0)
    if n0 != a0size:
        raise ValueError("`a0` must match size of `a` along specified axis")
    for i1 in range(n1):
        xsum = 0
        for i0 in range(n0):
            d = a[i0, i1] - a0[i0]
            xsum += d * d
        if xsum < xsummin:
            xsummin = xsum
            imin = i1
    if imin == -1:
        dist = NAN
        imin = 0
    else:    
        dist = sqrt(xsummin)        
    return dist, imin

@cython.boundscheck(False)
@cython.wraparound(False)
def nn_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a,
                              np.ndarray[np.float64_t, ndim=1] a0):
    "Nearest neighbor of 1d `a0` in 2d `a` with dtype=float64, axis=1."
    cdef:
        np.float64_t xsum = 0, d, xsummin=np.inf, dist
        Py_ssize_t imin = -1, n, a0size
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    a0size = PyArray_SIZE(a0)
    if n1 != a0size:
        raise ValueError("`a0` must match size of `a` along specified axis")
    for i0 in range(n0):
        xsum = 0
        for i1 in range(n1):
            d = a[i0, i1] - a0[i1]
            xsum += d * d
        if xsum < xsummin:
            xsummin = xsum
            imin = i0
    if imin == -1:
        dist = NAN
        imin = 0
    else:    
        dist = sqrt(xsummin)        
    return dist, imin

cdef dict nn_dict = {}
nn_dict[(2, NPY_int8, 0)] = nn_2d_int8_axis0
nn_dict[(2, NPY_int8, 1)] = nn_2d_int8_axis1
nn_dict[(2, NPY_int32, 0)] = nn_2d_int32_axis0
nn_dict[(2, NPY_int32, 1)] = nn_2d_int32_axis1
nn_dict[(2, NPY_int64, 0)] = nn_2d_int64_axis0
nn_dict[(2, NPY_int64, 1)] = nn_2d_int64_axis1
nn_dict[(2, NPY_float32, 0)] = nn_2d_float32_axis0
nn_dict[(2, NPY_float32, 1)] = nn_2d_float32_axis1
nn_dict[(2, NPY_float64, 0)] = nn_2d_float64_axis0
nn_dict[(2, NPY_float64, 1)] = nn_2d_float64_axis1

def nn_slow_axis0(arr, arr0):
    "Unaccelerated (slow) nn along axis 0."
    return bn.slow.nn(arr, arr0, axis=0)

def nn_slow_axis1(arr, arr0):
    "Unaccelerated (slow) nn along axis 1."
    return bn.slow.nn(arr, arr0, axis=1)

def nn_slow_axis2(arr, arr0):
    "Unaccelerated (slow) nn along axis 2."
    return bn.slow.nn(arr, arr0, axis=2)

def nn_slow_axis3(arr, arr0):
    "Unaccelerated (slow) nn along axis 3."
    return bn.slow.nn(arr, arr0, axis=3)

def nn_slow_axis4(arr, arr0):
    "Unaccelerated (slow) nn along axis 4."
    return bn.slow.nn(arr, arr0, axis=4)

def nn_slow_axis5(arr, arr0):
    "Unaccelerated (slow) nn along axis 5."
    return bn.slow.nn(arr, arr0, axis=5)

def nn_slow_axis6(arr, arr0):
    "Unaccelerated (slow) nn along axis 6."
    return bn.slow.nn(arr, arr0, axis=6)

def nn_slow_axis7(arr, arr0):
    "Unaccelerated (slow) nn along axis 7."
    return bn.slow.nn(arr, arr0, axis=7)

def nn_slow_axis8(arr, arr0):
    "Unaccelerated (slow) nn along axis 8."
    return bn.slow.nn(arr, arr0, axis=8)

def nn_slow_axis9(arr, arr0):
    "Unaccelerated (slow) nn along axis 9."
    return bn.slow.nn(arr, arr0, axis=9)

def nn_slow_axis10(arr, arr0):
    "Unaccelerated (slow) nn along axis 10."
    return bn.slow.nn(arr, arr0, axis=10)

def nn_slow_axis11(arr, arr0):
    "Unaccelerated (slow) nn along axis 11."
    return bn.slow.nn(arr, arr0, axis=11)

def nn_slow_axis12(arr, arr0):
    "Unaccelerated (slow) nn along axis 12."
    return bn.slow.nn(arr, arr0, axis=12)

def nn_slow_axis13(arr, arr0):
    "Unaccelerated (slow) nn along axis 13."
    return bn.slow.nn(arr, arr0, axis=13)

def nn_slow_axis14(arr, arr0):
    "Unaccelerated (slow) nn along axis 14."
    return bn.slow.nn(arr, arr0, axis=14)

def nn_slow_axis15(arr, arr0):
    "Unaccelerated (slow) nn along axis 15."
    return bn.slow.nn(arr, arr0, axis=15)

def nn_slow_axis16(arr, arr0):
    "Unaccelerated (slow) nn along axis 16."
    return bn.slow.nn(arr, arr0, axis=16)

def nn_slow_axis17(arr, arr0):
    "Unaccelerated (slow) nn along axis 17."
    return bn.slow.nn(arr, arr0, axis=17)

def nn_slow_axis18(arr, arr0):
    "Unaccelerated (slow) nn along axis 18."
    return bn.slow.nn(arr, arr0, axis=18)

def nn_slow_axis19(arr, arr0):
    "Unaccelerated (slow) nn along axis 19."
    return bn.slow.nn(arr, arr0, axis=19)

def nn_slow_axis20(arr, arr0):
    "Unaccelerated (slow) nn along axis 20."
    return bn.slow.nn(arr, arr0, axis=20)

def nn_slow_axis21(arr, arr0):
    "Unaccelerated (slow) nn along axis 21."
    return bn.slow.nn(arr, arr0, axis=21)

def nn_slow_axis22(arr, arr0):
    "Unaccelerated (slow) nn along axis 22."
    return bn.slow.nn(arr, arr0, axis=22)

def nn_slow_axis23(arr, arr0):
    "Unaccelerated (slow) nn along axis 23."
    return bn.slow.nn(arr, arr0, axis=23)

def nn_slow_axis24(arr, arr0):
    "Unaccelerated (slow) nn along axis 24."
    return bn.slow.nn(arr, arr0, axis=24)

def nn_slow_axis25(arr, arr0):
    "Unaccelerated (slow) nn along axis 25."
    return bn.slow.nn(arr, arr0, axis=25)

def nn_slow_axis26(arr, arr0):
    "Unaccelerated (slow) nn along axis 26."
    return bn.slow.nn(arr, arr0, axis=26)

def nn_slow_axis27(arr, arr0):
    "Unaccelerated (slow) nn along axis 27."
    return bn.slow.nn(arr, arr0, axis=27)

def nn_slow_axis28(arr, arr0):
    "Unaccelerated (slow) nn along axis 28."
    return bn.slow.nn(arr, arr0, axis=28)

def nn_slow_axis29(arr, arr0):
    "Unaccelerated (slow) nn along axis 29."
    return bn.slow.nn(arr, arr0, axis=29)

def nn_slow_axis30(arr, arr0):
    "Unaccelerated (slow) nn along axis 30."
    return bn.slow.nn(arr, arr0, axis=30)

def nn_slow_axis31(arr, arr0):
    "Unaccelerated (slow) nn along axis 31."
    return bn.slow.nn(arr, arr0, axis=31)

def nn_slow_axis32(arr, arr0):
    "Unaccelerated (slow) nn along axis 32."
    return bn.slow.nn(arr, arr0, axis=32)

def nn_slow_axisNone(arr, arr0):
    "Unaccelerated (slow) nn along axis None."
    return bn.slow.nn(arr, arr0, axis=None)


cdef dict nn_slow_dict = {}
nn_slow_dict[0] = nn_slow_axis0
nn_slow_dict[1] = nn_slow_axis1
nn_slow_dict[2] = nn_slow_axis2
nn_slow_dict[3] = nn_slow_axis3
nn_slow_dict[4] = nn_slow_axis4
nn_slow_dict[5] = nn_slow_axis5
nn_slow_dict[6] = nn_slow_axis6
nn_slow_dict[7] = nn_slow_axis7
nn_slow_dict[8] = nn_slow_axis8
nn_slow_dict[9] = nn_slow_axis9
nn_slow_dict[10] = nn_slow_axis10
nn_slow_dict[11] = nn_slow_axis11
nn_slow_dict[12] = nn_slow_axis12
nn_slow_dict[13] = nn_slow_axis13
nn_slow_dict[14] = nn_slow_axis14
nn_slow_dict[15] = nn_slow_axis15
nn_slow_dict[16] = nn_slow_axis16
nn_slow_dict[17] = nn_slow_axis17
nn_slow_dict[18] = nn_slow_axis18
nn_slow_dict[19] = nn_slow_axis19
nn_slow_dict[20] = nn_slow_axis20
nn_slow_dict[21] = nn_slow_axis21
nn_slow_dict[22] = nn_slow_axis22
nn_slow_dict[23] = nn_slow_axis23
nn_slow_dict[24] = nn_slow_axis24
nn_slow_dict[25] = nn_slow_axis25
nn_slow_dict[26] = nn_slow_axis26
nn_slow_dict[27] = nn_slow_axis27
nn_slow_dict[28] = nn_slow_axis28
nn_slow_dict[29] = nn_slow_axis29
nn_slow_dict[30] = nn_slow_axis30
nn_slow_dict[31] = nn_slow_axis31
nn_slow_dict[32] = nn_slow_axis32
nn_slow_dict[None] = nn_slow_axisNone