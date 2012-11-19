# 32 bit version
"nanrankdata auto-generated from template"

def nanrankdata(arr, axis=None):
    """
    Ranks the data, dealing with ties and NaNs appropriately.

    Equal values are assigned a rank that is the average of the ranks that
    would have been otherwise assigned to all of the values within that set.
    Ranks begin at 1, not 0.

    NaNs in the input array are returned as NaNs.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the elements of the array are ranked. The default
        (axis=None) is to rank the elements of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`. The dtype is 'float64'.
    
    See also
    --------
    bottleneck.rankdata: Ranks the data, dealing with ties and appropriately.
    
    Examples
    --------
    >>> bn.nanrankdata([np.nan, 2, 2, 3])
    array([ nan,  1.5,  1.5,  3. ])
    >>> bn.nanrankdata([[np.nan, 2], [2, 3]])
    array([ nan,  1.5,  1.5,  3. ])
    >>> bn.nanrankdata([[np.nan, 2], [2, 3]], axis=0)
    array([[ nan,   1.],
           [  1.,   2.]])
    >>> bn.nanrankdata([[np.nan, 2], [2, 3]], axis=1)
    array([[ nan,   1.],
           [  1.,   2.]])
    
    """
    func, arr = nanrankdata_selector(arr, axis)
    return func(arr)

def nanrankdata_selector(arr, axis):
    """
    Return nanrankdata function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.nanrankdata() is in checking that `axis` is within range, converting
    `arr` into an array (if it is not already an array), and selecting the
    function to use to rank the elements.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}
        Axis along which to rank the elements of the array.
    
    Returns
    -------
    func : function
        The nanrankdata function that matches the number of dimensions and
        dtype of the input array and the axis along which you wish to rank.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([np.nan, 2, 2, 3])
    
    Obtain the function needed to rank the elements of `arr` along axis=0:

    >>> func, a = bn.func.nanrankdata_selector(arr, axis=0)
    >>> func
    <function nanrankdata_1d_float64_axis0>
    
    Use the returned function and array:

    >>> func(a)
    array([ nan,  1.5,  1.5,  3. ])

    """
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:    
        a = np.array(arr, copy=False)
    cdef tuple key
    cdef int ndim = PyArray_NDIM(a)
    cdef int dtype = PyArray_TYPE(a)
    if axis is not None:
        if axis < 0:
            axis += ndim
    else:
        a = PyArray_Ravel(a, NPY_CORDER)
        axis = 0
        ndim = 1
    key = (ndim, dtype, axis)
    try:
        func = nanrankdata_dict[key]
    except KeyError:
        if (axis < 0) or (axis >= ndim):
            raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = nanrankdata_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a

@cython.boundscheck(False)
@cython.wraparound(False)
def nanrankdata_1d_float32_axis0(np.ndarray[np.float32_t, ndim=1] a):
    "Ranks n1d array with dtype=float32 along axis=0, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j=0, k, idx
    cdef np.ndarray[np.intp_t, ndim=1] ivec = PyArray_ArgSort(a, 0, NPY_QUICKSORT)
    cdef np.float64_t old, new, averank, sumranks = 0
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
		NPY_float64, 0)
    if n0 == 0:
        PyArray_FillWithScalar(y, NAN)
        return y
    old = a[ivec[0]]
    for i0 in xrange(n0-1):
        sumranks += i0
        dupcount += 1
        k = i0 + 1
        new = a[ivec[k]]
        if old != new:
            if old == old:
                averank = sumranks / dupcount + 1
                for j in xrange(k - dupcount, k):
                    y[ivec[j]] = averank
            else:
                y[ivec[i0]] = NAN
            sumranks = 0
            dupcount = 0
        old = new    
    sumranks += (n0 - 1)
    dupcount += 1
    if old == old:
        averank = sumranks / dupcount + 1
        for j in xrange(n0 - dupcount, n0):
                y[ivec[j]] = averank
    else:
        y[ivec[n0 - 1]] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanrankdata_1d_float64_axis0(np.ndarray[np.float64_t, ndim=1] a):
    "Ranks n1d array with dtype=float64 along axis=0, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j=0, k, idx
    cdef np.ndarray[np.intp_t, ndim=1] ivec = PyArray_ArgSort(a, 0, NPY_QUICKSORT)
    cdef np.float64_t old, new, averank, sumranks = 0
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
		NPY_float64, 0)
    if n0 == 0:
        PyArray_FillWithScalar(y, NAN)
        return y
    old = a[ivec[0]]
    for i0 in xrange(n0-1):
        sumranks += i0
        dupcount += 1
        k = i0 + 1
        new = a[ivec[k]]
        if old != new:
            if old == old:
                averank = sumranks / dupcount + 1
                for j in xrange(k - dupcount, k):
                    y[ivec[j]] = averank
            else:
                y[ivec[i0]] = NAN
            sumranks = 0
            dupcount = 0
        old = new    
    sumranks += (n0 - 1)
    dupcount += 1
    if old == old:
        averank = sumranks / dupcount + 1
        for j in xrange(n0 - dupcount, n0):
                y[ivec[j]] = averank
    else:
        y[ivec[n0 - 1]] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanrankdata_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a):
    "Ranks n2d array with dtype=float32 along axis=0, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j=0, k, idx
    cdef np.ndarray[np.intp_t, ndim=2] ivec = PyArray_ArgSort(a, 0, NPY_QUICKSORT)
    cdef np.float64_t old, new, averank, sumranks = 0
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
		NPY_float64, 0)
    if n0 == 0:
        PyArray_FillWithScalar(y, NAN)
        return y
    for i1 in xrange(n1):
        idx = ivec[0, i1]
        old = a[idx, i1]
        sumranks = 0
        dupcount = 0
        for i0 in xrange(n0-1):
            sumranks += i0
            dupcount += 1
            k = i0 + 1
            idx = ivec[k, i1]
            new = a[idx, i1]
            if old != new:
                if old == old:
                    averank = sumranks / dupcount + 1
                    for j in xrange(k - dupcount, k):
                        idx = ivec[j, i1]
                        y[idx, i1] = averank
                else:
                    idx = ivec[i0, i1]
                    y[idx, i1] = NAN
                sumranks = 0
                dupcount = 0
            old = new    
        sumranks += (n0 - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        if old == old:
            for j in xrange(n0 - dupcount, n0):
                idx = ivec[j, i1]
                y[idx, i1] = averank
        else:
            idx = ivec[n0 - 1, i1]
            y[idx, i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanrankdata_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a):
    "Ranks n2d array with dtype=float32 along axis=1, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j=0, k, idx
    cdef np.ndarray[np.intp_t, ndim=2] ivec = PyArray_ArgSort(a, 1, NPY_QUICKSORT)
    cdef np.float64_t old, new, averank, sumranks = 0
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
		NPY_float64, 0)
    if n1 == 0:
        PyArray_FillWithScalar(y, NAN)
        return y
    for i0 in xrange(n0):
        idx = ivec[i0, 0]
        old = a[i0, idx]
        sumranks = 0
        dupcount = 0
        for i1 in xrange(n1-1):
            sumranks += i1
            dupcount += 1
            k = i1 + 1
            idx = ivec[i0, k]
            new = a[i0, idx]
            if old != new:
                if old == old:
                    averank = sumranks / dupcount + 1
                    for j in xrange(k - dupcount, k):
                        idx = ivec[i0, j]
                        y[i0, idx] = averank
                else:
                    idx = ivec[i0, i1]
                    y[i0, idx] = NAN
                sumranks = 0
                dupcount = 0
            old = new    
        sumranks += (n1 - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        if old == old:
            for j in xrange(n1 - dupcount, n1):
                idx = ivec[i0, j]
                y[i0, idx] = averank
        else:
            idx = ivec[i0, n1 - 1]
            y[i0, idx] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanrankdata_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "Ranks n2d array with dtype=float64 along axis=0, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j=0, k, idx
    cdef np.ndarray[np.intp_t, ndim=2] ivec = PyArray_ArgSort(a, 0, NPY_QUICKSORT)
    cdef np.float64_t old, new, averank, sumranks = 0
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
		NPY_float64, 0)
    if n0 == 0:
        PyArray_FillWithScalar(y, NAN)
        return y
    for i1 in xrange(n1):
        idx = ivec[0, i1]
        old = a[idx, i1]
        sumranks = 0
        dupcount = 0
        for i0 in xrange(n0-1):
            sumranks += i0
            dupcount += 1
            k = i0 + 1
            idx = ivec[k, i1]
            new = a[idx, i1]
            if old != new:
                if old == old:
                    averank = sumranks / dupcount + 1
                    for j in xrange(k - dupcount, k):
                        idx = ivec[j, i1]
                        y[idx, i1] = averank
                else:
                    idx = ivec[i0, i1]
                    y[idx, i1] = NAN
                sumranks = 0
                dupcount = 0
            old = new    
        sumranks += (n0 - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        if old == old:
            for j in xrange(n0 - dupcount, n0):
                idx = ivec[j, i1]
                y[idx, i1] = averank
        else:
            idx = ivec[n0 - 1, i1]
            y[idx, i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanrankdata_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "Ranks n2d array with dtype=float64 along axis=1, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j=0, k, idx
    cdef np.ndarray[np.intp_t, ndim=2] ivec = PyArray_ArgSort(a, 1, NPY_QUICKSORT)
    cdef np.float64_t old, new, averank, sumranks = 0
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
		NPY_float64, 0)
    if n1 == 0:
        PyArray_FillWithScalar(y, NAN)
        return y
    for i0 in xrange(n0):
        idx = ivec[i0, 0]
        old = a[i0, idx]
        sumranks = 0
        dupcount = 0
        for i1 in xrange(n1-1):
            sumranks += i1
            dupcount += 1
            k = i1 + 1
            idx = ivec[i0, k]
            new = a[i0, idx]
            if old != new:
                if old == old:
                    averank = sumranks / dupcount + 1
                    for j in xrange(k - dupcount, k):
                        idx = ivec[i0, j]
                        y[i0, idx] = averank
                else:
                    idx = ivec[i0, i1]
                    y[i0, idx] = NAN
                sumranks = 0
                dupcount = 0
            old = new    
        sumranks += (n1 - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        if old == old:
            for j in xrange(n1 - dupcount, n1):
                idx = ivec[i0, j]
                y[i0, idx] = averank
        else:
            idx = ivec[i0, n1 - 1]
            y[i0, idx] = NAN
    return y

cdef dict nanrankdata_dict = {}
nanrankdata_dict[(1, NPY_int32, 0)] = rankdata_1d_int32_axis0
nanrankdata_dict[(1, NPY_int64, 0)] = rankdata_1d_int64_axis0
nanrankdata_dict[(2, NPY_int32, 0)] = rankdata_2d_int32_axis0
nanrankdata_dict[(2, NPY_int32, 1)] = rankdata_2d_int32_axis1
nanrankdata_dict[(2, NPY_int64, 0)] = rankdata_2d_int64_axis0
nanrankdata_dict[(2, NPY_int64, 1)] = rankdata_2d_int64_axis1
nanrankdata_dict[(1, NPY_float32, 0)] = nanrankdata_1d_float32_axis0
nanrankdata_dict[(1, NPY_float64, 0)] = nanrankdata_1d_float64_axis0
nanrankdata_dict[(2, NPY_float32, 0)] = nanrankdata_2d_float32_axis0
nanrankdata_dict[(2, NPY_float32, 1)] = nanrankdata_2d_float32_axis1
nanrankdata_dict[(2, NPY_float64, 0)] = nanrankdata_2d_float64_axis0
nanrankdata_dict[(2, NPY_float64, 1)] = nanrankdata_2d_float64_axis1

def nanrankdata_slow_axis0(arr):
    "Unaccelerated (slow) nanrankdata along axis 0."
    return bn.slow.nanrankdata(arr, axis=0)

def nanrankdata_slow_axis1(arr):
    "Unaccelerated (slow) nanrankdata along axis 1."
    return bn.slow.nanrankdata(arr, axis=1)

def nanrankdata_slow_axis2(arr):
    "Unaccelerated (slow) nanrankdata along axis 2."
    return bn.slow.nanrankdata(arr, axis=2)

def nanrankdata_slow_axis3(arr):
    "Unaccelerated (slow) nanrankdata along axis 3."
    return bn.slow.nanrankdata(arr, axis=3)

def nanrankdata_slow_axis4(arr):
    "Unaccelerated (slow) nanrankdata along axis 4."
    return bn.slow.nanrankdata(arr, axis=4)

def nanrankdata_slow_axis5(arr):
    "Unaccelerated (slow) nanrankdata along axis 5."
    return bn.slow.nanrankdata(arr, axis=5)

def nanrankdata_slow_axis6(arr):
    "Unaccelerated (slow) nanrankdata along axis 6."
    return bn.slow.nanrankdata(arr, axis=6)

def nanrankdata_slow_axis7(arr):
    "Unaccelerated (slow) nanrankdata along axis 7."
    return bn.slow.nanrankdata(arr, axis=7)

def nanrankdata_slow_axis8(arr):
    "Unaccelerated (slow) nanrankdata along axis 8."
    return bn.slow.nanrankdata(arr, axis=8)

def nanrankdata_slow_axis9(arr):
    "Unaccelerated (slow) nanrankdata along axis 9."
    return bn.slow.nanrankdata(arr, axis=9)

def nanrankdata_slow_axis10(arr):
    "Unaccelerated (slow) nanrankdata along axis 10."
    return bn.slow.nanrankdata(arr, axis=10)

def nanrankdata_slow_axis11(arr):
    "Unaccelerated (slow) nanrankdata along axis 11."
    return bn.slow.nanrankdata(arr, axis=11)

def nanrankdata_slow_axis12(arr):
    "Unaccelerated (slow) nanrankdata along axis 12."
    return bn.slow.nanrankdata(arr, axis=12)

def nanrankdata_slow_axis13(arr):
    "Unaccelerated (slow) nanrankdata along axis 13."
    return bn.slow.nanrankdata(arr, axis=13)

def nanrankdata_slow_axis14(arr):
    "Unaccelerated (slow) nanrankdata along axis 14."
    return bn.slow.nanrankdata(arr, axis=14)

def nanrankdata_slow_axis15(arr):
    "Unaccelerated (slow) nanrankdata along axis 15."
    return bn.slow.nanrankdata(arr, axis=15)

def nanrankdata_slow_axis16(arr):
    "Unaccelerated (slow) nanrankdata along axis 16."
    return bn.slow.nanrankdata(arr, axis=16)

def nanrankdata_slow_axis17(arr):
    "Unaccelerated (slow) nanrankdata along axis 17."
    return bn.slow.nanrankdata(arr, axis=17)

def nanrankdata_slow_axis18(arr):
    "Unaccelerated (slow) nanrankdata along axis 18."
    return bn.slow.nanrankdata(arr, axis=18)

def nanrankdata_slow_axis19(arr):
    "Unaccelerated (slow) nanrankdata along axis 19."
    return bn.slow.nanrankdata(arr, axis=19)

def nanrankdata_slow_axis20(arr):
    "Unaccelerated (slow) nanrankdata along axis 20."
    return bn.slow.nanrankdata(arr, axis=20)

def nanrankdata_slow_axis21(arr):
    "Unaccelerated (slow) nanrankdata along axis 21."
    return bn.slow.nanrankdata(arr, axis=21)

def nanrankdata_slow_axis22(arr):
    "Unaccelerated (slow) nanrankdata along axis 22."
    return bn.slow.nanrankdata(arr, axis=22)

def nanrankdata_slow_axis23(arr):
    "Unaccelerated (slow) nanrankdata along axis 23."
    return bn.slow.nanrankdata(arr, axis=23)

def nanrankdata_slow_axis24(arr):
    "Unaccelerated (slow) nanrankdata along axis 24."
    return bn.slow.nanrankdata(arr, axis=24)

def nanrankdata_slow_axis25(arr):
    "Unaccelerated (slow) nanrankdata along axis 25."
    return bn.slow.nanrankdata(arr, axis=25)

def nanrankdata_slow_axis26(arr):
    "Unaccelerated (slow) nanrankdata along axis 26."
    return bn.slow.nanrankdata(arr, axis=26)

def nanrankdata_slow_axis27(arr):
    "Unaccelerated (slow) nanrankdata along axis 27."
    return bn.slow.nanrankdata(arr, axis=27)

def nanrankdata_slow_axis28(arr):
    "Unaccelerated (slow) nanrankdata along axis 28."
    return bn.slow.nanrankdata(arr, axis=28)

def nanrankdata_slow_axis29(arr):
    "Unaccelerated (slow) nanrankdata along axis 29."
    return bn.slow.nanrankdata(arr, axis=29)

def nanrankdata_slow_axis30(arr):
    "Unaccelerated (slow) nanrankdata along axis 30."
    return bn.slow.nanrankdata(arr, axis=30)

def nanrankdata_slow_axis31(arr):
    "Unaccelerated (slow) nanrankdata along axis 31."
    return bn.slow.nanrankdata(arr, axis=31)

def nanrankdata_slow_axis32(arr):
    "Unaccelerated (slow) nanrankdata along axis 32."
    return bn.slow.nanrankdata(arr, axis=32)

def nanrankdata_slow_axisNone(arr):
    "Unaccelerated (slow) nanrankdata along axis None."
    return bn.slow.nanrankdata(arr, axis=None)


cdef dict nanrankdata_slow_dict = {}
nanrankdata_slow_dict[0] = nanrankdata_slow_axis0
nanrankdata_slow_dict[1] = nanrankdata_slow_axis1
nanrankdata_slow_dict[2] = nanrankdata_slow_axis2
nanrankdata_slow_dict[3] = nanrankdata_slow_axis3
nanrankdata_slow_dict[4] = nanrankdata_slow_axis4
nanrankdata_slow_dict[5] = nanrankdata_slow_axis5
nanrankdata_slow_dict[6] = nanrankdata_slow_axis6
nanrankdata_slow_dict[7] = nanrankdata_slow_axis7
nanrankdata_slow_dict[8] = nanrankdata_slow_axis8
nanrankdata_slow_dict[9] = nanrankdata_slow_axis9
nanrankdata_slow_dict[10] = nanrankdata_slow_axis10
nanrankdata_slow_dict[11] = nanrankdata_slow_axis11
nanrankdata_slow_dict[12] = nanrankdata_slow_axis12
nanrankdata_slow_dict[13] = nanrankdata_slow_axis13
nanrankdata_slow_dict[14] = nanrankdata_slow_axis14
nanrankdata_slow_dict[15] = nanrankdata_slow_axis15
nanrankdata_slow_dict[16] = nanrankdata_slow_axis16
nanrankdata_slow_dict[17] = nanrankdata_slow_axis17
nanrankdata_slow_dict[18] = nanrankdata_slow_axis18
nanrankdata_slow_dict[19] = nanrankdata_slow_axis19
nanrankdata_slow_dict[20] = nanrankdata_slow_axis20
nanrankdata_slow_dict[21] = nanrankdata_slow_axis21
nanrankdata_slow_dict[22] = nanrankdata_slow_axis22
nanrankdata_slow_dict[23] = nanrankdata_slow_axis23
nanrankdata_slow_dict[24] = nanrankdata_slow_axis24
nanrankdata_slow_dict[25] = nanrankdata_slow_axis25
nanrankdata_slow_dict[26] = nanrankdata_slow_axis26
nanrankdata_slow_dict[27] = nanrankdata_slow_axis27
nanrankdata_slow_dict[28] = nanrankdata_slow_axis28
nanrankdata_slow_dict[29] = nanrankdata_slow_axis29
nanrankdata_slow_dict[30] = nanrankdata_slow_axis30
nanrankdata_slow_dict[31] = nanrankdata_slow_axis31
nanrankdata_slow_dict[32] = nanrankdata_slow_axis32
nanrankdata_slow_dict[None] = nanrankdata_slow_axisNone