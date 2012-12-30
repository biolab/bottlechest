# 32 bit version
"rankdata auto-generated from template"

def rankdata(arr, axis=None):
    """
    Ranks the data, dealing with ties appropriately.

    Equal values are assigned a rank that is the average of the ranks that
    would have been otherwise assigned to all of the values within that set.
    Ranks begin at 1, not 0.

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
    bottleneck.nanrankdata: Ranks the data dealing with ties and NaNs.
    
    Examples
    --------
    >>> bn.rankdata([0, 2, 2, 3])
    array([ 1. ,  2.5,  2.5,  4. ])
    >>> bn.rankdata([[0, 2], [2, 3]])
    array([ 1. ,  2.5,  2.5,  4. ])
    >>> bn.rankdata([[0, 2], [2, 3]], axis=0)     
    array([[ 1.,  1.],
           [ 2.,  2.]])
    >>> bn.rankdata([[0, 2], [2, 3]], axis=1)
    array([[ 1.,  2.],
           [ 1.,  2.]])
    
    """
    func, arr = rankdata_selector(arr, axis)
    return func(arr)

def rankdata_selector(arr, axis):
    """
    Return rankdata function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.rankdata() is in checking that `axis` is within range, converting `arr`
    into an array (if it is not already an array), and selecting the function
    to use to rank the elements.

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
        The rankdata function that matches the number of dimensions and dtype
        of the input array and the axis along which you wish to rank.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([0, 2, 2, 3])
    
    Obtain the function needed to rank the elements of `arr` along axis=0:

    >>> func, a = bn.func.rankdata_selector(arr, axis=0)
    >>> func
    <function rankdata_1d_int64_axis0>
    
    Use the returned function and array:

    >>> func(a)
    array([ 1. ,  2.5,  2.5,  4. ])

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
        func = rankdata_dict[key]
    except KeyError:
        if (axis < 0) or (axis >= ndim):
            raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = rankdata_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a

@cython.boundscheck(False)
@cython.wraparound(False)
def rankdata_1d_int8_axis0(np.ndarray[np.int8_t, ndim=1] a):
    "Ranks n1d array with dtype=int8 along axis=0, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j, k, idx
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
            averank = sumranks / dupcount + 1
            for j in xrange(k - dupcount, k):
                y[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
        old = new    
    sumranks += (n0 - 1)
    dupcount += 1 
    averank = sumranks / dupcount + 1
    for j in xrange(n0 - dupcount, n0):
        y[ivec[j]] = averank
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def rankdata_1d_int32_axis0(np.ndarray[np.int32_t, ndim=1] a):
    "Ranks n1d array with dtype=int32 along axis=0, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j, k, idx
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
            averank = sumranks / dupcount + 1
            for j in xrange(k - dupcount, k):
                y[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
        old = new    
    sumranks += (n0 - 1)
    dupcount += 1 
    averank = sumranks / dupcount + 1
    for j in xrange(n0 - dupcount, n0):
        y[ivec[j]] = averank
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def rankdata_1d_int64_axis0(np.ndarray[np.int64_t, ndim=1] a):
    "Ranks n1d array with dtype=int64 along axis=0, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j, k, idx
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
            averank = sumranks / dupcount + 1
            for j in xrange(k - dupcount, k):
                y[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
        old = new    
    sumranks += (n0 - 1)
    dupcount += 1 
    averank = sumranks / dupcount + 1
    for j in xrange(n0 - dupcount, n0):
        y[ivec[j]] = averank
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def rankdata_2d_int8_axis0(np.ndarray[np.int8_t, ndim=2] a):
    "Ranks n2d array with dtype=int8 along axis=0, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j, k, idx
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
                averank = sumranks / dupcount + 1
                for j in xrange(k - dupcount, k):
                    idx = ivec[j, i1]
                    y[idx, i1] = averank
                sumranks = 0
                dupcount = 0
            old = new    
        sumranks += (n0 - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        for j in xrange(n0 - dupcount, n0):
            idx = ivec[j, i1]
            y[idx, i1] = averank
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def rankdata_2d_int8_axis1(np.ndarray[np.int8_t, ndim=2] a):
    "Ranks n2d array with dtype=int8 along axis=1, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j, k, idx
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
                averank = sumranks / dupcount + 1
                for j in xrange(k - dupcount, k):
                    idx = ivec[i0, j]
                    y[i0, idx] = averank
                sumranks = 0
                dupcount = 0
            old = new    
        sumranks += (n1 - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        for j in xrange(n1 - dupcount, n1):
            idx = ivec[i0, j]
            y[i0, idx] = averank
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def rankdata_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "Ranks n2d array with dtype=int32 along axis=0, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j, k, idx
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
                averank = sumranks / dupcount + 1
                for j in xrange(k - dupcount, k):
                    idx = ivec[j, i1]
                    y[idx, i1] = averank
                sumranks = 0
                dupcount = 0
            old = new    
        sumranks += (n0 - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        for j in xrange(n0 - dupcount, n0):
            idx = ivec[j, i1]
            y[idx, i1] = averank
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def rankdata_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a):
    "Ranks n2d array with dtype=int32 along axis=1, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j, k, idx
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
                averank = sumranks / dupcount + 1
                for j in xrange(k - dupcount, k):
                    idx = ivec[i0, j]
                    y[i0, idx] = averank
                sumranks = 0
                dupcount = 0
            old = new    
        sumranks += (n1 - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        for j in xrange(n1 - dupcount, n1):
            idx = ivec[i0, j]
            y[i0, idx] = averank
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def rankdata_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a):
    "Ranks n2d array with dtype=int64 along axis=0, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j, k, idx
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
                averank = sumranks / dupcount + 1
                for j in xrange(k - dupcount, k):
                    idx = ivec[j, i1]
                    y[idx, i1] = averank
                sumranks = 0
                dupcount = 0
            old = new    
        sumranks += (n0 - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        for j in xrange(n0 - dupcount, n0):
            idx = ivec[j, i1]
            y[idx, i1] = averank
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def rankdata_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a):
    "Ranks n2d array with dtype=int64 along axis=1, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j, k, idx
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
                averank = sumranks / dupcount + 1
                for j in xrange(k - dupcount, k):
                    idx = ivec[i0, j]
                    y[i0, idx] = averank
                sumranks = 0
                dupcount = 0
            old = new    
        sumranks += (n1 - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        for j in xrange(n1 - dupcount, n1):
            idx = ivec[i0, j]
            y[i0, idx] = averank
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def rankdata_1d_float32_axis0(np.ndarray[np.float32_t, ndim=1] a):
    "Ranks n1d array with dtype=float32 along axis=0, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j, k, idx
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
            averank = sumranks / dupcount + 1
            for j in xrange(k - dupcount, k):
                y[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
        old = new    
    sumranks += (n0 - 1)
    dupcount += 1 
    averank = sumranks / dupcount + 1
    for j in xrange(n0 - dupcount, n0):
        y[ivec[j]] = averank
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def rankdata_1d_float64_axis0(np.ndarray[np.float64_t, ndim=1] a):
    "Ranks n1d array with dtype=float64 along axis=0, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j, k, idx
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
            averank = sumranks / dupcount + 1
            for j in xrange(k - dupcount, k):
                y[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
        old = new    
    sumranks += (n0 - 1)
    dupcount += 1 
    averank = sumranks / dupcount + 1
    for j in xrange(n0 - dupcount, n0):
        y[ivec[j]] = averank
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def rankdata_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a):
    "Ranks n2d array with dtype=float32 along axis=0, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j, k, idx
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
                averank = sumranks / dupcount + 1
                for j in xrange(k - dupcount, k):
                    idx = ivec[j, i1]
                    y[idx, i1] = averank
                sumranks = 0
                dupcount = 0
            old = new    
        sumranks += (n0 - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        for j in xrange(n0 - dupcount, n0):
            idx = ivec[j, i1]
            y[idx, i1] = averank
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def rankdata_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a):
    "Ranks n2d array with dtype=float32 along axis=1, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j, k, idx
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
                averank = sumranks / dupcount + 1
                for j in xrange(k - dupcount, k):
                    idx = ivec[i0, j]
                    y[i0, idx] = averank
                sumranks = 0
                dupcount = 0
            old = new    
        sumranks += (n1 - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        for j in xrange(n1 - dupcount, n1):
            idx = ivec[i0, j]
            y[i0, idx] = averank
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def rankdata_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "Ranks n2d array with dtype=float64 along axis=0, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j, k, idx
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
                averank = sumranks / dupcount + 1
                for j in xrange(k - dupcount, k):
                    idx = ivec[j, i1]
                    y[idx, i1] = averank
                sumranks = 0
                dupcount = 0
            old = new    
        sumranks += (n0 - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        for j in xrange(n0 - dupcount, n0):
            idx = ivec[j, i1]
            y[idx, i1] = averank
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def rankdata_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "Ranks n2d array with dtype=float64 along axis=1, dealing with ties." 
    cdef dupcount = 0
    cdef Py_ssize_t j, k, idx
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
                averank = sumranks / dupcount + 1
                for j in xrange(k - dupcount, k):
                    idx = ivec[i0, j]
                    y[i0, idx] = averank
                sumranks = 0
                dupcount = 0
            old = new    
        sumranks += (n1 - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        for j in xrange(n1 - dupcount, n1):
            idx = ivec[i0, j]
            y[i0, idx] = averank
    return y

cdef dict rankdata_dict = {}
rankdata_dict[(1, NPY_int8, 0)] = rankdata_1d_int8_axis0
rankdata_dict[(1, NPY_int32, 0)] = rankdata_1d_int32_axis0
rankdata_dict[(1, NPY_int64, 0)] = rankdata_1d_int64_axis0
rankdata_dict[(2, NPY_int8, 0)] = rankdata_2d_int8_axis0
rankdata_dict[(2, NPY_int8, 1)] = rankdata_2d_int8_axis1
rankdata_dict[(2, NPY_int32, 0)] = rankdata_2d_int32_axis0
rankdata_dict[(2, NPY_int32, 1)] = rankdata_2d_int32_axis1
rankdata_dict[(2, NPY_int64, 0)] = rankdata_2d_int64_axis0
rankdata_dict[(2, NPY_int64, 1)] = rankdata_2d_int64_axis1
rankdata_dict[(1, NPY_float32, 0)] = rankdata_1d_float32_axis0
rankdata_dict[(1, NPY_float64, 0)] = rankdata_1d_float64_axis0
rankdata_dict[(2, NPY_float32, 0)] = rankdata_2d_float32_axis0
rankdata_dict[(2, NPY_float32, 1)] = rankdata_2d_float32_axis1
rankdata_dict[(2, NPY_float64, 0)] = rankdata_2d_float64_axis0
rankdata_dict[(2, NPY_float64, 1)] = rankdata_2d_float64_axis1

def rankdata_slow_axis0(arr):
    "Unaccelerated (slow) rankdata along axis 0."
    return bn.slow.rankdata(arr, axis=0)

def rankdata_slow_axis1(arr):
    "Unaccelerated (slow) rankdata along axis 1."
    return bn.slow.rankdata(arr, axis=1)

def rankdata_slow_axis2(arr):
    "Unaccelerated (slow) rankdata along axis 2."
    return bn.slow.rankdata(arr, axis=2)

def rankdata_slow_axis3(arr):
    "Unaccelerated (slow) rankdata along axis 3."
    return bn.slow.rankdata(arr, axis=3)

def rankdata_slow_axis4(arr):
    "Unaccelerated (slow) rankdata along axis 4."
    return bn.slow.rankdata(arr, axis=4)

def rankdata_slow_axis5(arr):
    "Unaccelerated (slow) rankdata along axis 5."
    return bn.slow.rankdata(arr, axis=5)

def rankdata_slow_axis6(arr):
    "Unaccelerated (slow) rankdata along axis 6."
    return bn.slow.rankdata(arr, axis=6)

def rankdata_slow_axis7(arr):
    "Unaccelerated (slow) rankdata along axis 7."
    return bn.slow.rankdata(arr, axis=7)

def rankdata_slow_axis8(arr):
    "Unaccelerated (slow) rankdata along axis 8."
    return bn.slow.rankdata(arr, axis=8)

def rankdata_slow_axis9(arr):
    "Unaccelerated (slow) rankdata along axis 9."
    return bn.slow.rankdata(arr, axis=9)

def rankdata_slow_axis10(arr):
    "Unaccelerated (slow) rankdata along axis 10."
    return bn.slow.rankdata(arr, axis=10)

def rankdata_slow_axis11(arr):
    "Unaccelerated (slow) rankdata along axis 11."
    return bn.slow.rankdata(arr, axis=11)

def rankdata_slow_axis12(arr):
    "Unaccelerated (slow) rankdata along axis 12."
    return bn.slow.rankdata(arr, axis=12)

def rankdata_slow_axis13(arr):
    "Unaccelerated (slow) rankdata along axis 13."
    return bn.slow.rankdata(arr, axis=13)

def rankdata_slow_axis14(arr):
    "Unaccelerated (slow) rankdata along axis 14."
    return bn.slow.rankdata(arr, axis=14)

def rankdata_slow_axis15(arr):
    "Unaccelerated (slow) rankdata along axis 15."
    return bn.slow.rankdata(arr, axis=15)

def rankdata_slow_axis16(arr):
    "Unaccelerated (slow) rankdata along axis 16."
    return bn.slow.rankdata(arr, axis=16)

def rankdata_slow_axis17(arr):
    "Unaccelerated (slow) rankdata along axis 17."
    return bn.slow.rankdata(arr, axis=17)

def rankdata_slow_axis18(arr):
    "Unaccelerated (slow) rankdata along axis 18."
    return bn.slow.rankdata(arr, axis=18)

def rankdata_slow_axis19(arr):
    "Unaccelerated (slow) rankdata along axis 19."
    return bn.slow.rankdata(arr, axis=19)

def rankdata_slow_axis20(arr):
    "Unaccelerated (slow) rankdata along axis 20."
    return bn.slow.rankdata(arr, axis=20)

def rankdata_slow_axis21(arr):
    "Unaccelerated (slow) rankdata along axis 21."
    return bn.slow.rankdata(arr, axis=21)

def rankdata_slow_axis22(arr):
    "Unaccelerated (slow) rankdata along axis 22."
    return bn.slow.rankdata(arr, axis=22)

def rankdata_slow_axis23(arr):
    "Unaccelerated (slow) rankdata along axis 23."
    return bn.slow.rankdata(arr, axis=23)

def rankdata_slow_axis24(arr):
    "Unaccelerated (slow) rankdata along axis 24."
    return bn.slow.rankdata(arr, axis=24)

def rankdata_slow_axis25(arr):
    "Unaccelerated (slow) rankdata along axis 25."
    return bn.slow.rankdata(arr, axis=25)

def rankdata_slow_axis26(arr):
    "Unaccelerated (slow) rankdata along axis 26."
    return bn.slow.rankdata(arr, axis=26)

def rankdata_slow_axis27(arr):
    "Unaccelerated (slow) rankdata along axis 27."
    return bn.slow.rankdata(arr, axis=27)

def rankdata_slow_axis28(arr):
    "Unaccelerated (slow) rankdata along axis 28."
    return bn.slow.rankdata(arr, axis=28)

def rankdata_slow_axis29(arr):
    "Unaccelerated (slow) rankdata along axis 29."
    return bn.slow.rankdata(arr, axis=29)

def rankdata_slow_axis30(arr):
    "Unaccelerated (slow) rankdata along axis 30."
    return bn.slow.rankdata(arr, axis=30)

def rankdata_slow_axis31(arr):
    "Unaccelerated (slow) rankdata along axis 31."
    return bn.slow.rankdata(arr, axis=31)

def rankdata_slow_axis32(arr):
    "Unaccelerated (slow) rankdata along axis 32."
    return bn.slow.rankdata(arr, axis=32)

def rankdata_slow_axisNone(arr):
    "Unaccelerated (slow) rankdata along axis None."
    return bn.slow.rankdata(arr, axis=None)


cdef dict rankdata_slow_dict = {}
rankdata_slow_dict[0] = rankdata_slow_axis0
rankdata_slow_dict[1] = rankdata_slow_axis1
rankdata_slow_dict[2] = rankdata_slow_axis2
rankdata_slow_dict[3] = rankdata_slow_axis3
rankdata_slow_dict[4] = rankdata_slow_axis4
rankdata_slow_dict[5] = rankdata_slow_axis5
rankdata_slow_dict[6] = rankdata_slow_axis6
rankdata_slow_dict[7] = rankdata_slow_axis7
rankdata_slow_dict[8] = rankdata_slow_axis8
rankdata_slow_dict[9] = rankdata_slow_axis9
rankdata_slow_dict[10] = rankdata_slow_axis10
rankdata_slow_dict[11] = rankdata_slow_axis11
rankdata_slow_dict[12] = rankdata_slow_axis12
rankdata_slow_dict[13] = rankdata_slow_axis13
rankdata_slow_dict[14] = rankdata_slow_axis14
rankdata_slow_dict[15] = rankdata_slow_axis15
rankdata_slow_dict[16] = rankdata_slow_axis16
rankdata_slow_dict[17] = rankdata_slow_axis17
rankdata_slow_dict[18] = rankdata_slow_axis18
rankdata_slow_dict[19] = rankdata_slow_axis19
rankdata_slow_dict[20] = rankdata_slow_axis20
rankdata_slow_dict[21] = rankdata_slow_axis21
rankdata_slow_dict[22] = rankdata_slow_axis22
rankdata_slow_dict[23] = rankdata_slow_axis23
rankdata_slow_dict[24] = rankdata_slow_axis24
rankdata_slow_dict[25] = rankdata_slow_axis25
rankdata_slow_dict[26] = rankdata_slow_axis26
rankdata_slow_dict[27] = rankdata_slow_axis27
rankdata_slow_dict[28] = rankdata_slow_axis28
rankdata_slow_dict[29] = rankdata_slow_axis29
rankdata_slow_dict[30] = rankdata_slow_axis30
rankdata_slow_dict[31] = rankdata_slow_axis31
rankdata_slow_dict[32] = rankdata_slow_axis32
rankdata_slow_dict[None] = rankdata_slow_axisNone