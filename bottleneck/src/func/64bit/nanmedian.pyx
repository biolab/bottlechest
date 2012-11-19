# 64 bit version
"nanmedian auto-generated from template"
# Select smallest k elements code used for inner loop of median method:
# http://projects.scipy.org/numpy/attachment/ticket/1213/quickselect.pyx
# (C) 2009 Sturla Molden
# SciPy license 
#
# From the original C function (code in public domain) in:
#   Fast median search: an ANSI C implementation
#   Nicolas Devillard - ndevilla AT free DOT fr
#   July 1998
# which, in turn, took the algorithm from
#   Wirth, Niklaus
#   Algorithms + data structures = programs, p. 366
#   Englewood Cliffs: Prentice-Hall, 1976
#
# Adapted and expanded for Bottleneck:
# (C) 2010 Keith Goodman

def nanmedian(arr, axis=None):
    """
    Median of array elements along given axis ignoring NaNs.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the median is computed. The default (axis=None) is to
        compute the median of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, except that the specified axis
        has been removed. If `arr` is a 0d array, or if axis is None, a scalar
        is returned. `float64` return values are used for integer inputs. 
    
    See also
    --------
    bottleneck.median: Median along specified axis. 

    Examples
    --------
    >>> a = np.array([[np.nan, 7, 4], [3, 2, 1]])
    >>> a 
    array([[ nan,   7.,   4.],
           [  3.,   2.,   1.]])
    >>> bn.nanmedian(a)
    3.0
    >> bn.nanmedian(a, axis=0)
    array([ 3. ,  4.5,  2.5])
    >> bn.nanmedian(a, axis=1)
    array([ 5.5,  2. ])
    
    """
    func, arr = nanmedian_selector(arr, axis)
    return func(arr)

def nanmedian_selector(arr, axis):
    """
    Return nanmedian function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.nanmedian() is in checking that `axis` is within range, converting `arr`
    into an array (if it is not already an array), and selecting the function
    to use to calculate the mean.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}
        Axis along which the median is to be computed.
    
    Returns
    -------
    func : function
        The nanmedian function that matches the number of dimensions and dtype
        of the input array and the axis along which you wish to find the
        median.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])
    
    Obtain the function needed to determine the median of `arr` along axis=0:

    >>> func, a = bn.func.nanmedian_selector(arr, axis=0)
    >>> func
    <function nanmedian_1d_float64_axis0>
    
    Use the returned function and array to determine the median:

    >>> func(a)
    2.0

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
        func = nanmedian_dict[key]
    except KeyError:
        if (axis < 0) or (axis >= ndim):
            raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = nanmedian_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmedian_1d_float32_axis0(np.ndarray[np.float32_t, ndim=1] a):
    "Median of 1d array with dtype=float32 along axis=0."
    cdef int allnan = 1, flag = 0
    cdef np.npy_intp i = 0, j = 0, l, r, k, n 
    cdef np.float32_t x, tmp, amax, ai
    cdef np.ndarray[np.float32_t, ndim=1] b = PyArray_Copy(a)
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    if n0 == 0:
        return np.float32(NAN)
    j = n0 - 1
    flag = 1
    for i in range(n0):
        if b[i] != b[i]:
            while b[j] != b[j]:
                if j <= 0:
                    break
                j -= 1    
            if i >= j:
                flag = 0
                break
            tmp = b[i]
            b[i] = b[j]
            b[j] = tmp
    n = i + flag
    k = n >> 1
    l = 0
    r = n - 1 
    with nogil:       
        while l < r:
            x = b[k]
            i = l
            j = r
            while 1:
                while b[i] < x: i += 1
                while x < b[j]: j -= 1
                if i <= j:
                    tmp = b[i]
                    b[i] = b[j]
                    b[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
    if n % 2 == 0:        
        amax = MINfloat32
        allnan = 1
        for i in range(k):
            ai = b[i]
            if ai >= amax:
                amax = ai
                allnan = 0
        if allnan == 0:        
            return np.float32(0.5 * (b[k] + amax))
        else:
            return np.float32(b[k])
    else:
        return np.float32(b[k])

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmedian_1d_float64_axis0(np.ndarray[np.float64_t, ndim=1] a):
    "Median of 1d array with dtype=float64 along axis=0."
    cdef int allnan = 1, flag = 0
    cdef np.npy_intp i = 0, j = 0, l, r, k, n 
    cdef np.float64_t x, tmp, amax, ai
    cdef np.ndarray[np.float64_t, ndim=1] b = PyArray_Copy(a)
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    if n0 == 0:
        return np.float64(NAN)
    j = n0 - 1
    flag = 1
    for i in range(n0):
        if b[i] != b[i]:
            while b[j] != b[j]:
                if j <= 0:
                    break
                j -= 1    
            if i >= j:
                flag = 0
                break
            tmp = b[i]
            b[i] = b[j]
            b[j] = tmp
    n = i + flag
    k = n >> 1
    l = 0
    r = n - 1 
    with nogil:       
        while l < r:
            x = b[k]
            i = l
            j = r
            while 1:
                while b[i] < x: i += 1
                while x < b[j]: j -= 1
                if i <= j:
                    tmp = b[i]
                    b[i] = b[j]
                    b[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
    if n % 2 == 0:        
        amax = MINfloat64
        allnan = 1
        for i in range(k):
            ai = b[i]
            if ai >= amax:
                amax = ai
                allnan = 0
        if allnan == 0:        
            return np.float64(0.5 * (b[k] + amax))
        else:
            return np.float64(b[k])
    else:
        return np.float64(b[k])

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmedian_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a):
    "Median of 2d array with dtype=float32 along axis=0."
    cdef int allnan = 1, flag = 0
    cdef np.npy_intp i = 0, j = 0, l, r, k, n 
    cdef np.float32_t x, tmp, amax, ai
    cdef np.ndarray[np.float32_t, ndim=2] b = PyArray_Copy(a)
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float32_t, ndim=1] y = PyArray_EMPTY(1, dims,
		NPY_float32, 0)
    if n0 == 0:
        PyArray_FillWithScalar(y, NAN)
        return y
    for i1 in range(n1):
        j = n0 - 1
        flag = 1
        for i in range(n0):
            if b[i, i1] != b[i, i1]:
                while b[j, i1] != b[j, i1]:
                    if j <= 0:
                        break
                    j -= 1    
                if i >= j:
                    flag = 0
                    break
                tmp = b[i, i1]
                b[i, i1] = b[j, i1]
                b[j, i1] = tmp
        n = i + flag
        k = n >> 1
        l = 0
        r = n - 1
        while l < r:
            x = b[k, i1]
            i = l
            j = r
            while 1:
                while b[i, i1] < x: i += 1
                while x < b[j, i1]: j -= 1
                if i <= j:
                    tmp = b[i, i1]
                    b[i, i1] = b[j, i1]
                    b[j, i1] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n % 2 == 0:        
            amax = MINfloat32
            allnan = 1
            for i in range(k):
                ai = b[i, i1]
                if ai >= amax:
                    amax = ai
                    allnan = 0
            if allnan == 0:        
                y[i1] = 0.5 * (b[k, i1] + amax)
            else:
                y[i1] = b[k, i1]         
        else:
            y[i1] = b[k, i1]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmedian_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a):
    "Median of 2d array with dtype=float32 along axis=1."
    cdef int allnan = 1, flag = 0
    cdef np.npy_intp i = 0, j = 0, l, r, k, n 
    cdef np.float32_t x, tmp, amax, ai
    cdef np.ndarray[np.float32_t, ndim=2] b = PyArray_Copy(a)
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float32_t, ndim=1] y = PyArray_EMPTY(1, dims,
		NPY_float32, 0)
    if n1 == 0:
        PyArray_FillWithScalar(y, NAN)
        return y
    for i0 in range(n0):
        j = n1 - 1
        flag = 1
        for i in range(n1):
            if b[i0, i] != b[i0, i]:
                while b[i0, j] != b[i0, j]:
                    if j <= 0:
                        break
                    j -= 1    
                if i >= j:
                    flag = 0
                    break
                tmp = b[i0, i]
                b[i0, i] = b[i0, j]
                b[i0, j] = tmp
        n = i + flag
        k = n >> 1
        l = 0
        r = n - 1
        while l < r:
            x = b[i0, k]
            i = l
            j = r
            while 1:
                while b[i0, i] < x: i += 1
                while x < b[i0, j]: j -= 1
                if i <= j:
                    tmp = b[i0, i]
                    b[i0, i] = b[i0, j]
                    b[i0, j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n % 2 == 0:        
            amax = MINfloat32
            allnan = 1
            for i in range(k):
                ai = b[i0, i]
                if ai >= amax:
                    amax = ai
                    allnan = 0
            if allnan == 0:        
                y[i0] = 0.5 * (b[i0, k] + amax)
            else:
                y[i0] = b[i0, k]         
        else:
            y[i0] = b[i0, k]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmedian_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "Median of 2d array with dtype=float64 along axis=0."
    cdef int allnan = 1, flag = 0
    cdef np.npy_intp i = 0, j = 0, l, r, k, n 
    cdef np.float64_t x, tmp, amax, ai
    cdef np.ndarray[np.float64_t, ndim=2] b = PyArray_Copy(a)
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
		NPY_float64, 0)
    if n0 == 0:
        PyArray_FillWithScalar(y, NAN)
        return y
    for i1 in range(n1):
        j = n0 - 1
        flag = 1
        for i in range(n0):
            if b[i, i1] != b[i, i1]:
                while b[j, i1] != b[j, i1]:
                    if j <= 0:
                        break
                    j -= 1    
                if i >= j:
                    flag = 0
                    break
                tmp = b[i, i1]
                b[i, i1] = b[j, i1]
                b[j, i1] = tmp
        n = i + flag
        k = n >> 1
        l = 0
        r = n - 1
        while l < r:
            x = b[k, i1]
            i = l
            j = r
            while 1:
                while b[i, i1] < x: i += 1
                while x < b[j, i1]: j -= 1
                if i <= j:
                    tmp = b[i, i1]
                    b[i, i1] = b[j, i1]
                    b[j, i1] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n % 2 == 0:        
            amax = MINfloat64
            allnan = 1
            for i in range(k):
                ai = b[i, i1]
                if ai >= amax:
                    amax = ai
                    allnan = 0
            if allnan == 0:        
                y[i1] = 0.5 * (b[k, i1] + amax)
            else:
                y[i1] = b[k, i1]         
        else:
            y[i1] = b[k, i1]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmedian_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "Median of 2d array with dtype=float64 along axis=1."
    cdef int allnan = 1, flag = 0
    cdef np.npy_intp i = 0, j = 0, l, r, k, n 
    cdef np.float64_t x, tmp, amax, ai
    cdef np.ndarray[np.float64_t, ndim=2] b = PyArray_Copy(a)
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
		NPY_float64, 0)
    if n1 == 0:
        PyArray_FillWithScalar(y, NAN)
        return y
    for i0 in range(n0):
        j = n1 - 1
        flag = 1
        for i in range(n1):
            if b[i0, i] != b[i0, i]:
                while b[i0, j] != b[i0, j]:
                    if j <= 0:
                        break
                    j -= 1    
                if i >= j:
                    flag = 0
                    break
                tmp = b[i0, i]
                b[i0, i] = b[i0, j]
                b[i0, j] = tmp
        n = i + flag
        k = n >> 1
        l = 0
        r = n - 1
        while l < r:
            x = b[i0, k]
            i = l
            j = r
            while 1:
                while b[i0, i] < x: i += 1
                while x < b[i0, j]: j -= 1
                if i <= j:
                    tmp = b[i0, i]
                    b[i0, i] = b[i0, j]
                    b[i0, j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n % 2 == 0:        
            amax = MINfloat64
            allnan = 1
            for i in range(k):
                ai = b[i0, i]
                if ai >= amax:
                    amax = ai
                    allnan = 0
            if allnan == 0:        
                y[i0] = 0.5 * (b[i0, k] + amax)
            else:
                y[i0] = b[i0, k]         
        else:
            y[i0] = b[i0, k]         
    return y

cdef dict nanmedian_dict = {}
nanmedian_dict[(1, NPY_int32, 0)] = median_1d_int32_axis0
nanmedian_dict[(1, NPY_int64, 0)] = median_1d_int64_axis0
nanmedian_dict[(2, NPY_int32, 0)] = median_2d_int32_axis0
nanmedian_dict[(2, NPY_int32, 1)] = median_2d_int32_axis1
nanmedian_dict[(2, NPY_int64, 0)] = median_2d_int64_axis0
nanmedian_dict[(2, NPY_int64, 1)] = median_2d_int64_axis1
nanmedian_dict[(1, NPY_float32, 0)] = nanmedian_1d_float32_axis0
nanmedian_dict[(1, NPY_float64, 0)] = nanmedian_1d_float64_axis0
nanmedian_dict[(2, NPY_float32, 0)] = nanmedian_2d_float32_axis0
nanmedian_dict[(2, NPY_float32, 1)] = nanmedian_2d_float32_axis1
nanmedian_dict[(2, NPY_float64, 0)] = nanmedian_2d_float64_axis0
nanmedian_dict[(2, NPY_float64, 1)] = nanmedian_2d_float64_axis1

def nanmedian_slow_axis0(arr):
    "Unaccelerated (slow) nanmedian along axis 0."
    return bn.slow.nanmedian(arr, axis=0)

def nanmedian_slow_axis1(arr):
    "Unaccelerated (slow) nanmedian along axis 1."
    return bn.slow.nanmedian(arr, axis=1)

def nanmedian_slow_axis2(arr):
    "Unaccelerated (slow) nanmedian along axis 2."
    return bn.slow.nanmedian(arr, axis=2)

def nanmedian_slow_axis3(arr):
    "Unaccelerated (slow) nanmedian along axis 3."
    return bn.slow.nanmedian(arr, axis=3)

def nanmedian_slow_axis4(arr):
    "Unaccelerated (slow) nanmedian along axis 4."
    return bn.slow.nanmedian(arr, axis=4)

def nanmedian_slow_axis5(arr):
    "Unaccelerated (slow) nanmedian along axis 5."
    return bn.slow.nanmedian(arr, axis=5)

def nanmedian_slow_axis6(arr):
    "Unaccelerated (slow) nanmedian along axis 6."
    return bn.slow.nanmedian(arr, axis=6)

def nanmedian_slow_axis7(arr):
    "Unaccelerated (slow) nanmedian along axis 7."
    return bn.slow.nanmedian(arr, axis=7)

def nanmedian_slow_axis8(arr):
    "Unaccelerated (slow) nanmedian along axis 8."
    return bn.slow.nanmedian(arr, axis=8)

def nanmedian_slow_axis9(arr):
    "Unaccelerated (slow) nanmedian along axis 9."
    return bn.slow.nanmedian(arr, axis=9)

def nanmedian_slow_axis10(arr):
    "Unaccelerated (slow) nanmedian along axis 10."
    return bn.slow.nanmedian(arr, axis=10)

def nanmedian_slow_axis11(arr):
    "Unaccelerated (slow) nanmedian along axis 11."
    return bn.slow.nanmedian(arr, axis=11)

def nanmedian_slow_axis12(arr):
    "Unaccelerated (slow) nanmedian along axis 12."
    return bn.slow.nanmedian(arr, axis=12)

def nanmedian_slow_axis13(arr):
    "Unaccelerated (slow) nanmedian along axis 13."
    return bn.slow.nanmedian(arr, axis=13)

def nanmedian_slow_axis14(arr):
    "Unaccelerated (slow) nanmedian along axis 14."
    return bn.slow.nanmedian(arr, axis=14)

def nanmedian_slow_axis15(arr):
    "Unaccelerated (slow) nanmedian along axis 15."
    return bn.slow.nanmedian(arr, axis=15)

def nanmedian_slow_axis16(arr):
    "Unaccelerated (slow) nanmedian along axis 16."
    return bn.slow.nanmedian(arr, axis=16)

def nanmedian_slow_axis17(arr):
    "Unaccelerated (slow) nanmedian along axis 17."
    return bn.slow.nanmedian(arr, axis=17)

def nanmedian_slow_axis18(arr):
    "Unaccelerated (slow) nanmedian along axis 18."
    return bn.slow.nanmedian(arr, axis=18)

def nanmedian_slow_axis19(arr):
    "Unaccelerated (slow) nanmedian along axis 19."
    return bn.slow.nanmedian(arr, axis=19)

def nanmedian_slow_axis20(arr):
    "Unaccelerated (slow) nanmedian along axis 20."
    return bn.slow.nanmedian(arr, axis=20)

def nanmedian_slow_axis21(arr):
    "Unaccelerated (slow) nanmedian along axis 21."
    return bn.slow.nanmedian(arr, axis=21)

def nanmedian_slow_axis22(arr):
    "Unaccelerated (slow) nanmedian along axis 22."
    return bn.slow.nanmedian(arr, axis=22)

def nanmedian_slow_axis23(arr):
    "Unaccelerated (slow) nanmedian along axis 23."
    return bn.slow.nanmedian(arr, axis=23)

def nanmedian_slow_axis24(arr):
    "Unaccelerated (slow) nanmedian along axis 24."
    return bn.slow.nanmedian(arr, axis=24)

def nanmedian_slow_axis25(arr):
    "Unaccelerated (slow) nanmedian along axis 25."
    return bn.slow.nanmedian(arr, axis=25)

def nanmedian_slow_axis26(arr):
    "Unaccelerated (slow) nanmedian along axis 26."
    return bn.slow.nanmedian(arr, axis=26)

def nanmedian_slow_axis27(arr):
    "Unaccelerated (slow) nanmedian along axis 27."
    return bn.slow.nanmedian(arr, axis=27)

def nanmedian_slow_axis28(arr):
    "Unaccelerated (slow) nanmedian along axis 28."
    return bn.slow.nanmedian(arr, axis=28)

def nanmedian_slow_axis29(arr):
    "Unaccelerated (slow) nanmedian along axis 29."
    return bn.slow.nanmedian(arr, axis=29)

def nanmedian_slow_axis30(arr):
    "Unaccelerated (slow) nanmedian along axis 30."
    return bn.slow.nanmedian(arr, axis=30)

def nanmedian_slow_axis31(arr):
    "Unaccelerated (slow) nanmedian along axis 31."
    return bn.slow.nanmedian(arr, axis=31)

def nanmedian_slow_axis32(arr):
    "Unaccelerated (slow) nanmedian along axis 32."
    return bn.slow.nanmedian(arr, axis=32)

def nanmedian_slow_axisNone(arr):
    "Unaccelerated (slow) nanmedian along axis None."
    return bn.slow.nanmedian(arr, axis=None)


cdef dict nanmedian_slow_dict = {}
nanmedian_slow_dict[0] = nanmedian_slow_axis0
nanmedian_slow_dict[1] = nanmedian_slow_axis1
nanmedian_slow_dict[2] = nanmedian_slow_axis2
nanmedian_slow_dict[3] = nanmedian_slow_axis3
nanmedian_slow_dict[4] = nanmedian_slow_axis4
nanmedian_slow_dict[5] = nanmedian_slow_axis5
nanmedian_slow_dict[6] = nanmedian_slow_axis6
nanmedian_slow_dict[7] = nanmedian_slow_axis7
nanmedian_slow_dict[8] = nanmedian_slow_axis8
nanmedian_slow_dict[9] = nanmedian_slow_axis9
nanmedian_slow_dict[10] = nanmedian_slow_axis10
nanmedian_slow_dict[11] = nanmedian_slow_axis11
nanmedian_slow_dict[12] = nanmedian_slow_axis12
nanmedian_slow_dict[13] = nanmedian_slow_axis13
nanmedian_slow_dict[14] = nanmedian_slow_axis14
nanmedian_slow_dict[15] = nanmedian_slow_axis15
nanmedian_slow_dict[16] = nanmedian_slow_axis16
nanmedian_slow_dict[17] = nanmedian_slow_axis17
nanmedian_slow_dict[18] = nanmedian_slow_axis18
nanmedian_slow_dict[19] = nanmedian_slow_axis19
nanmedian_slow_dict[20] = nanmedian_slow_axis20
nanmedian_slow_dict[21] = nanmedian_slow_axis21
nanmedian_slow_dict[22] = nanmedian_slow_axis22
nanmedian_slow_dict[23] = nanmedian_slow_axis23
nanmedian_slow_dict[24] = nanmedian_slow_axis24
nanmedian_slow_dict[25] = nanmedian_slow_axis25
nanmedian_slow_dict[26] = nanmedian_slow_axis26
nanmedian_slow_dict[27] = nanmedian_slow_axis27
nanmedian_slow_dict[28] = nanmedian_slow_axis28
nanmedian_slow_dict[29] = nanmedian_slow_axis29
nanmedian_slow_dict[30] = nanmedian_slow_axis30
nanmedian_slow_dict[31] = nanmedian_slow_axis31
nanmedian_slow_dict[32] = nanmedian_slow_axis32
nanmedian_slow_dict[None] = nanmedian_slow_axisNone