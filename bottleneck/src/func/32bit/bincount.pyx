# 32 bit version
"nanequal auto-generated from template"

def bincount(arr, max_val, weights=None, mask=None):
    """
    Count number of occurrences of each value in array.

    The number of bins is max_val+1. Each bin gives the number of occurrences
    of its index value in `x`. If `weights` is specified the input array is
    weighted by it, i.e. if a value ``n`` is found at position ``i``,
    ``out[n] += weight[i]`` instead of ``out[n] += 1``. A subset of columns
    can be selected by additional argument `mask`.

    Function differs from numpy in that it can handle float arrays; values
    need to be close to integers (allowed error is 1e-6). The function also
    returns a count of NaN values. The maximal value must be given as an
    argument.

    Unlike numpy.bincount, this function also handles 2d arrays.

    Parameters
    ----------
    x : array_like, 1 or 2 dimensions, nonnegative elements
        Input array.
    max_val : int
        The maximal value in the array
    weights : array_like, optional
        Weights, array of the same length as `x`.
    mask: array_like, of type char (interpreted as bool)
        Selects the columns

    Returns
    -------
    out : ndarray of ints, 1- or 2-dimensional
        The result of binning the input array.
    nans: the number of NaNs; a scalar or a 1-d vector of length x.shape[1]

    Raises
    ------
    ValueError
        If the input is not 1- or 2-dimensional, or contains elements that are
        not close enough to integers, negative or grater than max_val, or if the
        length of the weight vector does not match the length of the array

    """
    func, a, weights, mask = bincount_selector(arr, max_val, weights, mask)
    return func(a, max_val, weights, mask)




def bincount_selector(arr, max_val, weights, mask):
    cdef int dtype
    cdef tuple key

    if sp.issparse(arr):
        a = arr
        dtype = PyArray_TYPE(arr.data)
        ndim = 0
        key = (0, dtype, None)
    else:
        if type(arr) is np.ndarray:
            a = arr
        else:
            a = np.array(arr, copy=False)
        dtype = PyArray_TYPE(arr)
        ndim = PyArray_NDIM(a)
        key = (ndim, dtype, None)

    if weights is not None and (
            type(weights) is not np.ndarray or
            weights.dtype is not np.float):
        weights = np.array(weights, copy=False, dtype=np.float)
    if type(mask) is np.ndarray:
        if mask.dtype is not np.int8:
            if mask.dtype.itemsize == 1:
                mask = np.frombuffer(mask, dtype=np.int8)
            else:
                mask = np.array(mask, copy=False, dtype=np.int8)
    elif mask is not None:
        mask = np.array(mask, copy=False, dtype=np.int8)

    try:
        func = bincount_dict[key]
        return func, a, weights, mask
    except KeyError:
        pass

    try:
        func = bincount_slow_dict[None]
    except KeyError:
        tup = (str(ndim), str(a.dtype))
        raise TypeError("Unsupported ndim/dtype (%s/%s)." % tup)
    return func, a, weights, mask

@cython.boundscheck(False)
@cython.wraparound(False)
def bincount_1d_int8_axisNone(np.ndarray[np.int8_t, ndim=1] a,
                              int max_val,
                              np.ndarray[np.float_t, ndim=1] w = None,
                              np.ndarray[np.int8_t, ndim=1] mask = None):
    '''Bincount that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    if w is not None and len(w) != n0:
        raise ValueError("invalid length of the weight vector")

    cdef:
        np.npy_intp *dims = [max_val+1]
        np.ndarray[np.float64_t, ndim=1] y = PyArray_ZEROS(1, dims, NPY_float64, 0)
        int ai
    if mask is not None and not mask[0]:
        return y, 0.0
    if w is None:
        for i0 in range(n0):
            ai = a[i0]
            if ai < 0:
                raise ValueError("negative value in bincount")
            if ai > max_val:
                raise ValueError("value %i is greater than max_val (%i)"
                                 % (ai, max_val))
            y[ai] += 1
    else:
        for i0 in range(n0):
            ai = a[i0]
            if ai < 0:
                raise ValueError("negative value in bincount")
            if ai > max_val:
                raise ValueError("value %i is greater than max_val (%i)"
                                 % (ai, max_val))
            y[ai] += w[i0]
    return y, 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
def bincount_1d_int32_axisNone(np.ndarray[np.int32_t, ndim=1] a,
                              int max_val,
                              np.ndarray[np.float_t, ndim=1] w = None,
                              np.ndarray[np.int8_t, ndim=1] mask = None):
    '''Bincount that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    if w is not None and len(w) != n0:
        raise ValueError("invalid length of the weight vector")

    cdef:
        np.npy_intp *dims = [max_val+1]
        np.ndarray[np.float64_t, ndim=1] y = PyArray_ZEROS(1, dims, NPY_float64, 0)
        int ai
    if mask is not None and not mask[0]:
        return y, 0.0
    if w is None:
        for i0 in range(n0):
            ai = a[i0]
            if ai < 0:
                raise ValueError("negative value in bincount")
            if ai > max_val:
                raise ValueError("value %i is greater than max_val (%i)"
                                 % (ai, max_val))
            y[ai] += 1
    else:
        for i0 in range(n0):
            ai = a[i0]
            if ai < 0:
                raise ValueError("negative value in bincount")
            if ai > max_val:
                raise ValueError("value %i is greater than max_val (%i)"
                                 % (ai, max_val))
            y[ai] += w[i0]
    return y, 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
def bincount_1d_int64_axisNone(np.ndarray[np.int64_t, ndim=1] a,
                              int max_val,
                              np.ndarray[np.float_t, ndim=1] w = None,
                              np.ndarray[np.int8_t, ndim=1] mask = None):
    '''Bincount that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    if w is not None and len(w) != n0:
        raise ValueError("invalid length of the weight vector")

    cdef:
        np.npy_intp *dims = [max_val+1]
        np.ndarray[np.float64_t, ndim=1] y = PyArray_ZEROS(1, dims, NPY_float64, 0)
        int ai
    if mask is not None and not mask[0]:
        return y, 0.0
    if w is None:
        for i0 in range(n0):
            ai = a[i0]
            if ai < 0:
                raise ValueError("negative value in bincount")
            if ai > max_val:
                raise ValueError("value %i is greater than max_val (%i)"
                                 % (ai, max_val))
            y[ai] += 1
    else:
        for i0 in range(n0):
            ai = a[i0]
            if ai < 0:
                raise ValueError("negative value in bincount")
            if ai > max_val:
                raise ValueError("value %i is greater than max_val (%i)"
                                 % (ai, max_val))
            y[ai] += w[i0]
    return y, 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
def bincount_2d_int8_axisNone(np.ndarray[np.int8_t, ndim=2] a,
                              int max_val,
                              np.ndarray[np.float_t, ndim=1] w = None,
                              np.ndarray[np.int8_t, ndim=1] mask = None):
    '''Bincount that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    if w is not None and len(w) != len(a):
        raise ValueError("invalid length of the weight vector")

    cdef:
       np.npy_intp *dims = [n1, max_val+1]
       np.ndarray[np.float_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)
       np.npy_intp *nandims = [n1]
       np.ndarray[np.float64_t, ndim=1] nans = PyArray_ZEROS(1, nandims, NPY_float64, 0)
       int ai
       float wt
    if mask is None:
        for i0 in range(n0):
            wt = 1 if w is None else w[i0]
            for i1 in range(n1):
                ai = a[i0, i1]
                if ai < 0:
                    raise ValueError("negative value in bincount")
                if ai > max_val:
                    raise ValueError("value %i is greater than max_val (%i)"
                                     % (ai, max_val))
                y[i1, ai] += wt
    elif w is None:
        for i1 in range(n1):
            if mask[i1]:
                for i0 in range(n0):
                    ai = a[i0, i1]
                    if ai < 0:
                        raise ValueError("negative value in bincount")
                    if ai > max_val:
                        raise ValueError("value %i is greater than max_val (%i)"
                                         % (ai, max_val))
                    y[i1, ai] += 1
    else:
        for i1 in range(n1):
            if mask[i1]:
                for i0 in range(n0):
                    ai = a[i0, i1]
                    if ai < 0:
                        raise ValueError("negative value in bincount")
                    if ai > max_val:
                        raise ValueError("value %i is greater than max_val (%i)"
                                         % (ai, max_val))
                    y[i1, ai] += w[i0]
    return y, nans

@cython.boundscheck(False)
@cython.wraparound(False)
def bincount_2d_int32_axisNone(np.ndarray[np.int32_t, ndim=2] a,
                              int max_val,
                              np.ndarray[np.float_t, ndim=1] w = None,
                              np.ndarray[np.int8_t, ndim=1] mask = None):
    '''Bincount that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    if w is not None and len(w) != len(a):
        raise ValueError("invalid length of the weight vector")

    cdef:
       np.npy_intp *dims = [n1, max_val+1]
       np.ndarray[np.float_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)
       np.npy_intp *nandims = [n1]
       np.ndarray[np.float64_t, ndim=1] nans = PyArray_ZEROS(1, nandims, NPY_float64, 0)
       int ai
       float wt
    if mask is None:
        for i0 in range(n0):
            wt = 1 if w is None else w[i0]
            for i1 in range(n1):
                ai = a[i0, i1]
                if ai < 0:
                    raise ValueError("negative value in bincount")
                if ai > max_val:
                    raise ValueError("value %i is greater than max_val (%i)"
                                     % (ai, max_val))
                y[i1, ai] += wt
    elif w is None:
        for i1 in range(n1):
            if mask[i1]:
                for i0 in range(n0):
                    ai = a[i0, i1]
                    if ai < 0:
                        raise ValueError("negative value in bincount")
                    if ai > max_val:
                        raise ValueError("value %i is greater than max_val (%i)"
                                         % (ai, max_val))
                    y[i1, ai] += 1
    else:
        for i1 in range(n1):
            if mask[i1]:
                for i0 in range(n0):
                    ai = a[i0, i1]
                    if ai < 0:
                        raise ValueError("negative value in bincount")
                    if ai > max_val:
                        raise ValueError("value %i is greater than max_val (%i)"
                                         % (ai, max_val))
                    y[i1, ai] += w[i0]
    return y, nans

@cython.boundscheck(False)
@cython.wraparound(False)
def bincount_2d_int64_axisNone(np.ndarray[np.int64_t, ndim=2] a,
                              int max_val,
                              np.ndarray[np.float_t, ndim=1] w = None,
                              np.ndarray[np.int8_t, ndim=1] mask = None):
    '''Bincount that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    if w is not None and len(w) != len(a):
        raise ValueError("invalid length of the weight vector")

    cdef:
       np.npy_intp *dims = [n1, max_val+1]
       np.ndarray[np.float_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)
       np.npy_intp *nandims = [n1]
       np.ndarray[np.float64_t, ndim=1] nans = PyArray_ZEROS(1, nandims, NPY_float64, 0)
       int ai
       float wt
    if mask is None:
        for i0 in range(n0):
            wt = 1 if w is None else w[i0]
            for i1 in range(n1):
                ai = a[i0, i1]
                if ai < 0:
                    raise ValueError("negative value in bincount")
                if ai > max_val:
                    raise ValueError("value %i is greater than max_val (%i)"
                                     % (ai, max_val))
                y[i1, ai] += wt
    elif w is None:
        for i1 in range(n1):
            if mask[i1]:
                for i0 in range(n0):
                    ai = a[i0, i1]
                    if ai < 0:
                        raise ValueError("negative value in bincount")
                    if ai > max_val:
                        raise ValueError("value %i is greater than max_val (%i)"
                                         % (ai, max_val))
                    y[i1, ai] += 1
    else:
        for i1 in range(n1):
            if mask[i1]:
                for i0 in range(n0):
                    ai = a[i0, i1]
                    if ai < 0:
                        raise ValueError("negative value in bincount")
                    if ai > max_val:
                        raise ValueError("value %i is greater than max_val (%i)"
                                         % (ai, max_val))
                    y[i1, ai] += w[i0]
    return y, nans

@cython.boundscheck(False)
@cython.wraparound(False)
def bincount_0d_int8_axisNone(object a,
           int max_val,
           np.ndarray[np.float_t, ndim=1] w = None,
           np.ndarray[np.int8_t, ndim=1] mask = None):
    '''Bincount that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''

    cdef:
        Py_ssize_t n_rows = a.shape[0]
        Py_ssize_t n_cols = a.shape[1]

    if w is not None and len(w) != n_rows:
        raise ValueError("invalid length of the weight vector")

    cdef:
        np.npy_intp *dims = [n_cols, max_val+1]
        np.ndarray[np.float64_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)
        np.ndarray[np.float64_t, ndim=1] nans = PyArray_ZEROS(1, dims, NPY_float64, 0)
        float wt

        np.ndarray[np.int8_t, ndim=1] data = a.data
        np.ndarray[int, ndim=1] indices = a.indices
        np.ndarray[int, ndim=1] indptr = a.indptr
        int ri, i, ci
    for ri in range(a.shape[0]):
        wt = 1 if w is None else w[ri]
        for i in range(indptr[ri], indptr[ri + 1]):
            ci = indices[i]
            if mask is None or mask[ci]:
                y[ci, data[i]] += wt
    return y, nans



@cython.boundscheck(False)
@cython.wraparound(False)
def bincount_0d_int32_axisNone(object a,
           int max_val,
           np.ndarray[np.float_t, ndim=1] w = None,
           np.ndarray[np.int8_t, ndim=1] mask = None):
    '''Bincount that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''

    cdef:
        Py_ssize_t n_rows = a.shape[0]
        Py_ssize_t n_cols = a.shape[1]

    if w is not None and len(w) != n_rows:
        raise ValueError("invalid length of the weight vector")

    cdef:
        np.npy_intp *dims = [n_cols, max_val+1]
        np.ndarray[np.float64_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)
        np.ndarray[np.float64_t, ndim=1] nans = PyArray_ZEROS(1, dims, NPY_float64, 0)
        float wt

        np.ndarray[np.int32_t, ndim=1] data = a.data
        np.ndarray[int, ndim=1] indices = a.indices
        np.ndarray[int, ndim=1] indptr = a.indptr
        int ri, i, ci
    for ri in range(a.shape[0]):
        wt = 1 if w is None else w[ri]
        for i in range(indptr[ri], indptr[ri + 1]):
            ci = indices[i]
            if mask is None or mask[ci]:
                y[ci, data[i]] += wt
    return y, nans



@cython.boundscheck(False)
@cython.wraparound(False)
def bincount_0d_int64_axisNone(object a,
           int max_val,
           np.ndarray[np.float_t, ndim=1] w = None,
           np.ndarray[np.int8_t, ndim=1] mask = None):
    '''Bincount that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''

    cdef:
        Py_ssize_t n_rows = a.shape[0]
        Py_ssize_t n_cols = a.shape[1]

    if w is not None and len(w) != n_rows:
        raise ValueError("invalid length of the weight vector")

    cdef:
        np.npy_intp *dims = [n_cols, max_val+1]
        np.ndarray[np.float64_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)
        np.ndarray[np.float64_t, ndim=1] nans = PyArray_ZEROS(1, dims, NPY_float64, 0)
        float wt

        np.ndarray[np.int64_t, ndim=1] data = a.data
        np.ndarray[int, ndim=1] indices = a.indices
        np.ndarray[int, ndim=1] indptr = a.indptr
        int ri, i, ci
    for ri in range(a.shape[0]):
        wt = 1 if w is None else w[ri]
        for i in range(indptr[ri], indptr[ri + 1]):
            ci = indices[i]
            if mask is None or mask[ci]:
                y[ci, data[i]] += wt
    return y, nans



@cython.boundscheck(False)
@cython.wraparound(False)
def bincount_1d_float32_axisNone(np.ndarray[np.float32_t, ndim=1] a,
                              int max_val,
                              np.ndarray[np.float_t, ndim=1] w = None,
                              np.ndarray[np.int8_t, ndim=1] mask = None):
    '''Bincount that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    if w is not None and len(w) != n0:
        raise ValueError("invalid length of the weight vector")

    cdef:
        np.npy_intp *dims = [max_val+1]
        np.ndarray[np.float64_t, ndim=1] y = PyArray_ZEROS(1, dims, NPY_float64, 0)
        float nans = 0
        int ain
        float ai, wt
    if mask is not None and not mask[0]:
        return y, nans
    if w is None:
        for i0 in range(n0):
            ai = a[i0]
            if ai != ai:
                nans += 1
                continue
            if ai < -1e-6:
                raise ValueError("negative value in bincount")
            ain = int(ai + 0.1)
            if not -1e-6 < ain - ai < 1e-6:
                raise ValueError("%f is not an integer value" % ai)
            if ain > max_val:
                raise ValueError("value %i is greater than max_val (%i)" %
                                 (ain, max_val))
            y[ain] += 1
    else:
        for i0 in range(n0):
            wt = w[i0]
            ai = a[i0]
            if ai != ai:
                nans += wt
                continue
            if ai < -1e-6:
                raise ValueError("negative value in bincount")
            ain = int(ai + 0.1)
            if not -1e-6 < ain - ai < 1e-6:
                raise ValueError("%f is not an integer value" % ai)
            if ain > max_val:
                raise ValueError("value %i is greater than max_val (%i)" %
                                 (ain, max_val))
            y[ain] += wt
    return y, nans

@cython.boundscheck(False)
@cython.wraparound(False)
def bincount_1d_float64_axisNone(np.ndarray[np.float64_t, ndim=1] a,
                              int max_val,
                              np.ndarray[np.float_t, ndim=1] w = None,
                              np.ndarray[np.int8_t, ndim=1] mask = None):
    '''Bincount that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    if w is not None and len(w) != n0:
        raise ValueError("invalid length of the weight vector")

    cdef:
        np.npy_intp *dims = [max_val+1]
        np.ndarray[np.float64_t, ndim=1] y = PyArray_ZEROS(1, dims, NPY_float64, 0)
        float nans = 0
        int ain
        float ai, wt
    if mask is not None and not mask[0]:
        return y, nans
    if w is None:
        for i0 in range(n0):
            ai = a[i0]
            if ai != ai:
                nans += 1
                continue
            if ai < -1e-6:
                raise ValueError("negative value in bincount")
            ain = int(ai + 0.1)
            if not -1e-6 < ain - ai < 1e-6:
                raise ValueError("%f is not an integer value" % ai)
            if ain > max_val:
                raise ValueError("value %i is greater than max_val (%i)" %
                                 (ain, max_val))
            y[ain] += 1
    else:
        for i0 in range(n0):
            wt = w[i0]
            ai = a[i0]
            if ai != ai:
                nans += wt
                continue
            if ai < -1e-6:
                raise ValueError("negative value in bincount")
            ain = int(ai + 0.1)
            if not -1e-6 < ain - ai < 1e-6:
                raise ValueError("%f is not an integer value" % ai)
            if ain > max_val:
                raise ValueError("value %i is greater than max_val (%i)" %
                                 (ain, max_val))
            y[ain] += wt
    return y, nans

@cython.boundscheck(False)
@cython.wraparound(False)
def bincount_2d_float32_axisNone(np.ndarray[np.float32_t, ndim=2] a,
                              int max_val,
                              np.ndarray[np.float_t, ndim=1] w = None,
                              np.ndarray[np.int8_t, ndim=1] mask = None):
    '''Bincount that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    if w is not None and len(w) != len(a):
        raise ValueError("invalid length of the weight vector")

    cdef:
       np.npy_intp *dims = [n1, max_val+1]
       np.ndarray[np.float64_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)
       np.npy_intp *nandims = [n1]
       np.ndarray[np.float_t, ndim=1] nans = PyArray_ZEROS(1, nandims, NPY_float64, 0)
       int ain
       float ai, wt
    if mask is None:
        for i0 in range(n0):
            wt = 1 if w is None else w[i0]
            for i1 in range(n1):
                ai = a[i0, i1]
                if ai != ai:
                    nans[i1] += wt
                    continue
                if ai < -1e-6:
                    raise ValueError("negative value in bincount")
                ain = int(ai + 0.1)
                if not -1e-6 < ain - ai < 1e-6:
                    raise ValueError("%f is not an integer value" % ai)
                if ain > max_val:
                    raise ValueError("value %i is greater than max_val (%i)" %
                                     (ain, max_val))
                y[i1, ain] += wt
    elif w is None:
        for i1 in range(n1):
            if mask[i1]:
                for i0 in range(n0):
                    ai = a[i0, i1]
                    if ai != ai:
                        nans[i1] += 1
                        continue
                    if ai < -1e-6:
                        raise ValueError("negative value in bincount")
                    ain = int(ai + 0.1)
                    if not -1e-6 < ain - ai < 1e-6:
                        raise ValueError("%f is not an integer value" % ai)
                    if ain > max_val:
                        raise ValueError("value %i is greater than max_val (%i)" %
                                         (ain, max_val))
                    y[i1, ain] += 1
    else:
        for i1 in range(n1):
            if mask[i1]:
                for i0 in range(n0):
                    wt = w[i0]
                    ai = a[i0, i1]
                    if ai != ai:
                        nans[i1] += wt
                        continue
                    if ai < -1e-6:
                        raise ValueError("negative value in bincount")
                    ain = int(ai + 0.1)
                    if not -1e-6 < ain - ai < 1e-6:
                        raise ValueError("%f is not an integer value" % ai)
                    if ain > max_val:
                        raise ValueError("value %i is greater than max_val (%i)" %
                                         (ain, max_val))
                    y[i1, ain] += wt
    return y, nans

@cython.boundscheck(False)
@cython.wraparound(False)
def bincount_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a,
                              int max_val,
                              np.ndarray[np.float_t, ndim=1] w = None,
                              np.ndarray[np.int8_t, ndim=1] mask = None):
    '''Bincount that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    if w is not None and len(w) != len(a):
        raise ValueError("invalid length of the weight vector")

    cdef:
       np.npy_intp *dims = [n1, max_val+1]
       np.ndarray[np.float64_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)
       np.npy_intp *nandims = [n1]
       np.ndarray[np.float_t, ndim=1] nans = PyArray_ZEROS(1, nandims, NPY_float64, 0)
       int ain
       float ai, wt
    if mask is None:
        for i0 in range(n0):
            wt = 1 if w is None else w[i0]
            for i1 in range(n1):
                ai = a[i0, i1]
                if ai != ai:
                    nans[i1] += wt
                    continue
                if ai < -1e-6:
                    raise ValueError("negative value in bincount")
                ain = int(ai + 0.1)
                if not -1e-6 < ain - ai < 1e-6:
                    raise ValueError("%f is not an integer value" % ai)
                if ain > max_val:
                    raise ValueError("value %i is greater than max_val (%i)" %
                                     (ain, max_val))
                y[i1, ain] += wt
    elif w is None:
        for i1 in range(n1):
            if mask[i1]:
                for i0 in range(n0):
                    ai = a[i0, i1]
                    if ai != ai:
                        nans[i1] += 1
                        continue
                    if ai < -1e-6:
                        raise ValueError("negative value in bincount")
                    ain = int(ai + 0.1)
                    if not -1e-6 < ain - ai < 1e-6:
                        raise ValueError("%f is not an integer value" % ai)
                    if ain > max_val:
                        raise ValueError("value %i is greater than max_val (%i)" %
                                         (ain, max_val))
                    y[i1, ain] += 1
    else:
        for i1 in range(n1):
            if mask[i1]:
                for i0 in range(n0):
                    wt = w[i0]
                    ai = a[i0, i1]
                    if ai != ai:
                        nans[i1] += wt
                        continue
                    if ai < -1e-6:
                        raise ValueError("negative value in bincount")
                    ain = int(ai + 0.1)
                    if not -1e-6 < ain - ai < 1e-6:
                        raise ValueError("%f is not an integer value" % ai)
                    if ain > max_val:
                        raise ValueError("value %i is greater than max_val (%i)" %
                                         (ain, max_val))
                    y[i1, ain] += wt
    return y, nans

@cython.boundscheck(False)
@cython.wraparound(False)
def bincount_0d_float32_axisNone(object a,
           int max_val,
           np.ndarray[np.float_t, ndim=1] w = None,
           np.ndarray[np.int8_t, ndim=1] mask = None):
    '''Bincount that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''

    cdef:
        Py_ssize_t n_rows = a.shape[0]
        Py_ssize_t n_cols = a.shape[1]

    if w is not None and len(w) != n_rows:
        raise ValueError("invalid length of the weight vector")

    cdef:
        np.npy_intp *dims = [n_cols, max_val+1]
        np.ndarray[np.float64_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)
        np.ndarray[np.float64_t, ndim=1] nans = PyArray_ZEROS(1, dims, NPY_float64, 0)
        float wt

        np.ndarray[np.float32_t, ndim=1] data = a.data
        np.ndarray[int, ndim=1] indices = a.indices
        np.ndarray[int, ndim=1] indptr = a.indptr
        int ri, i, ci
        np.float32_t ai
        int ain
    for ri in range(a.shape[0]):
        wt = 1 if w is None else w[ri]
        for i in range(indptr[ri], indptr[ri + 1]):
            ci = indices[i]
            if mask is None or mask[ci]:
                ai = data[i]
                if ai != ai:
                    nans[indices[i]] += wt
                    continue
                ain = int(ai + 0.1)
                if not -1e-6 < ain - ai < 1e-6:
                    raise ValueError("%f is not an integer value" % ai)
                if ain > max_val:
                    raise ValueError("value %i is greater than max_val (%i)" %
                                     (ain, max_val))
                y[ci, ain] += wt
    return y, nans



@cython.boundscheck(False)
@cython.wraparound(False)
def bincount_0d_float64_axisNone(object a,
           int max_val,
           np.ndarray[np.float_t, ndim=1] w = None,
           np.ndarray[np.int8_t, ndim=1] mask = None):
    '''Bincount that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''

    cdef:
        Py_ssize_t n_rows = a.shape[0]
        Py_ssize_t n_cols = a.shape[1]

    if w is not None and len(w) != n_rows:
        raise ValueError("invalid length of the weight vector")

    cdef:
        np.npy_intp *dims = [n_cols, max_val+1]
        np.ndarray[np.float64_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)
        np.ndarray[np.float64_t, ndim=1] nans = PyArray_ZEROS(1, dims, NPY_float64, 0)
        float wt

        np.ndarray[np.float64_t, ndim=1] data = a.data
        np.ndarray[int, ndim=1] indices = a.indices
        np.ndarray[int, ndim=1] indptr = a.indptr
        int ri, i, ci
        np.float64_t ai
        int ain
    for ri in range(a.shape[0]):
        wt = 1 if w is None else w[ri]
        for i in range(indptr[ri], indptr[ri + 1]):
            ci = indices[i]
            if mask is None or mask[ci]:
                ai = data[i]
                if ai != ai:
                    nans[indices[i]] += wt
                    continue
                ain = int(ai + 0.1)
                if not -1e-6 < ain - ai < 1e-6:
                    raise ValueError("%f is not an integer value" % ai)
                if ain > max_val:
                    raise ValueError("value %i is greater than max_val (%i)" %
                                     (ain, max_val))
                y[ci, ain] += wt
    return y, nans



cdef dict bincount_dict = {}
bincount_dict[(1, NPY_int8, 0)] = bincount_1d_int8_axisNone
bincount_dict[(1, NPY_int8, None)] = bincount_1d_int8_axisNone
bincount_dict[(1, NPY_int32, 0)] = bincount_1d_int32_axisNone
bincount_dict[(1, NPY_int32, None)] = bincount_1d_int32_axisNone
bincount_dict[(1, NPY_int64, 0)] = bincount_1d_int64_axisNone
bincount_dict[(1, NPY_int64, None)] = bincount_1d_int64_axisNone
bincount_dict[(2, NPY_int8, None)] = bincount_2d_int8_axisNone
bincount_dict[(2, NPY_int32, None)] = bincount_2d_int32_axisNone
bincount_dict[(2, NPY_int64, None)] = bincount_2d_int64_axisNone
bincount_dict[(0, NPY_int8, None)] = bincount_0d_int8_axisNone
bincount_dict[(0, NPY_int32, None)] = bincount_0d_int32_axisNone
bincount_dict[(0, NPY_int64, None)] = bincount_0d_int64_axisNone
bincount_dict[(1, NPY_float32, 0)] = bincount_1d_float32_axisNone
bincount_dict[(1, NPY_float32, None)] = bincount_1d_float32_axisNone
bincount_dict[(1, NPY_float64, 0)] = bincount_1d_float64_axisNone
bincount_dict[(1, NPY_float64, None)] = bincount_1d_float64_axisNone
bincount_dict[(2, NPY_float32, None)] = bincount_2d_float32_axisNone
bincount_dict[(2, NPY_float64, None)] = bincount_2d_float64_axisNone
bincount_dict[(0, NPY_float32, None)] = bincount_0d_float32_axisNone
bincount_dict[(0, NPY_float64, None)] = bincount_0d_float64_axisNone

def bincount_slow_axis0(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 0."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis1(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 1."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis2(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 2."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis3(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 3."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis4(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 4."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis5(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 5."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis6(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 6."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis7(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 7."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis8(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 8."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis9(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 9."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis10(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 10."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis11(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 11."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis12(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 12."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis13(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 13."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis14(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 14."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis15(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 15."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis16(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 16."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis17(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 17."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis18(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 18."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis19(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 19."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis20(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 20."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis21(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 21."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis22(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 22."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis23(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 23."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis24(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 24."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis25(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 25."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis26(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 26."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis27(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 27."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis28(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 28."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis29(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 29."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis30(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 30."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis31(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 31."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axis32(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis 32."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)

def bincount_slow_axisNone(arr, max_val, weights, mask):
    "Unaccelerated (slow) bincount along axis None."
    return bn.slow.bincount(arr, max_val, weights=None, mask=None)


cdef dict bincount_slow_dict = {}
bincount_slow_dict[0] = bincount_slow_axis0
bincount_slow_dict[1] = bincount_slow_axis1
bincount_slow_dict[2] = bincount_slow_axis2
bincount_slow_dict[3] = bincount_slow_axis3
bincount_slow_dict[4] = bincount_slow_axis4
bincount_slow_dict[5] = bincount_slow_axis5
bincount_slow_dict[6] = bincount_slow_axis6
bincount_slow_dict[7] = bincount_slow_axis7
bincount_slow_dict[8] = bincount_slow_axis8
bincount_slow_dict[9] = bincount_slow_axis9
bincount_slow_dict[10] = bincount_slow_axis10
bincount_slow_dict[11] = bincount_slow_axis11
bincount_slow_dict[12] = bincount_slow_axis12
bincount_slow_dict[13] = bincount_slow_axis13
bincount_slow_dict[14] = bincount_slow_axis14
bincount_slow_dict[15] = bincount_slow_axis15
bincount_slow_dict[16] = bincount_slow_axis16
bincount_slow_dict[17] = bincount_slow_axis17
bincount_slow_dict[18] = bincount_slow_axis18
bincount_slow_dict[19] = bincount_slow_axis19
bincount_slow_dict[20] = bincount_slow_axis20
bincount_slow_dict[21] = bincount_slow_axis21
bincount_slow_dict[22] = bincount_slow_axis22
bincount_slow_dict[23] = bincount_slow_axis23
bincount_slow_dict[24] = bincount_slow_axis24
bincount_slow_dict[25] = bincount_slow_axis25
bincount_slow_dict[26] = bincount_slow_axis26
bincount_slow_dict[27] = bincount_slow_axis27
bincount_slow_dict[28] = bincount_slow_axis28
bincount_slow_dict[29] = bincount_slow_axis29
bincount_slow_dict[30] = bincount_slow_axis30
bincount_slow_dict[31] = bincount_slow_axis31
bincount_slow_dict[32] = bincount_slow_axis32
bincount_slow_dict[None] = bincount_slow_axisNone