"contingency template"

from copy import deepcopy
import bottlechest as bn

__all__ = ["contingency"]

FLOAT_DTYPES = [x for x in bn.dtypes if 'float' in x]
INT_DTYPES = [x for x in bn.dtypes if 'int' in x]

# Float dtypes (not axis=None) ----------------------------------------------

floats = {}
floats['dtypes'] = FLOAT_DTYPES
floats['axisNone'] = True
floats['force_output_dtype'] = 'bool'
floats['reuse_non_nan_func'] = False

floats['top'] = """
cdef extern from "stdint.h":
    ctypedef unsigned char uint8_t


@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a,
                              np.ndarray[np.uint8_t, ndim=1] b,
                              int max_val, int max_val2,
                              np.ndarray[np.float_t, ndim=1] w = None,
                              np.ndarray[np.int8_t, ndim=1] mask = None):
    '''Contingency matrix from array. For 2d-data, it returns 3d array
       [variable, value-from-a, value-from-b]. For 3-d, it returns a 2d array'''
"""

loop = {}

loop[1] = """\
    if len(b) != n0:
        raise ValueError("invalid length of the column variable vector")
    if w is not None and len(w) != n0:
        raise ValueError("invalid length of the weight vector")

    cdef:
        np.npy_intp *dims = [max_val2 + 1, max_val + 1]
        np.ndarray[np.float64_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)
        np.npy_intp *nandims = [max_val2 + 1]
        np.ndarray[np.float_t, ndim=1] nans = PyArray_ZEROS(1, nandims, NPY_float64, 0)
        int ain
        uint8_t bin
        float ai, wt
    if mask is not None and not mask[0]:
        return y, nans
    if w is None:
        for iINDEX0 in range(nINDEX0):
            bin = b[iINDEX0]
            if bin > max_val2:
                raise ValueError("value %i is greater than max_val2 (%i)" %
                                 (bin, max_val2))
            ai = a[INDEXALL]
            if ai != ai:
                nans[bin] += 1
                continue
            if ai < -1e-6:
                raise ValueError("negative value in contingency")
            ain = int(ai + 0.1)
            if not -1e-6 < ain - ai < 1e-6:
                raise ValueError("%f is not an integer value" % ai)
            if ain > max_val:
                raise ValueError("value %i is greater than max_val (%i)" %
                                 (ain, max_val))
            y[bin, ain] += 1
    else:
        for iINDEX0 in range(nINDEX0):
            bin = b[iINDEX0]
            if bin > max_val2:
                raise ValueError("value %i is greater than max_val2 (%i)" %
                                 (bin, max_val2))
            wt = w[iINDEX0]
            ai = a[INDEXALL]
            if ai != ai:
                nans[bin] += wt
                continue
            if ai < -1e-6:
                raise ValueError("negative value in contingency")
            ain = int(ai + 0.1)
            if not -1e-6 < ain - ai < 1e-6:
                raise ValueError("%f is not an integer value" % ai)

            y[bin, ain] += wt
    return y, nans
"""

loop[2] = """\
    if len(b) != n0:
        raise ValueError("invalid length of the column variable vector")
    if w is not None and len(w) != len(a):
        raise ValueError("invalid length of the weight vector")

    cdef:
       np.npy_intp *dims = [n1, max_val2 + 1, max_val + 1]
       np.ndarray[np.float64_t, ndim=3] y = PyArray_ZEROS(3, dims, NPY_float64, 0)
       np.npy_intp *nandims = [n1, max_val2 + 1]
       np.ndarray[np.float_t, ndim=2] nans = PyArray_ZEROS(2, nandims, NPY_float64, 0)
       int ain
       uint8_t bin
       float ai, wt
    for iINDEX0 in range(nINDEX0):
        wt = 1 if w is None else w[iINDEX0]
        bin = b[iINDEX0]
        if bin > max_val2:
            raise ValueError("value %i is greater than max_val2 (%i)" %
                             (bin, max_val2))
        for iINDEX1 in range(nINDEX1):
            if mask is None or mask[iINDEX1]:
                ai = a[INDEXALL]
                if ai != ai:
                    nans[iINDEX1, bin] += wt
                    continue
                if ai < -1e-6:
                    raise ValueError("negative value in contingency")
                ain = int(ai + 0.1)
                if not -1e-6 < ain - ai < 1e-6:
                    raise ValueError("%f is not an integer value" % ai)
                if ain > max_val:
                    raise ValueError("value %i is greater than max_val (%i)" %
                                     (ain, max_val))
                y[iINDEX1, bin, ain] += wt
    return y, nans
"""

sparse = """
@cython.boundscheck(False)
@cython.wraparound(False)
def SPARSE(object a,
           np.ndarray[np.uint8_t, ndim=1] b,
           int max_val, int max_val2,
           np.ndarray[np.float_t, ndim=1] w = None,
           np.ndarray[np.int8_t, ndim=1] mask = None):
    '''contingency that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''

    cdef:
        Py_ssize_t n_rows = a.shape[0]
        Py_ssize_t n_cols = a.shape[1]

    if len(b) != n_rows:
        raise ValueError("invalid length of the column variable vector")
    if w is not None and len(w) != n_rows:
        raise ValueError("invalid length of the weight vector")

    cdef:
        np.npy_intp *dims = [n_cols, max_val2 + 1, max_val + 1]
        np.ndarray[np.float64_t, ndim=3] y = PyArray_ZEROS(3, dims, NPY_float64, 0)
        np.npy_intp *nandims = [n_cols, max_val2 + 1]
        np.ndarray[np.float_t, ndim=2] nans = PyArray_ZEROS(2, nandims, NPY_float64, 0)
        float wt

        np.ndarray[np.DTYPE_t, ndim=1] data = a.data
        np.ndarray[int, ndim=1] indices = a.indices
        np.ndarray[int, ndim=1] indptr = a.indptr
        int ri, i, ci
        np.DTYPE_t ai
        int ain
        uint8_t bin
    for ri in range(a.shape[0]):
        wt = 1 if w is None else w[ri]
        bin = b[ri]
        if bin > max_val2:
            raise ValueError("value %i is greater than max_val2 (%i)" %
                             (bin, max_val2))
        for i in range(indptr[ri], indptr[ri + 1]):
            ci = indices[i]
            if mask is None or mask[ci]:
                ai = data[i]
                if ai != ai:
                    nans[indices[i], bin] += wt
                    continue
                ain = int(ai + 0.1)
                if not -1e-6 < ain - ai < 1e-6:
                    raise ValueError("%f is not an integer value" % ai)
                if ain > max_val:
                    raise ValueError("value %i is greater than max_val (%i)" %
                                     (ain, max_val))
                y[ci, bin, ain] += wt
    return y, nans
"""

floats['loop'] = loop
floats['sparse'] = sparse

# Int dtypes (not axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['dtypes'] = INT_DTYPES

loop = {}
loop[1] = """\
    if w is not None and len(w) != n0:
        raise ValueError("invalid length of the weight vector")

    cdef:
        np.npy_intp *dims = [max_val2 + 1, max_val + 1]
        np.ndarray[np.float64_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)
        np.npy_intp *nandims = [max_val2 + 1]
        np.ndarray[np.float_t, ndim=1] nans = PyArray_ZEROS(1, nandims, NPY_float64, 0)
        int ai
        uint8_t bin
    if mask is not None and not mask[0]:
        return y, nans
    if w is None:
        for iINDEX0 in range(nINDEX0):
            ai = a[INDEXALL]
            if ai < 0:
                raise ValueError("negative value in contingency")
            if ai > max_val:
                raise ValueError("value %i is greater than max_val (%i)"
                                 % (ai, max_val))
            bin = b[iINDEX0]
            if bin > max_val2:
                raise ValueError("value %i is greater than max_val2 (%i)" %
                                 (bin, max_val2))
            y[bin, ai] += 1
    else:
        for iINDEX0 in range(nINDEX0):
            ai = a[INDEXALL]
            if ai < 0:
                raise ValueError("negative value in contingency")
            if ai > max_val:
                raise ValueError("value %i is greater than max_val (%i)"
                                 % (ai, max_val))
            bin = b[iINDEX0]
            if bin > max_val2:
                raise ValueError("value %i is greater than max_val2 (%i)" %
                                 (bin, max_val2))
            y[bin, ai] += w[iINDEX0]
    return y, nans
"""

loop[2] = """\
    if w is not None and len(w) != len(a):
        raise ValueError("invalid length of the weight vector")

    cdef:
       np.npy_intp *dims = [n1, max_val2 + 1, max_val + 1]
       np.ndarray[np.float_t, ndim=3] y = PyArray_ZEROS(3, dims, NPY_float64, 0)
       np.npy_intp *nandims = [n1, max_val2+1]
       np.ndarray[np.float64_t, ndim=2] nans = PyArray_ZEROS(2, nandims, NPY_float64, 0)
       int ai
       uint8_t bin
       float wt
    for iINDEX0 in range(nINDEX0):
        bin = b[iINDEX0]
        if bin > max_val2:
            raise ValueError("value %i is greater than max_val2 (%i)" %
                             (bin, max_val2))
        wt = 1 if w is None else w[iINDEX0]
        for iINDEX1 in range(nINDEX1):
            if mask is None or mask[iINDEX1]:
                ai = a[INDEXALL]
                if ai < 0:
                    raise ValueError("negative value in contingency")
                if ai > max_val:
                    raise ValueError("value %i is greater than max_val (%i)"
                                     % (ai, max_val))
                y[iINDEX1, bin, ai] += wt
    return y, nans
"""

sparse = """
@cython.boundscheck(False)
@cython.wraparound(False)
def SPARSE(object a,
           np.ndarray[np.uint8_t, ndim=1] b,
           int max_val, int max_val2,
           np.ndarray[np.float_t, ndim=1] w = None,
           np.ndarray[np.int8_t, ndim=1] mask = None):
    '''contingency that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''

    cdef:
        Py_ssize_t n_rows = a.shape[0]
        Py_ssize_t n_cols = a.shape[1]

    if w is not None and len(w) != n_rows:
        raise ValueError("invalid length of the weight vector")

    cdef:
        np.npy_intp *dims = [n_cols, max_val2 + 1, max_val + 1]
        np.ndarray[np.float64_t, ndim=3] y = PyArray_ZEROS(3, dims, NPY_float64, 0)
        np.npy_intp *nandims = [n_cols, max_val2+1]
        np.ndarray[np.float_t, ndim=2] nans = PyArray_ZEROS(2, nandims, NPY_float64, 0)
        float wt
        uint8_t bin

        np.ndarray[np.DTYPE_t, ndim=1] data = a.data
        np.ndarray[int, ndim=1] indices = a.indices
        np.ndarray[int, ndim=1] indptr = a.indptr
        int ri, i, ci
    for ri in range(a.shape[0]):
        wt = 1 if w is None else w[ri]
        bin = b[ri]
        if bin > max_val2:
            raise ValueError("value %i is greater than max_val2 (%i)" %
                             (bin, max_val2))
        for i in range(indptr[ri], indptr[ri + 1]):
            ci = indices[i]
            if mask is None or mask[ci]:
                y[ci, bin, data[i]] += wt
    return y, nans
"""

ints['loop'] = loop
ints['sparse'] = sparse



# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "contingency"
slow['signature'] = "arr, b, max_val, max_val2, weights, mask"
slow['func'] = "bn.slow.contingency(arr, b, max_val, max_val2, weights=None, mask=None)"

# Template ------------------------------------------------------------------

contingency = {}
contingency['name'] = 'contingency'
contingency['is_reducing_function'] = False
contingency['cdef_output'] = False
contingency['slow'] = slow
contingency['sparse'] = {}
contingency['templates'] = {}
contingency['templates']['float_None'] = floats
contingency['templates']['int_None'] = ints
contingency['pyx_file'] = 'func/%sbit/contingency.pyx'

contingency['main'] = '''"nanequal auto-generated from template"

def contingency(arr, b, max_val, max_val2, weights=None, mask=None):
    """
    Compute the contingency matrices for each column (excluding the masked)
    versus the vector b.

    If the array is 1-dimensional, a 2d contingency matrix is returned. If the
    array is 2d, the function returns a 3d array, with the first dimension
    corresponding to column index (variable in the input array).

    The rows of contingency matrix correspond to values of variables, the
    columns correspond to values in vector `b`. The number of rows is
    `max_val+1`, the nuber of columns is `max_val2+1`.

    Rows in the input array can be weights (argument `weights`). A subset of
    columns can be selected by additional argument `mask`.

    The function also returns a count of NaN values per each value of `b`.

    Parameters
    ----------
    x : array_like, 1 or 2 dimensions, nonnegative elements
        Input array.
    b : array_like, 1 dimension, of type uint8
        The values of the second variable.
    max_val : int
        The maximal value in the array
    max_val2 : int
        The maximal value in `b`
    weights : array_like, optional
        Weights, array of the same length as `x`.
    mask: array_like, of type char (interpreted as bool)
        Selects the columns

    Returns
    -------
    out : ndarray of ints, 1- or 2-dimensional
        The result of binning the input array.
    nans: the number of NaNs; a 1-d vector of length `max_val`, or 2-d array `(max_val2, x.shape[1])`

    Raises
    ------
    ValueError
        If the input is not 1- or 2-dimensional, or contains elements that are
        not close enough to integers, negative or grater than max_val, or if the
        length of the weight vector does not match the length of the array

    """
    func, a, b, weights, mask = contingency_selector(arr, b, max_val, max_val2, weights, mask)
    return func(a, b, max_val, max_val2, weights, mask)




def contingency_selector(arr, b, max_val, max_val2, weights, mask):
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

    if type(b) is np.ndarray:
        if b.dtype != np.uint8:
            b = b.astype(np.uint8)
    else:
        b = np.array(b, copy=False, dtype=np.uint8)
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
        func = contingency_dict[key]
        return func, a, b, weights, mask
    except KeyError:
        pass

    try:
        func = contingency_slow_dict[None]
    except KeyError:
        tup = (str(ndim), str(a.dtype))
        raise TypeError("Unsupported ndim/dtype (%s/%s)." % tup)
    return func, a, b, weights, mask
'''
