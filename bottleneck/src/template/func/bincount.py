"nanequal template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["bincount"]

FLOAT_DTYPES = [x for x in bn.dtypes if 'float' in x]
INT_DTYPES = [x for x in bn.dtypes if 'int' in x]

# Float dtypes (not axis=None) ----------------------------------------------

floats = {}
floats['dtypes'] = FLOAT_DTYPES
floats['axisNone'] = True
floats['force_output_dtype'] = 'bool'
floats['reuse_non_nan_func'] = False

floats['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a,
                              int max_val,
                              np.ndarray[np.float_t, ndim=1] w = None):
    '''Bincount that can handle floats (casts to int), handles NaNs and needs to
     be specified a fixed number of values'''
"""

loop = {}

loop[1] = """\
    if w is not None and len(w) != n0:
        raise ValueError("invalid length of the weight vector")

    cdef:
        np.npy_intp *dims = [max_val+1]
        np.ndarray[np.float64_t, ndim=1] y = PyArray_ZEROS(1, dims, NPY_float64, 0)
        float nans = 0
        int ain
        float ai, wt
    for iINDEX0 in range(nINDEX0):
        wt = 1 if w is None else w[iINDEX0]
        ai = a[INDEXALL]
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
"""

loop[2] = """\
    if w is not None and len(w) != len(a):
        raise ValueError("invalid length of the weight vector")

    cdef:
       np.npy_intp *dims = [n1, max_val+1]
       np.ndarray[np.float64_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)
       np.npy_intp *nandims = [n1]
       np.ndarray[np.float_t, ndim=1] nans = PyArray_ZEROS(1, nandims, NPY_float64, 0)
       int ain
       float ai, wt
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            wt = 1 if w is None else w[iINDEX0]
            ai = a[INDEXALL]
            if ai != ai:
                nans[iINDEX1] += wt
                continue
            if ai < -1e-6:
                raise ValueError("negative value in bincount")
            ain = int(ai + 0.1)
            if not -1e-6 < ain - ai < 1e-6:
                raise ValueError("%f is not an integer value" % ai)
            if ain > max_val:
                raise ValueError("value %i is greater than max_val (%i)" %
                                 (ain, max_val))
            y[iINDEX1, ain] += wt
    return y, nans
"""

sparse = """
@cython.boundscheck(False)
@cython.wraparound(False)
def SPARSE(object a, int max_val,
                      np.ndarray[np.float_t, ndim=1] w = None):
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

        np.ndarray[np.DTYPE_t, ndim=1] data = a.data
        np.ndarray[int, ndim=1] indices = a.indices
        np.ndarray[int, ndim=1] indptr = a.indptr
        int ri, i
        np.DTYPE_t ai
        int ain
    for ri in range(a.shape[0]):
        wt = 1 if w is None else w[ri]
        for i in range(indptr[ri], indptr[ri + 1]):
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
            y[indices[i], ain] += wt
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
        np.npy_intp *dims = [max_val+1]
        np.ndarray[np.float64_t, ndim=1] y = PyArray_ZEROS(1, dims, NPY_float64, 0)
        int ai
    for iINDEX0 in range(nINDEX0):
        ai = a[INDEXALL]
        if ai < 0:
            raise ValueError("negative value in bincount")
        if ai > max_val:
            raise ValueError("value %i is greater than max_val (%i)"
                             % (ai, max_val))
        y[ai] += 1 if w is None else w[iINDEX0]
    return y, 0.0
"""

loop[2] = """\
    if w is not None and len(w) != len(a):
        raise ValueError("invalid length of the weight vector")

    cdef:
       np.npy_intp *dims = [n1, max_val+1]
       np.ndarray[np.float_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)
       np.npy_intp *nandims = [n1]
       np.ndarray[np.float64_t, ndim=1] nans = PyArray_ZEROS(1, nandims, NPY_float64, 0)
       int ai
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai < 0:
                raise ValueError("negative value in bincount")
            if ai > max_val:
                raise ValueError("value %i is greater than max_val (%i)"
                                 % (ai, max_val))
            y[iINDEX1, ai] += 1 if w is None else w[iINDEX0]
    return y, nans
"""

sparse = """
@cython.boundscheck(False)
@cython.wraparound(False)
def SPARSE(object a, int max_val,
                      np.ndarray[np.float_t, ndim=1] w = None):
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

        np.ndarray[np.DTYPE_t, ndim=1] data = a.data
        np.ndarray[int, ndim=1] indices = a.indices
        np.ndarray[int, ndim=1] indptr = a.indptr
        int ri, i
    for ri in range(a.shape[0]):
        wt = 1 if w is None else w[ri]
        for i in range(indptr[ri], indptr[ri + 1]):
            y[indices[i], data[i]] += wt
    return y, nans
"""

ints['loop'] = loop
ints['sparse'] = sparse



# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "bincount"
slow['signature'] = "arr, max_val, weights"
slow['func'] = "bn.slow.bincount(arr, max_val, weights=None)"

# Template ------------------------------------------------------------------

bincount = {}
bincount['name'] = 'bincount'
bincount['is_reducing_function'] = False
bincount['cdef_output'] = False
bincount['slow'] = slow
bincount['sparse'] = {}
bincount['templates'] = {}
bincount['templates']['float_None'] = floats
bincount['templates']['int_None'] = ints
bincount['pyx_file'] = 'func/%sbit/bincount.pyx'

bincount['main'] = '''"nanequal auto-generated from template"

def bincount(arr, max_val, weights=None):
    """
    Count number of occurrences of each value in array.

    The number of bins is max_val+1. Each bin gives the number of occurrences
    of its index value in `x`. If `weights` is specified the input array is
    weighted by it, i.e. if a value ``n`` is found at position ``i``,
    ``out[n] += weight[i]`` instead of ``out[n] += 1``.

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
    func, a = bincount_selector(arr, max_val, weights)
    return func(a, max_val, weights)




def bincount_selector(arr, max_val, weights):
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
    try:
        func = bincount_dict[key]
        return func, a,
    except KeyError:
        pass

    try:
        func = bincount_slow_dict[None]
    except KeyError:
        tup = (str(ndim), str(a.dtype))
        raise TypeError("Unsupported ndim/dtype (%s/%s)." % tup)
    return func, a
'''
