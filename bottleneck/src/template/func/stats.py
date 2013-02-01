"stats template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["stats"]

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
                              np.ndarray[np.float_t, ndim=1] w = None):
    '''Compute min, max, mean and #nans.
    '''
"""

loop = {}

loop[1] = """\
    cdef:
        np.DTYPE_t ai
        np.float_t wt
        np.DTYPE_t a_min = MAXDTYPE
        np.DTYPE_t a_max = MINDTYPE
        np.float_t mean = 0
        np.float_t non_nans = 0
        np.float_t nans = 0
    if w is None:
        for iINDEX0 in range(nINDEX0):
            ai = a[iINDEX0]
            if ai != ai:
                nans += 1
                continue
            if ai < a_min:
                a_min = ai
            if ai > a_max:
                a_max = ai
            mean += ai
        non_nans = nINDEX0 - nans
        if non_nans != 0:
            mean /= non_nans
    else:
        if len(w) != n0:
            raise ValueError("invalid length of the weight vector ({} != {})".
                format(len(w), n0))
        for iINDEX0 in range(nINDEX0):
            ai = a[iINDEX0]
            wt = w[iINDEX0]
            if ai != ai:
                nans += wt
                continue
            else:
                non_nans += wt
            if ai < a_min:
                a_min = ai
            if ai > a_max:
                a_max = ai
            mean += wt * ai
        if non_nans != 0:
            mean /= non_nans
    return a_min, a_max, mean, 0, nans, non_nans
"""

loop[2] = """\
    cdef:
       np.npy_intp *dims = [nINDEX1, 6]
       np.ndarray[np.float64_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)
       np.DTYPE_t ai
       np.float64_t mean
       np.float_t wt

    for iINDEX1 in range(nINDEX1):
       y[iINDEX1, 0] = MAXfloat64
       y[iINDEX1, 1] = MINfloat64
    if w is None:
        for iINDEX0 in range(nINDEX0):
            for iINDEX1 in range(nINDEX1):
                ai = a[INDEXALL]
                if ai != ai:
                    y[iINDEX1, 4] += 1
                    continue
                if ai < y[iINDEX1, 0]:
                    y[iINDEX1, 0] = ai
                if ai > y[iINDEX1, 1]:
                    y[iINDEX1, 1] = ai
                y[iINDEX1, 2] += ai
        for iINDEX1 in range(nINDEX1):
            y[iINDEX1, 5] = nINDEX0 - y[iINDEX1, 4]
            if y[iINDEX1, 5] > 0:
                y[iINDEX1, 2] /= y[iINDEX1, 5]
    else:
        for iINDEX0 in range(nINDEX0):
            wt = w[iINDEX0]
            for iINDEX1 in range(nINDEX1):
                ai = a[INDEXALL]
                if ai != ai:
                    y[iINDEX1, 4] += wt
                    continue
                y[iINDEX1, 5] += wt
                if ai < y[iINDEX1, 0]:
                    y[iINDEX1, 0] = ai
                if ai > y[iINDEX1, 1]:
                    y[iINDEX1, 1] = ai
                y[iINDEX1, 2] += wt * ai
        for iINDEX1 in range(nINDEX1):
            if y[iINDEX1, 5] > 0:
                y[iINDEX1, 2] /= y[iINDEX1, 5]
    return y
"""

sparse = """
@cython.boundscheck(False)
@cython.wraparound(False)
def SPARSE(object a,
           np.ndarray[np.float_t, ndim=1] w = None):
    '''Compute min, max, #nans, mean and variance.
    '''

    cdef:
        Py_ssize_t n_rows = a.shape[0]
        Py_ssize_t n_cols = a.shape[1]

    if w is not None and len(w) != n_rows:
        raise ValueError("invalid length of the weight vector")

    cdef:
        np.ndarray[np.DTYPE_t, ndim=1] data = a.data
        np.ndarray[int, ndim=1] indices = a.indices
        np.ndarray[int, ndim=1] indptr = a.indptr

        np.npy_intp *dims = [n_cols, 6]
        np.ndarray[np.float64_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)

        int ri, ci
        np.float_t wt
        np.float_t tot_w = 0
        np.DTYPE_t ai

    for ci in range(n_cols):
       y[ci, 0] = MAXfloat64
       y[ci, 1] = MINfloat64
    if w is None:
       tot_w = n_rows
    else:
        for ri in range(n_rows):
            tot_w += w[ri]
    if tot_w == 0:
        return y

    for ri in range(a.shape[0]):
        wt = 1 if w is None else w[ri]
        for i in range(indptr[ri], indptr[ri + 1]):
            ai = data[i]
            if ai != ai:
                continue
            ci = indices[i]
            y[ci, 5] += wt
            if ai < y[ci, 0]:
                y[ci, 0] = ai
            if ai > y[ci, 1]:
                y[ci, 1] = ai
            y[ci, 2] += wt * ai
    for ci in range(n_cols):
        y[ci, 4] = tot_w - y[ci, 5]
        if y[ci, 5] != 0:
            y[ci, 2] /= y[ci, 5]
    return y
"""

floats['loop'] = loop
floats['sparse'] = sparse

# Int dtypes (not axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['dtypes'] = INT_DTYPES

loop = {}
loop[1] = """\
    cdef:
        np.DTYPE_t ai
        np.float_t wt
        np.DTYPE_t a_min = MAXDTYPE
        np.DTYPE_t a_max = MINDTYPE
        np.float64_t mean = 0
        np.float_t tot_w
    if n0 == 0:
        return (a_min, a_max, 0, 0, 0, 0)
    if w is None:
        tot_w = nINDEX0
        for iINDEX0 in range(nINDEX0):
            ai = a[INDEXALL]
            if ai < a_min:
                a_min = ai
            if ai > a_max:
                a_max = ai
            mean += ai
        mean /= n0
    else:
        tot_w = 0
        if len(w) != n0:
            raise ValueError("invalid length of the weight vector")
        for iINDEX0 in range(nINDEX0):
            ai = a[INDEXALL]
            wt = w[iINDEX0]
            tot_w += wt
            if ai < a_min:
                a_min = ai
            if ai > a_max:
                a_max = ai
            mean += wt * ai
        if tot_w != 0:
            mean /= tot_w
    return a_min, a_max, mean, 0, 0, tot_w
"""

loop[2] = """\
    cdef:
       np.npy_intp *dims = [n1, 6]
       np.ndarray[np.float64_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)
       np.DTYPE_t ai
       np.float64_t mean
       np.float_t wt
       np.float_t tot_w = 0

    if w is None:
       tot_w = nINDEX0
    else:
        for iINDEX0 in range(nINDEX0):
            tot_w += w[iINDEX0]

    for iINDEX1 in range(nINDEX1):
       y[iINDEX1, 0] = MAXfloat64
       y[iINDEX1, 1] = MINfloat64
       y[iINDEX1, 5] = tot_w

    if tot_w == 0:
        return y

    if w is None:
        for iINDEX0 in range(nINDEX0):
            for iINDEX1 in range(nINDEX1):
                ai = a[INDEXALL]
                if ai < y[iINDEX1, 0]:
                    y[iINDEX1, 0] = ai
                if ai > y[iINDEX1, 1]:
                    y[iINDEX1, 1] = ai
                y[iINDEX1, 2] += ai
        for iINDEX1 in range(nINDEX1):
            y[iINDEX1, 2] /= nINDEX0
            mean = y[iINDEX1, 2]
    else:
        for iINDEX0 in range(nINDEX0):
            wt = w[iINDEX0]
            for iINDEX1 in range(nINDEX1):
                ai = a[INDEXALL]
                if ai < y[iINDEX1, 0]:
                    y[iINDEX1, 0] = ai
                if ai > y[iINDEX1, 1]:
                    y[iINDEX1, 1] = ai
                y[iINDEX1, 2] += wt * ai
        for iINDEX1 in range(nINDEX1):
            y[iINDEX1, 2] /= tot_w
            mean = y[iINDEX1, 2]
    return y
"""

sparse = """
@cython.boundscheck(False)
@cython.wraparound(False)
def SPARSE(object a,
           np.ndarray[np.float_t, ndim=1] w = None):
    '''Compute min, max, #nans, mean and variance.
    '''

    cdef:
        Py_ssize_t n_rows = a.shape[0]
        Py_ssize_t n_cols = a.shape[1]

    if w is not None and len(w) != n_rows:
        raise ValueError("invalid length of the weight vector")

    cdef:
        np.ndarray[np.DTYPE_t, ndim=1] data = a.data
        np.ndarray[int, ndim=1] indices = a.indices
        np.ndarray[int, ndim=1] indptr = a.indptr

        np.npy_intp *dims = [n_cols, 6]
        np.ndarray[np.float64_t, ndim=2] y = PyArray_ZEROS(2, dims, NPY_float64, 0)
        np.float_t wt
        np.float_t tot_w = 0
        int ri, ci
        np.DTYPE_t ai

    for ci in range(n_cols):
       y[ci, 0] = MAXfloat64
       y[ci, 1] = MINfloat64

    if w is None:
       tot_w = n_rows
    else:
        for ri in range(n_rows):
            tot_w += w[ri]
    if tot_w == 0:
        return y

    for ri in range(a.shape[0]):
        wt = 1 if w is None else w[ri]
        for i in range(indptr[ri], indptr[ri + 1]):
            ci = indices[i]
            ai = data[i]
            if ai < y[ci, 0]:
                y[ci, 0] = ai
            if ai > y[ci, 1]:
                y[ci, 1] = ai
            y[ci, 5] += wt
            y[ci, 2] += wt * ai
    for ci in range(n_cols):
        y[ci, 4] = tot_w - y[ci, 5]
        if y[ci, 5] != 0:
            y[ci, 2] /= y[ci, 5]
    return y
"""

ints['loop'] = loop
ints['sparse'] = sparse



# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "stats"
slow['signature'] = "arr, weights"
slow['func'] = "bn.slow.stats(arr, weights=None)"

# Template ------------------------------------------------------------------

stats = {}
stats['name'] = 'stats'
stats['is_reducing_function'] = False
stats['cdef_output'] = False
stats['slow'] = slow
stats['sparse'] = {}
stats['templates'] = {}
stats['templates']['float_None'] = floats
stats['templates']['int_None'] = ints
stats['pyx_file'] = 'func/%sbit/stats.pyx'

stats['main'] = '''"stats auto-generated from template"

def stats(arr, weights=None):
    """
    Compute min, max, #nans, mean and variance.

    Result is a tuple (min, max, mean, 0, #nans, #non-nans) or an
    array of shape (len(arr), 6).

    The mean and the number of nans and non-nans are weighted.

    The zero element can be filled-in by var if statvar function
    is subsequently called with this data.

    Parameters
    ----------
    x : array_like, 1 or 2 dimensions
        Input array.
    weights : array_like, optional
        Weights, array of the same length as `x`.

    Returns
    -------
    out : a 6-element tuple or an array of shape (len(x), 6)
        Computed (min, max, mean, 0, #nans and #non-nans)

    Raises
    ------
    ValueError
        If the length of the weight vector does not match the length of the
        array
    """
    func, a, weights = stats_selector(arr, weights)
    return func(a, weights)




def stats_selector(arr, weights):
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

    try:
        func = stats_dict[key]
        return func, a, weights
    except KeyError:
        pass

    try:
        func = stats_slow_dict[None]
    except KeyError:
        tup = (str(ndim), str(a.dtype))
        raise TypeError("Unsupported ndim/dtype (%s/%s)." % tup)
    return func, a, weights
'''
