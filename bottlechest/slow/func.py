import math
import warnings
import numpy as np
import scipy.sparse as sp

__all__ = ['median', 'nanmedian', 'nansum', 'nanmean', 'nanvar', 'nanstd',
           'nanmin', 'nanmax', 'nanargmin', 'nanargmax', 'rankdata',
           'nanrankdata', 'ss', 'nn', 'partsort', 'argpartsort', 'replace',
           'anynan', 'allnan',
           'bincount', 'valuecount', 'countnans', 'stats',
           'contingency', 'nanequal']

def median(arr, axis=None):
    "Slow median function used for unaccelerated ndim/dtype combinations."
    arr = np.asarray(arr)
    y = np.median(arr, axis=axis)
    if y.dtype != arr.dtype:
        if issubclass(arr.dtype.type, np.inexact):
            y = y.astype(arr.dtype)
    return y

def nansum(arr, axis=None):
    "Slow nansum function used for unaccelerated ndim/dtype combinations."
    arr = np.asarray(arr)
    y = np.nansum(arr, axis=axis)
    if not hasattr(y, "dtype"):
        y = arr.dtype.type(y)
    if y.dtype != arr.dtype:
        if issubclass(arr.dtype.type, np.inexact):
            y = y.astype(arr.dtype)
    return y

def nanmedian(arr, axis=None):
    "Slow nanmedian function used for unaccelerated ndim/dtype combinations."
    arr = np.asarray(arr)
    y = scipy_nanmedian(arr, axis=axis)
    if not hasattr(y, "dtype"):
        if issubclass(arr.dtype.type, np.inexact):
            y = arr.dtype.type(y)
        else:
            y = np.float64(y)
    if y.dtype != arr.dtype:
        if issubclass(arr.dtype.type, np.inexact):
            y = y.astype(arr.dtype)
    if (y.size == 1) and (y.ndim == 0):
        y = y[()]
    return y

def nanmean(arr, axis=None):
    "Slow nanmean function used for unaccelerated ndim/dtype combinations."
    return np.nanmean(arr, axis=axis)

def nanvar(arr, axis=None, ddof=0):
    "Slow nanvar function used for unaccelerated ndim/dtype combinations."
    return np.nanvar(arr, axis=axis, ddof=ddof)

def nanstd(arr, axis=None, ddof=0):
    "Slow nanstd function used for unaccelerated ndim/dtype combinations."
    return np.nanstd(arr, axis=axis, ddof=ddof)

def nanmin(arr, axis=None):
    "Slow nanmin function used for unaccelerated ndim/dtype combinations."
    y =  np.nanmin(arr, axis=axis)
    if not hasattr(y, "dtype"):
        # Numpy 1.5.1 doesn't return object with dtype when input is all NaN
        y = arr.dtype.type(y)
    return y

def nanmax(arr, axis=None):
    "Slow nanmax function used for unaccelerated ndim/dtype combinations."
    y =  np.nanmax(arr, axis=axis)
    if not hasattr(y, "dtype"):
        # Numpy 1.5.1 doesn't return object with dtype when input is all NaN
        y = arr.dtype.type(y)
    return y

def nanargmin(arr, axis=None):
    "Slow nanargmin function used for unaccelerated ndim/dtype combinations."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanargmin(arr, axis=axis)

def nanargmax(arr, axis=None):
    "Slow nanargmax function used for unaccelerated ndim/dtype combinations."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanargmax(arr, axis=axis)

def rankdata(arr, axis=None):
    "Slow rankdata function used for unaccelerated ndim/dtype combinations."
    arr = np.asarray(arr)
    if axis is None:
        arr = arr.ravel()
        axis = 0
    elif axis < 0:
        axis = range(arr.ndim)[axis]
    y = np.empty(arr.shape)
    itshape = list(arr.shape)
    itshape.pop(axis)
    for ij in np.ndindex(*itshape):
        ijslice = list(ij[:axis]) + [slice(None)] + list(ij[axis:])
        y[ijslice] = scipy_rankdata(arr[ijslice].astype('float'))
    return y

def nanrankdata(arr, axis=None):
    "Slow nanrankdata function used for unaccelerated ndim/dtype combinations."
    arr = np.asarray(arr)
    if axis is None:
        arr = arr.ravel()
        axis = 0
    elif axis < 0:
        axis = range(arr.ndim)[axis]
    y = np.empty(arr.shape)
    y.fill(np.nan)
    itshape = list(arr.shape)
    itshape.pop(axis)
    for ij in np.ndindex(*itshape):
        ijslice = list(ij[:axis]) + [slice(None)] + list(ij[axis:])
        x1d = arr[ijslice].astype(float)
        mask1d = ~np.isnan(x1d)
        x1d[mask1d] = scipy_rankdata(x1d[mask1d])
        y[ijslice] = x1d
    return y

def ss(arr, axis=0):
    "Slow sum of squares used for unaccelerated ndim/dtype combinations."
    return scipy_ss(arr, axis)

def nn(arr, arr0, axis=1):
    "Slow nearest neighbor used for unaccelerated ndim/dtype combinations."
    arr = np.array(arr, copy=False)
    arr0 = np.array(arr0, copy=False)
    if arr.ndim != 2:
        raise ValueError("`arr` must be 2d")
    if arr0.ndim != 1:
        raise ValueError("`arr0` must be 1d")
    if axis == 1:
        d = (arr - arr0) ** 2
    elif axis == 0:
        d = (arr - arr0.reshape(-1,1)) ** 2
    else:
        raise ValueError("`axis` must be 0 or 1.")
    d = d.sum(axis)
    idx = np.argmin(d)
    return np.sqrt(d[idx]), idx

def partsort(arr, n, axis=-1):
    "Slow partial sort used for unaccelerated ndim/dtype combinations."
    return np.sort(arr, axis)

def argpartsort(arr, n, axis=-1):
    "Slow partial argsort used for unaccelerated ndim/dtype combinations."
    return np.argsort(arr, axis)

def replace(arr, old, new):
    "Slow replace (inplace) used for unaccelerated ndim/dtype combinations."
    if type(arr) is not np.ndarray:
        raise TypeError("`arr` must be a numpy array.")
    if not issubclass(arr.dtype.type, np.inexact):
        if old != old:
            # int arrays do not contain NaN
            return
        if int(old) != old:
            raise ValueError("Cannot safely cast `old` to int.")
        if int(new) != new:
            raise ValueError("Cannot safely cast `new` to int.")
    if old != old:
        mask = np.isnan(arr)
    else:
        mask = arr == old
    np.putmask(arr, mask, new)

def anynan(arr, axis=None):
    "Slow check for Nans used for unaccelerated ndim/dtype combinations."
    return np.isnan(arr).any(axis)

def allnan(arr, axis=None):
    "Slow check for all Nans used for unaccelerated ndim/dtype combinations."
    return np.isnan(arr).all(axis)


def nanequal(arr1, arr2, axis=None):
    "Slow check for equality that ignores NaNs"
    if axis is None:
        nans = np.isnan(arr1) | np.isnan(arr2)
        return np.array_equal(arr1[~nans], arr2[~nans])
    if arr1.size == 0:
        if axis < 0:
            axis += arr1.ndim
        return np.ones(arr1.shape[:axis]+arr1.shape[axis+1:], np.bool)
    if arr1.size == 1:
        return arr1 == arr2 or arr1 != arr1 or arr2 != arr2
    return np.apply_along_axis(lambda x:nanequal(x["f0"], x["f1"]), axis,
                               np.core.records.fromarrays([arr1, arr2]))


def bincount(arr, max_val, weights=None, mask=None):
    "Slow bincount"
    if arr.ndim == 1:
        out = np.zeros((max_val+1, ), float)
        nans = 0.0
        if mask is None or mask[0]:
            for i, ai in enumerate(arr):
                if ai != ai:
                    nans += 1 if weights is None else weights[i]
                    continue
                ain = int(ai+0.1)
                if abs(ain - ai) > 1e-6:
                    raise ValueError("%f is not integer" % ain)
                if ain < 0:
                    raise ValueError("negative value in bincount")
                if ain > max_val:
                    raise ValueError("value %i is greater than max_val (%i)" %
                                      (ain, max_val))
                out[ain] += 1 if weights is None else weights[i]
    elif arr.ndim == 2:
        out = np.zeros((arr.shape[1], max_val+1), float)
        nans = np.zeros((arr.shape[1], ), float)
        if sp.issparse(arr):
            indptr, indices, data = arr.indptr, arr.indices, arr.data
            for ri in range(arr.shape[0]):
                wt = 1 if weights is None else weights[ri]
                for i in range(indptr[ri], indptr[ri + 1]):
                    ci = indices[i]
                    if mask is None or mask[ci]:
                        out[ci, data[i]] += wt
        else:
            for i in range(arr.shape[1]):
                if mask is None or mask[i]:
                    out[i, :], nans[i] = bincount(arr[:, i], max_val, weights)

    else:
        raise ValueError("bincount expects 1- or 2-dimensional array")
    return out, nans


def contingency(arr, b, max_val, max_val2, weights=None, mask=None):
    raise NotImplemented("bottlechest.slow.contingency is not implemented yet")


def valuecount(arr):
    "slow valuecount"
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError("valuecount expects an array with shape (2, N)")
    N = arr.shape[1]
    dst = 0
    for src in range(1, N):
        if math.isnan(arr[0, src]):
            break
        if arr[0, src] == arr[0, dst]:
            arr[1, dst] += arr[1, src]
        else:
            dst += 1
            arr[:, dst] = arr[:, src]
    return arr[:, :dst + 1]


def countnans(arr, weights=None, axis=None):
    if axis is None:
        if weights is None:
            return np.sum(np.isnan(arr))
        else:
            return np.sum(np.isnan(arr)*weights)
    else:
        if weights is not None:
            if arr.shape[axis] != len(weights):
                raise ValueError("shape of weights does not match the data")
            return np.apply_along_axis(lambda a: np.sum(np.isnan(a)*weights),
                                       axis, arr)
        else:
            return np.sum(np.isnan(arr), axis=axis)


def stats_object(arr, weights=None):
    pinf, minf = float("inf"), float("-inf")

    if arr.ndim == 1:
        if weights is None:
            nones = sum(np.equal(arr, None))
            return pinf, minf, 0, 0, nones, len(arr) - nones
        else:
            nones = sum(np.equal(arr, None) * weights)
            return pinf, minf, 0, 0, nones, sum(weights) - nones

    if sp.issparse(arr) and weights is not None:
        raise NotImplementedError("counting of missing values for"
            "weighted arrays of type 'object' is not implemented")

    y = np.zeros((arr.shape[1], 6), float)
    y[:, 0] = pinf
    y[:, 1] = minf
    if sp.issparse(arr):
        y[:, 4] = np.bincount(arr.indices, minlength=arr.shape[1])
    elif weights is None:
        y[:, 4] = np.sum(np.equal(arr, None), axis=0)
    else:
        y[:, 4] = np.sum(np.equal(arr, None) * weights[:, np.newaxis], axis=0)
    y[:, 5] = arr.shape[0] - y[:, 4]
    return y


def stats(arr, weights=None, compute_variance=False):
    if not 1 <= arr.ndim <= 2:
        raise ValueError("bottlechest.stats handles only 1-d and 2-d arrays")
    if arr.dtype == object:
        # can't compute min and max, but we can count 'nans'
        return stats_object(arr, weights)

    if arr.ndim == 1:
        a_min, a_max = np.nanmin(arr), np.nanmax(arr)
        if weights is None:
            nans = np.sum(np.isnan(arr))
            non_nans = len(arr) - nans
            mean = np.nansum(arr) / non_nans
            var = np.nansum((arr - mean) ** 2) / (non_nans - 1)
        else:
            tot_w = np.sum(weights)
            nans = np.sum(np.isnan(arr) * weights)
            non_nans = tot_w - nans
            mean = np.nansum(arr * weights) / non_nans
            var = np.nansum(weights * (arr - mean) ** 2)
            tot_w2 = np.sum((1 - np.isnan(arr)) * weights ** 2)
            d = non_nans ** 2 - tot_w2
            if d > 1e-6:
                var *= non_nans / d
        return a_min, a_max, mean, var, nans, non_nans

    if sp.issparse(arr):
        arr = arr.todense()
    y = np.zeros((arr.shape[1], 6), dtype=float)
    y[:, 0] = nanmin(arr, 0)
    y[:, 1] = nanmax(arr, 0)
    if weights:
        tot_w = np.sum(weights)
        y[:, 4] = countnans(arr, weights, 0)
        y[:, 5] = tot_w - y[:, 4]
        y[:, 2] = nanmean(arr * weights, 0) / y[:, 4]
        y[:, 3] = nansum(weights * (arr - y[:, 2]) ** 2, 0)
        tot_w2 = np.sum((1 - np.isnan(arr)) * weights ** 2)
        d = y[:, 5] ** 2 - tot_w2
        if d > 1e-6:
            y[:, 3] *= y[:, 5] / d
    else:
        y[:, 4] = countnans(arr, axis=0)
        y[:, 5] = arr.shape[0] - y[:, 4]
        y[:, 2] = nanmean(arr, 0) / y[:, 4]
        y[:, 3] = nansum((arr - y[:, 2]) ** 2, 0) / (y[:, 4] - 1)
    y[:, 2][np.isinf(y[:, 2])] = 0
    return y

# ---------------------------------------------------------------------------
#
# SciPy
#
# Local copy of scipy.stats functions to avoid (by popular demand) a SciPy
# dependency. The SciPy license is included in the Bottleneck license file,
# which is distributed with Bottleneck.
#
# Code taken from scipy trunk on Dec 16, 2010.
# nanmedian taken from scipy trunk on Dec 17, 2010.
# rankdata taken from scipy HEAD on Mar 16, 2011.

def scipy_nanmean(x, axis=0):
    """
    Compute the mean over the given axis ignoring nans.

    Parameters
    ----------
    x : ndarray
        Input array.
    axis : int, optional
        Axis along which the mean is computed. Default is 0, i.e. the
        first axis.

    Returns
    -------
    m : float
        The mean of `x`, ignoring nans.

    See Also
    --------
    nanstd, nanmedian

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.linspace(0, 4, 3)
    >>> a
    array([ 0.,  2.,  4.])
    >>> a[-1] = np.nan
    >>> stats.nanmean(a)
    1.0

    """
    x, axis = _chk_asarray(x,axis)
    x = x.copy()
    Norig = x.shape[axis]
    factor = 1.0-np.sum(np.isnan(x),axis)*1.0/Norig

    x[np.isnan(x)] = 0
    return np.mean(x,axis)/factor

def scipy_nanstd(x, axis=0, bias=False):
    """
    Compute the standard deviation over the given axis, ignoring nans.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or None, optional
        Axis along which the standard deviation is computed. Default is 0.
        If None, compute over the whole array `x`.
    bias : bool, optional
        If True, the biased (normalized by N) definition is used. If False
        (default), the unbiased definition is used.

    Returns
    -------
    s : float
        The standard deviation.

    See Also
    --------
    nanmean, nanmedian

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.arange(10, dtype=float)
    >>> a[1:3] = np.nan
    >>> np.std(a)
    nan
    >>> stats.nanstd(a)
    2.9154759474226504
    >>> stats.nanstd(a.reshape(2, 5), axis=1)
    array([ 2.0817,  1.5811])
    >>> stats.nanstd(a.reshape(2, 5), axis=None)
    2.9154759474226504

    """
    x, axis = _chk_asarray(x,axis)
    x = x.copy()
    Norig = x.shape[axis]

    Nnan = np.sum(np.isnan(x),axis)*1.0
    n = Norig - Nnan

    x[np.isnan(x)] = 0.
    m1 = np.sum(x,axis)/n

    if axis:
        d = (x - np.expand_dims(m1, axis))**2.0
    else:
        d = (x - m1)**2.0

    m2 = np.sum(d,axis)-(m1*m1)*Nnan
    if bias:
        m2c = m2 / n
    else:
        m2c = m2 / (n - 1.)
    return np.sqrt(m2c)

def _nanmedian(arr1d):  # This only works on 1d arrays
    """Private function for rank a arrays. Compute the median ignoring Nan.

    Parameters
    ----------
    arr1d : ndarray
        Input array, of rank 1.

    Results
    -------
    m : float
        The median.
    """
    cond = 1-np.isnan(arr1d)
    x = np.sort(np.compress(cond,arr1d,axis=-1))
    if x.size == 0:
        return np.nan
    return np.median(x)

# Feb 2011: patched nanmedian to handle nanmedian(a, 1) with a = np.ones((2,0))
def scipy_nanmedian(x, axis=0):
    """
    Compute the median along the given axis ignoring nan values.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int, optional
        Axis along which the median is computed. Default is 0, i.e. the
        first axis.

    Returns
    -------
    m : float
        The median of `x` along `axis`.

    See Also
    --------
    nanstd, nanmean

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.array([0, 3, 1, 5, 5, np.nan])
    >>> stats.nanmedian(a)
    array(3.0)

    >>> b = np.array([0, 3, 1, 5, 5, np.nan, 5])
    >>> stats.nanmedian(b)
    array(4.0)

    Example with axis:

    >>> c = np.arange(30.).reshape(5,6)
    >>> idx = np.array([False, False, False, True, False] * 6).reshape(5,6)
    >>> c[idx] = np.nan
    >>> c
    array([[  0.,   1.,   2.,  nan,   4.,   5.],
           [  6.,   7.,  nan,   9.,  10.,  11.],
           [ 12.,  nan,  14.,  15.,  16.,  17.],
           [ nan,  19.,  20.,  21.,  22.,  nan],
           [ 24.,  25.,  26.,  27.,  nan,  29.]])
    >>> stats.nanmedian(c, axis=1)
    array([  2. ,   9. ,  15. ,  20.5,  26. ])

    """
    x, axis = _chk_asarray(x, axis)
    if x.ndim == 0:
        return float(x.item())
    shape = list(x.shape)
    shape.pop(axis)
    if 0 in shape:
        x = np.empty(shape)
    else:
        x = x.copy()
        x = np.apply_along_axis(_nanmedian, axis, x)
        if x.ndim == 0:
            x = float(x.item())
    return x

def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis
    return a, outaxis

def fastsort(a):
    """
    Sort an array and provide the argsort.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    fastsort : ndarray of type int
        sorted indices into the original array

    """
    # TODO: the wording in the docstring is nonsense.
    it = np.argsort(a)
    as_ = a[it]
    return as_, it

def scipy_rankdata(a):
    """
    Ranks the data, dealing with ties appropriately.

    Equal values are assigned a rank that is the average of the ranks that
    would have been otherwise assigned to all of the values within that set.
    Ranks begin at 1, not 0.

    Parameters
    ----------
    a : array_like
        This array is first flattened.

    Returns
    -------
    rankdata : ndarray
         An array of length equal to the size of `a`, containing rank scores.

    Examples
    --------
    >>> stats.rankdata([0, 2, 2, 3])
    array([ 1. ,  2.5,  2.5,  4. ])

    """
    a = np.ravel(a)
    n = len(a)
    svec, ivec = fastsort(a)
    sumranks = 0
    dupcount = 0
    newarray = np.zeros(n, float)
    for i in range(n):
        sumranks += i
        dupcount += 1
        if i==n-1 or svec[i] != svec[i+1]:
            averank = sumranks / float(dupcount) + 1
            for j in range(i-dupcount+1,i+1):
                newarray[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newarray

def scipy_ss(a, axis=0):
    """
    Squares each element of the input array, and returns the square(s) of that.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        The axis along which to calculate. If None, use whole array.
        Default is 0, i.e. along the first axis.

    Returns
    -------
    ss : ndarray
        The sum along the given axis for (a**2).

    See also
    --------
    square_of_sums : The square(s) of the sum(s) (the opposite of `ss`).

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.array([1., 2., 5.])
    >>> stats.ss(a)
    30.0

    And calculating along an axis:

    >>> b = np.array([[1., 2., 5.], [2., 5., 6.]])
    >>> stats.ss(b, axis=1)
    array([ 30., 65.])

    """
    a, axis = _chk_asarray(a, axis)

    if 'int' in str(a.dtype):
        a = a.astype('int64')
    return np.sum(a*a, axis)


