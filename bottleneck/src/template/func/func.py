from __future__ import absolute_import

from bottleneck.src.template.template import template
import bottleneck.src.template.template as tempmod
import os.path

from .median import median
from .nanmedian import nanmedian
from .nansum import nansum
from .nanmean import nanmean
from .nanvar import nanvar
from .nanstd import nanstd
from .nanmin import nanmin
from .nanmax import nanmax
from .nanargmin import nanargmin
from .nanargmax import nanargmax
from .rankdata import rankdata
from .nanrankdata import nanrankdata
from .ss import ss
from .nn import nn
from .replace import replace
from .anynan import anynan
from .allnan import allnan
from .nanequal import nanequal
from .bincount import bincount
from .valuecount import valuecount
from .countnans import countnans
from .contingency import contingency
from .stats import stats

funcs = {}
funcs['median'] = median
funcs['nanmedian'] = nanmedian
funcs['nansum'] = nansum
funcs['nanmean'] = nanmean
funcs['nanvar'] = nanvar
funcs['nanstd'] = nanstd
funcs['nanmin'] = nanmin
funcs['nanmax'] = nanmax
funcs['nanargmin'] = nanargmin
funcs['nanargmax'] = nanargmax
funcs['rankdata'] = rankdata
funcs['nanrankdata'] = nanrankdata
funcs['ss'] = ss
funcs['nn'] = nn
funcs['replace'] = replace
funcs['anynan'] = anynan
funcs['allnan'] = allnan
funcs['nanequal'] = nanequal
funcs['bincount'] = bincount
funcs['valuecount'] = valuecount
funcs['countnans'] = countnans
funcs['contingency'] = contingency
funcs['stats'] = stats


header = """#cython: embedsignature=True

import numpy as np
cimport numpy as np
import scipy.sparse as sp
import cython
from numpy cimport NPY_INT8 as NPY_int8
from numpy cimport NPY_INT32 as NPY_int32
from numpy cimport NPY_INT64 as NPY_int64
from numpy cimport NPY_FLOAT32 as NPY_float32
from numpy cimport NPY_FLOAT64 as NPY_float64
from numpy cimport (PyArray_EMPTY, PyArray_TYPE, PyArray_NDIM,
                    PyArray_SIZE, PyArray_DIMS, import_array,
                    PyArray_ArgSort, NPY_QUICKSORT, NPY_CORDER,
                    PyArray_Ravel, PyArray_FillWithScalar, PyArray_Copy,
                    NPY_BOOL,
                    PyArray_ZEROS)

# NPY_INTP is missing from numpy.pxd in cython 0.14.1 and earlier
cdef extern from "numpy/arrayobject.h":
    cdef enum NPY_TYPES:
        NPY_intp "NPY_INTP"

import_array()
import bottleneck as bn

cdef double NAN = <double> np.nan

cdef np.int8_t MINint8 = np.iinfo(np.int8).min
cdef np.int32_t MINint32 = np.iinfo(np.int32).min
cdef np.int64_t MINint64 = np.iinfo(np.int64).min
cdef np.float32_t MINfloat32 = np.NINF
cdef np.float64_t MINfloat64 = np.NINF

cdef np.int8_t MAXint8 = np.iinfo(np.int8).max
cdef np.int32_t MAXint32 = np.iinfo(np.int32).max
cdef np.int64_t MAXint64 = np.iinfo(np.int64).max
cdef np.float32_t MAXfloat32 = np.inf
cdef np.float64_t MAXfloat64 = np.inf

int8 = np.dtype(np.int8)
int32 = np.dtype(np.int32)
int64 = np.dtype(np.int64)
float32 = np.dtype(np.float32)
float64 = np.dtype(np.float64)

if np.int_ == np.int32:
    NPY_int_ = NPY_int32
elif np.int_ == np.int64:
    NPY_int_ = NPY_int64
else:
    raise RuntimeError('Expecting default NumPy int to be 32 or 64 bit.')

cdef extern from "math.h":
    double sqrt(double x)

PARTSORT_ERR_MSG = "`n` (=%d) must be between 1 and %d, inclusive."

include "nanmax.pyx"
include "nanmin.pyx"
include "nansum.pyx"
include "nanmean.pyx"
include "nanstd.pyx"
include "nanvar.pyx"
include "median.pyx"
include "nanmedian.pyx"
include "nanargmin.pyx"
include "nanargmax.pyx"
include "rankdata.pyx"
include "nanrankdata.pyx"
include "ss.pyx"
include "nn.pyx"
include "replace.pyx"
include "anynan.pyx"
include "allnan.pyx"
include "nanequal.pyx"
include "bincount.pyx"
include "valuecount.pyx"
include "countnans.pyx"
include "contingency.pyx"
include "stats.pyx"
"""

def funcpyx(funcs=funcs, bits=None):
    for func in funcs:
        template(funcs[func], bits)
    template_path = os.path.dirname(tempmod.__file__)
    fid = open(os.path.join(template_path, '..', "func/%sbit/func.pyx") % str(bits), 'w')
    fid.write(header)
    fid.close()
