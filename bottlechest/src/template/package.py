from __future__ import absolute_import

from bottlechest.src.template.template import template
import bottlechest.src.template.template as tempmod
import os.path

from collections import defaultdict
import importlib

HEADER = """#cython: embedsignature=True

import numpy as np
cimport numpy as np
import scipy.sparse as sp
import cython
from numpy cimport NPY_INT as NPY_int
from numpy cimport NPY_INT8 as NPY_int8
from numpy cimport NPY_INT32 as NPY_int32
from numpy cimport NPY_INT64 as NPY_int64
from numpy cimport NPY_FLOAT as NPY_float
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
import bottlechest as bn

cdef double NAN = <double> np.nan

cdef np.int_t MINint = np.iinfo(np.int).min
cdef np.int8_t MINint8 = np.iinfo(np.int8).min
cdef np.int32_t MINint32 = np.iinfo(np.int32).min
cdef np.int64_t MINint64 = np.iinfo(np.int64).min
cdef np.float_t MINfloat = np.NINF
cdef np.float32_t MINfloat32 = np.NINF
cdef np.float64_t MINfloat64 = np.NINF

cdef np.int_t MAXint = np.iinfo(np.int).max
cdef np.int8_t MAXint8 = np.iinfo(np.int8).max
cdef np.int32_t MAXint32 = np.iinfo(np.int32).max
cdef np.int64_t MAXint64 = np.iinfo(np.int64).max
cdef np.float_t MAXfloat = np.inf
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

"""

def single(templatepy, bits):
    b = importlib.import_module("bottlechest.src.template.%s" % (templatepy))
    funcs = []
    for f in b.__all__:
        funcs.append(b.__dict__[f])
    template(funcs, bits, HEADER)

def package(package, bits=None):
 
    pyfolder = os.path.join(os.path.dirname(tempmod.__file__), package)

    pyfiles = set([ a[:-3] for a in os.listdir(pyfolder) if a.endswith(".py") \
        and a != "func.py" and a != "__init__.py" ])

    for a in sorted(list(pyfiles)):
        try:
            single(package + "." + a, bits)
        except ImportError:
            print("Could not import", a + ". Skipping.")
