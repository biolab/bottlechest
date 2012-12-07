import unittest
import sys
sys.path.remove("/Users/janezdemsar/bottlebone")
import bottleneck as bn
import numpy as np
import scipy.sparse as sp


class TestBinCount(unittest.TestCase):
    def test_sparse_int(self):
        data = np.array([1, 1, 2, 2, 1, 3])
        indptr = [0, 3, 4, 6]
        indices = [0, 1, 2, 0, 1, 2]
        a = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
        counts, nans = bn.bincount(a, 3)
        np.testing.assert_almost_equal(counts, [[0, 1, 1, 0], [0, 2, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]])

    def test_sparse_float(self):
        data = np.array([1, 1, 2, 2, 1, 3], dtype=float)
        indptr = [0, 3, 4, 6]
        indices = [0, 1, 2, 0, 1, 2]
        a = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
        counts, nans = bn.bincount(a, 3)
        np.testing.assert_almost_equal(counts, [[0, 1, 1, 0], [0, 2, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
