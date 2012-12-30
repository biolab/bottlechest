import unittest
import sys
if "/Users/janezdemsar/bottleneck" in sys.path:
    sys.path.remove("/Users/janezdemsar/bottleneck")
import bottleneck as bn
import numpy as np
import scipy.sparse as sp


class TestBinCount(unittest.TestCase):
    def test_simple_int(self):
        data = np.array([[0, 1, 1, 2, 1],
                         [1, 1, 1, 0, 1],
                         [0, 0, 3, 0, 0]], dtype=float)
        counts, nans = bn.bincount(data, 3)
        np.testing.assert_almost_equal(counts, [[2, 1, 0, 0], [1, 2, 0, 0], [0, 2, 0, 1], [2, 0, 1, 0], [1, 2, 0, 0]])
        np.testing.assert_almost_equal(nans, [0, 0, 0, 0, 0])

    def test_simple_float(self):
        nan = float("nan")
        data = np.array([[0, 1, 1, 2, 1],
                         [1, 1, 1, nan, 1],
                         [0, 0, 3, nan, nan]], dtype=float)
        counts, nans = bn.bincount(data, 3)
        np.testing.assert_almost_equal(counts, [[2, 1, 0, 0], [1, 2, 0, 0], [0, 2, 0, 1], [0, 0, 1, 0], [0, 2, 0, 0]])
        np.testing.assert_almost_equal(nans, [0, 0, 0, 2, 1])

    def test_weighted_int(self):
        data = np.array([[0, 1, 1, 2, 1],
                         [1, 1, 1, 0, 1],
                         [0, 0, 3, 0, 0]], dtype=float)
        counts, nans = bn.bincount(data, 3, np.array([1, 2, 3], dtype=float))
        print(counts)
        np.testing.assert_almost_equal(counts, [[4, 2, 0, 0], [3, 3, 0, 0], [0, 3, 0, 3], [5, 0, 1, 0], [3, 3, 0, 0]])
        np.testing.assert_almost_equal(nans, [0, 0, 0, 0, 0])

    def test_weighted_float(self):
        nan = float("nan")
        data = np.array([[0, 1, 1, 2, 1],
                         [1, 1, 1, nan, 1],
                         [0, 0, 3, nan, nan]], dtype=float)
        counts, nans = bn.bincount(data, 3, [1, 2, 3])
        np.testing.assert_almost_equal(counts, [[4, 2, 0, 0], [3, 3, 0, 0], [0, 3, 0, 3], [0, 0, 1, 0], [0, 3, 0, 0]])
        np.testing.assert_almost_equal(nans, [0, 0, 0, 5, 3])

    def test_mask_int(self):
        data = np.array([[0, 1, 1, 2, 1],
                         [1, 1, 1, 0, 1],
                         [0, 0, 3, 0, 0]], dtype=float)
        counts, nans = bn.bincount(data, 3, mask=np.array([1, 1, 0, 0, 1]))
        np.testing.assert_almost_equal(counts, [[2, 1, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 0, 0]])
        np.testing.assert_almost_equal(nans, [0, 0, 0, 0, 0])
        counts, nans = bn.bincount(data, 3, mask=np.array([1, 1, 0, 0, 1], dtype=np.int8))
        np.testing.assert_almost_equal(counts, [[2, 1, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 0, 0]])
        np.testing.assert_almost_equal(nans, [0, 0, 0, 0, 0])
        counts, nans = bn.bincount(data, 3, mask=np.array([1, 1, 0, 0, 1], dtype=np.int32))
        np.testing.assert_almost_equal(counts, [[2, 1, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 0, 0]])
        np.testing.assert_almost_equal(nans, [0, 0, 0, 0, 0])
        counts, nans = bn.bincount(data, 3, mask=np.array([1, 1, 0, 0, 1])==1)
        np.testing.assert_almost_equal(counts, [[2, 1, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 0, 0]])
        np.testing.assert_almost_equal(nans, [0, 0, 0, 0, 0])

    def test_mask_float(self):
        nan = float("nan")
        data = np.array([[0, 1, 1, 2, 1],
                         [1, 1, 1, nan, 1],
                         [0, 0, 3, nan, nan]], dtype=float)
        counts, nans = bn.bincount(data, 3, mask=np.array([1, 1, 0, 0, 1]))
        np.testing.assert_almost_equal(counts, [[2, 1, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0]])
        np.testing.assert_almost_equal(nans, [0, 0, 0, 0, 1])
        counts, nans = bn.bincount(data, 3, mask=np.array([1, 1, 0, 0, 1], dtype=np.int8))
        np.testing.assert_almost_equal(counts, [[2, 1, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0]])
        np.testing.assert_almost_equal(nans, [0, 0, 0, 0, 1])
        counts, nans = bn.bincount(data, 3, mask=np.array([1, 1, 0, 0, 1], dtype=np.int8))
        np.testing.assert_almost_equal(counts, [[2, 1, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0]])
        np.testing.assert_almost_equal(nans, [0, 0, 0, 0, 1])
        counts, nans = bn.bincount(data, 3, mask=np.array([1, 1, 0, 0, 1])==1)
        np.testing.assert_almost_equal(counts, [[2, 1, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0]])
        np.testing.assert_almost_equal(nans, [0, 0, 0, 0, 1])

    def test_weighted_mask_int(self):
        data = np.array([[0, 1, 1, 2, 1],
                         [1, 1, 1, 0, 1],
                         [0, 0, 3, 0, 0]], dtype=float)
        counts, nans = bn.bincount(data, 3, weights=[1, 2, 3], mask=np.array([1, 1, 0, 0, 1]))
        np.testing.assert_almost_equal(counts, [[4, 2, 0, 0], [3, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [3, 3, 0, 0]])
        np.testing.assert_almost_equal(nans, [0, 0, 0, 0, 0])

    def test_weighted_mask_float(self):
        nan = float("nan")
        data = np.array([[0, 1, 1, 2, 1],
                         [1, 1, 1, nan, 1],
                         [0, 0, 3, nan, nan]], dtype=float)
        counts, nans = bn.bincount(data, 3, weights=[1, 2, 3], mask=np.array([1, 1, 0, 0, 1]))
        np.testing.assert_almost_equal(counts, [[4, 2, 0, 0], [3, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 3, 0, 0]])
        np.testing.assert_almost_equal(nans, [0, 0, 0, 0, 3])

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


