import unittest
import sys
import bottlechest as bn
import numpy as np
import scipy.sparse as sp

class TestContingency(unittest.TestCase):
    def test_1d_int(self):
        data = np.array([0, 1, 1, 2, 1])
        bb = [0, 1, 1, 0, 0]
        for b in [bb, np.array(bb, dtype=np.int8), np.array(bb, dtype=float)]:
            counts, nans = bn.contingency(data, b, 2, 1)
            np.testing.assert_almost_equal(counts, [[1, 1, 1], [0, 2, 0]])
            np.testing.assert_almost_equal(nans, np.zeros(2))

    def test_1d_float(self):
        nan = float("nan")
        data = np.array([0, 1, nan, 2, 1], dtype=float)
        bb = [0, 1, 1, 0, 0]
        for b in [bb, np.array(bb, dtype=np.int8), np.array(bb, dtype=float)]:
            counts, nans = bn.contingency(data, b, 2, 1)
            np.testing.assert_almost_equal(counts, [[1, 1, 1], [0, 1, 0]])
            np.testing.assert_almost_equal(nans, [0, 1])

    def test_1d_mask_int(self):
        data = np.array([0, 1, 1, 2, 1])
        bb = [0, 1, 1, 0, 0]
        counts, nans = bn.contingency(data, bb, 2, 1, mask=[1])
        np.testing.assert_almost_equal(counts, [[1, 1, 1], [0, 2, 0]])
        np.testing.assert_almost_equal(nans, [0, 0])

        counts, nans = bn.contingency(data, bb, 2, 1, mask=[0])
        np.testing.assert_almost_equal(counts, np.zeros((2, 3)))
        np.testing.assert_almost_equal(nans, [0, 0])

    def test_1d_mask_float(self):
        nan = float("nan")
        data = np.array([0, 1, nan, 2, 1], dtype=float)
        bb = [0, 1, 1, 0, 0]
        counts, nans = bn.contingency(data, bb, 2, 1, mask=[1])
        np.testing.assert_almost_equal(counts, [[1, 1, 1], [0, 1, 0]])
        np.testing.assert_almost_equal(nans, [0, 1])

        counts, nans = bn.contingency(data, bb, 2, 1, mask=[0])
        np.testing.assert_almost_equal(counts, np.zeros((2, 3)))
        np.testing.assert_almost_equal(nans, [0, 0])

    def test_1d_weighted_int(self):
        data = np.array([0, 1, 1, 2, 1])
        bb = [0, 1, 1, 0, 0]
        counts, nans = bn.contingency(data, bb, 2, 1, weights=[1, 2, 3, 4, 5])
        np.testing.assert_almost_equal(counts, [[1, 5, 3], [0, 3, 0]])
        np.testing.assert_almost_equal(nans, np.zeros(2))

    def test_1d_weighted_int(self):
        nan = float("nan")
        data = np.array([0, 1, nan, 2, 1], dtype=float)
        bb = [0, 1, 1, 0, 0]
        for b in [bb, np.array(bb, dtype=np.int8), np.array(bb, dtype=float)]:
            counts, nans = bn.contingency(data, b, 2, 1, weights=[1, 2, 3, 4, 5])
            np.testing.assert_almost_equal(counts, [[1, 5, 4], [0, 2, 0]])
            np.testing.assert_almost_equal(nans, [0, 3])

    def test_simple_int(self):
        data = np.array([[0, 1, 1, 2, 1],
                         [1, 1, 1, 0, 1],
                         [0, 0, 3, 0, 0]], dtype=int)
        for b in [
                np.array([1, 0, 1], dtype=np.int8),
                np.array([1, 0, 1], dtype=float),
                [1, 0, 1]]:
            counts, nans = bn.contingency(data, b, 3, 1)
            np.testing.assert_almost_equal(counts[0], [[0, 1, 0, 0], [2, 0, 0, 0]])
            np.testing.assert_almost_equal(counts[1], [[0, 1, 0, 0], [1, 1, 0, 0]])
            np.testing.assert_almost_equal(counts[2], [[0, 1, 0, 0], [0, 1, 0, 1]])
            np.testing.assert_almost_equal(counts[3], [[1, 0, 0, 0], [1, 0, 1, 0]])
            np.testing.assert_almost_equal(counts[4], [[0, 1, 0, 0], [1, 1, 0, 0]])
            np.testing.assert_almost_equal(nans, np.zeros((5, 2)))


    def test_simple_float(self):
        nan = float("nan")
        data = np.array([[0, 1, 1, 2, 1],
                         [1, 1, 1, nan, 1],
                         [0, 0, 3, nan, nan]], dtype=float)

        counts, nans = bn.contingency(data, [1, 0, 1], 3, 1)
        np.testing.assert_almost_equal(counts[0], [[0, 1, 0, 0], [2, 0, 0, 0]])
        np.testing.assert_almost_equal(counts[1], [[0, 1, 0, 0], [1, 1, 0, 0]])
        np.testing.assert_almost_equal(counts[2], [[0, 1, 0, 0], [0, 1, 0, 1]])
        np.testing.assert_almost_equal(counts[3], [[0, 0, 0, 0], [0, 0, 1, 0]])
        np.testing.assert_almost_equal(counts[4], [[0, 1, 0, 0], [0, 1, 0, 0]])
        np.testing.assert_almost_equal(nans, [[0, 0], [0, 0], [0, 0], [1, 1], [0, 1]])

    def test_weighted_int(self):
        data = np.array([[0, 1, 1, 2, 1],
                         [1, 1, 1, 0, 1],
                         [0, 0, 3, 0, 0]], dtype=int)
        counts, nans = bn.contingency(data, [1, 0, 1], 3, 1, weights=[1, 2, 3])
        np.testing.assert_almost_equal(counts[0], [[0, 2, 0, 0], [4, 0, 0, 0]])
        np.testing.assert_almost_equal(counts[1], [[0, 2, 0, 0], [3, 1, 0, 0]])
        np.testing.assert_almost_equal(counts[2], [[0, 2, 0, 0], [0, 1, 0, 3]])
        np.testing.assert_almost_equal(counts[3], [[2, 0, 0, 0], [3, 0, 1, 0]])
        np.testing.assert_almost_equal(counts[4], [[0, 2, 0, 0], [3, 1, 0, 0]])
        np.testing.assert_almost_equal(nans, np.zeros((5, 2)))


    def test_weighted_float(self):
        nan = float("nan")
        data = np.array([[0, 1, 1, 2, 1],
                         [1, 1, 1, nan, 1],
                         [0, 0, 3, nan, nan]], dtype=float)

        counts, nans = bn.contingency(data, [1, 0, 1], 3, 1, weights=[1, 2, 3])
        np.testing.assert_almost_equal(counts[0], [[0, 2, 0, 0], [4, 0, 0, 0]])
        np.testing.assert_almost_equal(counts[1], [[0, 2, 0, 0], [3, 1, 0, 0]])
        np.testing.assert_almost_equal(counts[2], [[0, 2, 0, 0], [0, 1, 0, 3]])
        np.testing.assert_almost_equal(counts[3], [[0, 0, 0, 0], [0, 0, 1, 0]])
        np.testing.assert_almost_equal(counts[4], [[0, 2, 0, 0], [0, 1, 0, 0]])
        np.testing.assert_almost_equal(nans, [[0, 0], [0, 0], [0, 0], [2, 3], [0, 3]])


    def test_mask_int(self):
        data = np.array([[0, 1, 1, 2, 1],
                         [1, 1, 1, 0, 1],
                         [0, 0, 3, 0, 0]], dtype=int)
        for b in [
                np.array([1, 0, 1], dtype=np.int8),
                np.array([1, 0, 1], dtype=float),
                [1, 0, 1]]:
            counts, nans = bn.contingency(data, b, 3, 1, mask=[1, 1, 0, 0, 1])
            np.testing.assert_almost_equal(counts[0], [[0, 1, 0, 0], [2, 0, 0, 0]])
            np.testing.assert_almost_equal(counts[1], [[0, 1, 0, 0], [1, 1, 0, 0]])
            np.testing.assert_almost_equal(counts[2], np.zeros((2, 4)))
            np.testing.assert_almost_equal(counts[3], np.zeros((2, 4)))
            np.testing.assert_almost_equal(counts[4], [[0, 1, 0, 0], [1, 1, 0, 0]])
            np.testing.assert_almost_equal(nans, np.zeros((5, 2)))


    def test_mask_float(self):
        nan = float("nan")
        data = np.array([[0, 1, 1, 2, 1],
                         [1, 1, 1, nan, 1],
                         [0, 0, 3, nan, nan]], dtype=float)

        counts, nans = bn.contingency(data, [1, 0, 1], 3, 1, mask=[1, 1, 0, 0, 1])
        np.testing.assert_almost_equal(counts[0], [[0, 1, 0, 0], [2, 0, 0, 0]])
        np.testing.assert_almost_equal(counts[1], [[0, 1, 0, 0], [1, 1, 0, 0]])
        np.testing.assert_almost_equal(counts[2], np.zeros((2, 4)))
        np.testing.assert_almost_equal(counts[3], np.zeros((2, 4)))
        np.testing.assert_almost_equal(counts[4], [[0, 1, 0, 0], [0, 1, 0, 0]])
        np.testing.assert_almost_equal(nans, [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]])

    def test_mask_weighted_int(self):
        data = np.array([[0, 1, 1, 2, 1],
                         [1, 1, 1, 0, 1],
                         [0, 0, 3, 0, 0]], dtype=int)
        counts, nans = bn.contingency(data, [1, 0, 1], 3, 1,
                                      weights=[1, 2, 3], mask=[1, 1, 0, 0, 1])
        np.testing.assert_almost_equal(counts[0], [[0, 2, 0, 0], [4, 0, 0, 0]])
        np.testing.assert_almost_equal(counts[1], [[0, 2, 0, 0], [3, 1, 0, 0]])
        np.testing.assert_almost_equal(counts[2], np.zeros((2, 4)))
        np.testing.assert_almost_equal(counts[3], np.zeros((2, 4)))
        np.testing.assert_almost_equal(counts[4], [[0, 2, 0, 0], [3, 1, 0, 0]])
        np.testing.assert_almost_equal(nans, np.zeros((5, 2)))


    def test_mask_weighted_float(self):
        nan = float("nan")
        data = np.array([[0, 1, 1, 2, 1],
                         [1, 1, 1, nan, 1],
                         [0, 0, 3, nan, nan]], dtype=float)

        counts, nans = bn.contingency(data, [1, 0, 1], 3, 1,
                                      weights=[1, 2, 3], mask=[1, 1, 0, 0, 1])
        np.testing.assert_almost_equal(counts[0], [[0, 2, 0, 0], [4, 0, 0, 0]])
        np.testing.assert_almost_equal(counts[1], [[0, 2, 0, 0], [3, 1, 0, 0]])
        np.testing.assert_almost_equal(counts[2], np.zeros((2, 4)))
        np.testing.assert_almost_equal(counts[3], np.zeros((2, 4)))
        np.testing.assert_almost_equal(counts[4], [[0, 2, 0, 0], [0, 1, 0, 0]])
        np.testing.assert_almost_equal(nans, [[0, 0], [0, 0], [0, 0], [0, 0], [0, 3]])



    def test_sparse_int(self):
        data = np.array([1, 1, 2, 2, 1, 3])
        indptr = [0, 3, 4, 6]
        indices = [0, 1, 2, 0, 1, 2]
        a = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
        counts, nans = bn.contingency(a, [1, 0, 1], 3, 1)
        np.testing.assert_almost_equal(counts[0], [[0, 0, 1, 0], [0, 1, 0, 0]])
        np.testing.assert_almost_equal(counts[1], [[0, 0, 0, 0], [0, 2, 0, 0]])
        np.testing.assert_almost_equal(counts[2], [[0, 0, 0, 0], [0, 0, 1, 1]])
        np.testing.assert_almost_equal(counts[3], np.zeros((2, 4)))
        np.testing.assert_almost_equal(nans, np.zeros((4, 2)))

    def test_sparse_float(self):
        data = np.array([1, 1, 2, 2, 1, 3], dtype=float)
        indptr = [0, 3, 4, 6]
        indices = [0, 1, 2, 0, 1, 2]
        a = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
        counts, nans = bn.contingency(a, [1, 0, 1], 3, 1)
        np.testing.assert_almost_equal(counts[0], [[0, 0, 1, 0], [0, 1, 0, 0]])
        np.testing.assert_almost_equal(counts[1], [[0, 0, 0, 0], [0, 2, 0, 0]])
        np.testing.assert_almost_equal(counts[2], [[0, 0, 0, 0], [0, 0, 1, 1]])
        np.testing.assert_almost_equal(counts[3], np.zeros((2, 4)))
        np.testing.assert_almost_equal(nans, np.zeros((4, 2)))

    def test_sparse_weight_int(self):
        data = np.array([1, 1, 2, 2, 1, 3])
        indptr = [0, 3, 4, 6]
        indices = [0, 1, 2, 0, 1, 2]
        a = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
        counts, nans = bn.contingency(a, [1, 0, 1], 3, 1, weights=[1, 2, 3])
        np.testing.assert_almost_equal(counts[0], [[0, 0, 2, 0], [0, 1, 0, 0]])
        np.testing.assert_almost_equal(counts[1], [[0, 0, 0, 0], [0, 4, 0, 0]])
        np.testing.assert_almost_equal(counts[2], [[0, 0, 0, 0], [0, 0, 1, 3]])
        np.testing.assert_almost_equal(counts[3], np.zeros((2, 4)))

    def test_sparse_weight_float(self):
        data = np.array([1, 1, 2, 2, 1, 3], dtype=float)
        indptr = [0, 3, 4, 6]
        indices = [0, 1, 2, 0, 1, 2]
        a = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
        counts, nans = bn.contingency(a, [1, 0, 1], 3, 1, weights=[1, 2, 3])
        np.testing.assert_almost_equal(counts[0], [[0, 0, 2, 0], [0, 1, 0, 0]])
        np.testing.assert_almost_equal(counts[1], [[0, 0, 0, 0], [0, 4, 0, 0]])
        np.testing.assert_almost_equal(counts[2], [[0, 0, 0, 0], [0, 0, 1, 3]])
        np.testing.assert_almost_equal(counts[3], np.zeros((2, 4)))

    def test_sparse_mask_int(self):
        data = np.array([1, 1, 2, 2, 1, 3])
        indptr = [0, 3, 4, 6]
        indices = [0, 1, 2, 0, 1, 2]
        a = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
        counts, nans = bn.contingency(a, [1, 0, 1], 3, 1, mask=[1, 0, 0, 1])
        np.testing.assert_almost_equal(counts[0], [[0, 0, 1, 0], [0, 1, 0, 0]])
        np.testing.assert_almost_equal(counts[1], np.zeros((2, 4)))
        np.testing.assert_almost_equal(counts[2], np.zeros((2, 4)))
        np.testing.assert_almost_equal(counts[3], np.zeros((2, 4)))

    def test_sparse_mask_float(self):
        data = np.array([1, 1, 2, 2, 1, 3], dtype=float)
        indptr = [0, 3, 4, 6]
        indices = [0, 1, 2, 0, 1, 2]
        a = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
        counts, nans = bn.contingency(a, [1, 0, 1], 3, 1, mask=[1, 0, 0, 1])
        np.testing.assert_almost_equal(counts[0], [[0, 0, 1, 0], [0, 1, 0, 0]])
        np.testing.assert_almost_equal(counts[1], np.zeros((2, 4)))
        np.testing.assert_almost_equal(counts[2], np.zeros((2, 4)))
        np.testing.assert_almost_equal(counts[3], np.zeros((2, 4)))


    def test_sparse_mask_weight_int(self):
        data = np.array([1, 1, 2, 2, 1, 3])
        indptr = [0, 3, 4, 6]
        indices = [0, 1, 2, 0, 1, 2]
        a = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
        counts, nans = bn.contingency(a, [1, 0, 1], 3, 1,
                                      weights=[1, 2, 3], mask=[1, 0, 0, 1])
        np.testing.assert_almost_equal(counts[0], [[0, 0, 2, 0], [0, 1, 0, 0]])
        np.testing.assert_almost_equal(counts[1], np.zeros((2, 4)))
        np.testing.assert_almost_equal(counts[2], np.zeros((2, 4)))
        np.testing.assert_almost_equal(counts[3], np.zeros((2, 4)))

    def test_sparse_mask_weight_float(self):
        data = np.array([1, 1, 2, 2, 1, 3], dtype=float)
        indptr = [0, 3, 4, 6]
        indices = [0, 1, 2, 0, 1, 2]
        a = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
        counts, nans = bn.contingency(a, [1, 0, 1], 3, 1,
                                      weights=[1, 2, 3], mask=[1, 0, 0, 1])
        np.testing.assert_almost_equal(counts[0], [[0, 0, 2, 0], [0, 1, 0, 0]])
        np.testing.assert_almost_equal(counts[1], np.zeros((2, 4)))
        np.testing.assert_almost_equal(counts[2], np.zeros((2, 4)))
        np.testing.assert_almost_equal(counts[3], np.zeros((2, 4)))

if __name__ == "__main__":
    unittest.main()
