import unittest
import sys
import bottlechest as bn
import numpy as np
import scipy.sparse as sp


class TestStats(unittest.TestCase):
    def test_simple_int(self):
        data = np.array([[0, 1, 1, 2, 0],
                         [1, 1, 1, 0, 0],
                         [0, 0, 3, 0, 0]], dtype=int)
        y = bn.stats(data)
        np.testing.assert_equal(y[:, 0], [0, 0, 1, 0, 0])
        np.testing.assert_equal(y[:, 1], [1, 1, 3, 2, 0])
        np.testing.assert_almost_equal(y[:, 2], [1/3, 2/3, 5/3, 2/3, 0])
        np.testing.assert_equal(y[:, 3], [0, 0, 0, 0, 0])
        np.testing.assert_equal(y[:, 4], [0, 0, 0, 0, 0])
        np.testing.assert_equal(y[:, 5], [3, 3, 3, 3, 3])

        y = bn.stats(data, None, True)
        np.testing.assert_equal(y[:, 0], [0, 0, 1, 0, 0])
        np.testing.assert_equal(y[:, 1], [1, 1, 3, 2, 0])
        np.testing.assert_almost_equal(y[:, 2], [1/3, 2/3, 5/3, 2/3, 0])
        np.testing.assert_almost_equal(y[:, 3], [1/3, 1/3, 4/3, 4/3, 0])
        np.testing.assert_equal(y[:, 4], [0, 0, 0, 0, 0])
        np.testing.assert_equal(y[:, 5], [3, 3, 3, 3, 3])

    def test_simple_float(self):
        nan = float("nan")
        data = np.array([[0, 1, 1, 2, 0],
                         [1, 1, 1, nan, 1],
                         [0, 0, 3, nan, nan]], dtype=float)
        y = bn.stats(data)
        np.testing.assert_equal(y[:, 0], [0, 0, 1, 2, 0])
        np.testing.assert_equal(y[:, 1], [1, 1, 3, 2, 1])
        np.testing.assert_almost_equal(y[:, 2], [1/3, 2/3, 5/3, 2, 0.5])
        np.testing.assert_equal(y[:, 3], [0, 0, 0, 0, 0])
        np.testing.assert_equal(y[:, 4], [0, 0, 0, 2, 1])
        np.testing.assert_equal(y[:, 5], [3, 3, 3, 1, 2])

        y = bn.stats(data, None, True)
        np.testing.assert_equal(y[:, 0], [0, 0, 1, 2, 0])
        np.testing.assert_equal(y[:, 1], [1, 1, 3, 2, 1])
        np.testing.assert_almost_equal(y[:, 2], [1/3, 2/3, 5/3, 2, 0.5])
        np.testing.assert_almost_equal(y[:, 3], [1/3, 1/3, 4/3, 0, 1/2])
        np.testing.assert_equal(y[:, 4], [0, 0, 0, 2, 1])
        np.testing.assert_equal(y[:, 5], [3, 3, 3, 1, 2])

    def test_simple_float_allnan(self):
        nan = float("nan")
        data = np.array([[0, 1, 1, nan, 0],
                         [1, 1, 1, nan, 1],
                         [0, 0, 3, nan, nan]], dtype=float)
        y = bn.stats(data)
        np.testing.assert_equal(y[3, :],
            [float("inf"), float("-inf"), 0, 0, 3, 0])

        y = bn.stats(data, None, True)
        np.testing.assert_equal(y[3, :],
            [float("inf"), float("-inf"), 0, 0, 3, 0])

    def test_weighted_int(self):
        data = np.array([[0, 1, 1, 2, 1],
                         [1, 1, 1, 0, 1],
                         [0, 0, 3, 0, 0]], dtype=int)
        y = bn.stats(data, np.array([1, 2, 3], dtype=float))
        np.testing.assert_equal(y[:, 0], [0, 0, 1, 0, 0])
        np.testing.assert_equal(y[:, 1], [1, 1, 3, 2, 1])
        np.testing.assert_almost_equal(y[:, 2], [2/6, 0.5, 2, 2/6, 0.5])
        np.testing.assert_equal(y[:, 3], [0, 0, 0, 0, 0])
        np.testing.assert_equal(y[:, 4], [0, 0, 0, 0, 0])
        np.testing.assert_equal(y[:, 5], [6, 6, 6, 6, 6])

        y = bn.stats(data, np.array([1, 2, 3], dtype=float), True)
        np.testing.assert_equal(y[:, 0], [0, 0, 1, 0, 0])
        np.testing.assert_equal(y[:, 1], [1, 1, 3, 2, 1])
        np.testing.assert_almost_equal(y[:, 2], [2/6, 0.5, 2, 2/6, 0.5])
        np.testing.assert_almost_equal(
            y[:, 3], [(1/9 + 2 * 4/9 + 3 * 1/9) * 6 / 22,
                      0.25 * 6 * 6 / 22,
                      6 * 6 / 22,
                      (25/9 + 2 * 1/9 + 3 * 1/9) * 6 / 22,
                      0.25 * 6 * 6 / 22])
        np.testing.assert_equal(y[:, 4], [0, 0, 0, 0, 0])
        np.testing.assert_equal(y[:, 5], [6, 6, 6, 6, 6])

    def test_weighted_float(self):
        nan = float("nan")
        data = np.array([[0, 1, 1, 2, 0],
                         [1, 1, 1, nan, 1],
                         [0, 0, 3, nan, nan]], dtype=float)
        y = bn.stats(data, np.array([1, 2, 3], dtype=float))
        np.testing.assert_equal(y[:, 0], [0, 0, 1, 2, 0])
        np.testing.assert_equal(y[:, 1], [1, 1, 3, 2, 1])
        np.testing.assert_almost_equal(y[:, 2], [2/6, 0.5, 2, 2, 2/3])
        np.testing.assert_equal(y[:, 3], [0, 0, 0, 0, 0])
        np.testing.assert_equal(y[:, 4], [0, 0, 0, 5, 3])
        np.testing.assert_equal(y[:, 5], [6, 6, 6, 1, 3])

        y = bn.stats(data, np.array([1, 2, 3], dtype=float), True)
        np.testing.assert_equal(y[:, 0], [0, 0, 1, 2, 0])
        np.testing.assert_equal(y[:, 1], [1, 1, 3, 2, 1])
        np.testing.assert_almost_equal(y[:, 2], [2/6, 0.5, 2, 2, 2/3])
        np.testing.assert_almost_equal(
            y[:, 3], [(1/9 + 2 * 4/9 + 3 * 1/9) * 6 / 22,
                      0.25 * 6 * 6 / 22,
                      6 * 6 / 22,
                      0,
                      (4/9 + 2 * 1/9) * 3 / 4])
        np.testing.assert_equal(y[:, 4], [0, 0, 0, 5, 3])
        np.testing.assert_equal(y[:, 5], [6, 6, 6, 1, 3])

    # 0 1 2 3
    #--------
    # 1 1 2
    # 2
    #   1 3
    def test_sparse_int(self):
        data = np.array([1, 1, 2, 2, 1, 3])
        indptr = [0, 3, 4, 6]
        indices = [0, 1, 2, 0, 1, 2]
        a = sp.csr_matrix((data, indices, indptr), shape=(3, 4))

        y = bn.stats(a)
        np.testing.assert_equal(y[:, 0], [1, 1, 2, float("inf")])
        np.testing.assert_equal(y[:, 1], [2, 1, 3, float("-inf")])
        np.testing.assert_almost_equal(y[:, 2], [1.5, 1, 2.5, 0])
        np.testing.assert_equal(y[:, 3], [0, 0, 0, 0])
        np.testing.assert_equal(y[:, 4], [1, 1, 1, 3])
        np.testing.assert_equal(y[:, 5], [2, 2, 2, 0])

        y = bn.stats(a, None, True)
        np.testing.assert_equal(y[:, 0], [1, 1, 2, float("inf")])
        np.testing.assert_equal(y[:, 1], [2, 1, 3, float("-inf")])
        np.testing.assert_almost_equal(y[:, 2], [1.5, 1, 2.5, 0])
        np.testing.assert_equal(y[:, 3], [0.5, 0, 0.5, 0])
        np.testing.assert_equal(y[:, 4], [1, 1, 1, 3])
        np.testing.assert_equal(y[:, 5], [2, 2, 2, 0])

    def test_sparse_float(self):
        data = np.array([1, 1, 2, 2, 1, 3], dtype=float)
        indptr = [0, 3, 4, 6]
        indices = [0, 1, 2, 0, 1, 2]
        a = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
        y = bn.stats(a)
        np.testing.assert_equal(y[:, 0], [1, 1, 2, float("inf")])
        np.testing.assert_equal(y[:, 1], [2, 1, 3, float("-inf")])
        np.testing.assert_almost_equal(y[:, 2], [1.5, 1, 2.5, 0])
        np.testing.assert_equal(y[:, 3], [0, 0, 0, 0])
        np.testing.assert_equal(y[:, 4], [1, 1, 1, 3])
        np.testing.assert_equal(y[:, 5], [2, 2, 2, 0])

        y = bn.stats(a, None, True)
        np.testing.assert_equal(y[:, 0], [1, 1, 2, float("inf")])
        np.testing.assert_equal(y[:, 1], [2, 1, 3, float("-inf")])
        np.testing.assert_almost_equal(y[:, 2], [1.5, 1, 2.5, 0])
        np.testing.assert_almost_equal(y[:, 3], [0.5, 0, 0.5, 0])
        np.testing.assert_equal(y[:, 4], [1, 1, 1, 3])
        np.testing.assert_equal(y[:, 5], [2, 2, 2, 0])

    # 0 1 2 3  w
    #--------
    # 1 1 2    1
    # 2        2
    #   1 3    3
    def test_sparse_weight_int(self):
        data = np.array([1, 1, 2, 2, 1, 3])
        indptr = [0, 3, 4, 6]
        indices = [0, 1, 2, 0, 1, 2]
        a = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
        y = bn.stats(a, [1, 2, 3])
        np.testing.assert_equal(y[:, 0], [1, 1, 2, float("inf")])
        np.testing.assert_equal(y[:, 1], [2, 1, 3, float("-inf")])
        np.testing.assert_almost_equal(y[:, 2], [5/3, 1, 2.75, 0])
        np.testing.assert_equal(y[:, 3], [0, 0, 0, 0])
        np.testing.assert_equal(y[:, 4], [3, 2, 2, 6])
        np.testing.assert_equal(y[:, 5], [3, 4, 4, 0])

        y = bn.stats(a, [1, 2, 3], True)
        np.testing.assert_equal(y[:, 0], [1, 1, 2, float("inf")])
        np.testing.assert_equal(y[:, 1], [2, 1, 3, float("-inf")])
        np.testing.assert_almost_equal(y[:, 2], [5/3, 1, 2.75, 0])
        np.testing.assert_almost_equal(y[:, 3], [(4/9 + 2 * 1/9) * 3/4, 0,
                                                 (9/16 + 3 * 1/16) * 4/6, 0])
        np.testing.assert_equal(y[:, 4], [3, 2, 2, 6])
        np.testing.assert_equal(y[:, 5], [3, 4, 4, 0])

    def test_sparse_weight_float(self):
        data = np.array([1, 1, 2, 2, 1, 3], dtype=float)
        indptr = [0, 3, 4, 6]
        indices = [0, 1, 2, 0, 1, 2]
        a = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
        y = bn.stats(a, [1, 2, 3])
        np.testing.assert_equal(y[:, 0], [1, 1, 2, float("inf")])
        np.testing.assert_equal(y[:, 1], [2, 1, 3, float("-inf")])
        np.testing.assert_almost_equal(y[:, 2], [5/3, 1, 2.75, 0])
        np.testing.assert_equal(y[:, 3], [0, 0, 0, 0])
        np.testing.assert_equal(y[:, 4], [3, 2, 2, 6])
        np.testing.assert_equal(y[:, 5], [3, 4, 4, 0])

        y = bn.stats(a, [1, 2, 3], True)
        np.testing.assert_equal(y[:, 0], [1, 1, 2, float("inf")])
        np.testing.assert_equal(y[:, 1], [2, 1, 3, float("-inf")])
        np.testing.assert_almost_equal(y[:, 2], [5/3, 1, 2.75, 0])
        np.testing.assert_almost_equal(y[:, 3], [(4/9 + 2 * 1/9) * 3/4, 0,
                                                 (9/16 + 3 * 1/16) * 4/6, 0])
        np.testing.assert_equal(y[:, 4], [3, 2, 2, 6])
        np.testing.assert_equal(y[:, 5], [3, 4, 4, 0])


    def test_int_1d(self):
        data = np.array([0, 1, 1, 2, 0])
        a_min, a_max, mean, var, nans, non_nans = bn.stats(data)
        self.assertEqual(a_min, 0)
        self.assertEqual(a_max, 2)
        self.assertAlmostEqual(mean, 4/5)
        self.assertEqual(var, 0)
        self.assertEqual(nans, 0)
        self.assertEqual(non_nans, 5)

        a_min, a_max, mean, var, nans, non_nans = bn.stats(data, None, True)
        self.assertEqual(a_min, 0)
        self.assertEqual(a_max, 2)
        self.assertAlmostEqual(mean, 4/5)
        self.assertAlmostEqual(var, (16 + 1 + 1 + 36 + 16) / 25 / 4)
        self.assertEqual(nans, 0)
        self.assertEqual(non_nans, 5)

    def test_float_1d(self):
        data = np.array([0, 1, float("nan"), 2, 0], dtype="float")
        a_min, a_max, mean, var, nans, non_nans = bn.stats(data)
        self.assertEqual(a_min, 0)
        self.assertEqual(a_max, 2)
        self.assertAlmostEqual(mean, 3/4)
        self.assertEqual(var, 0)
        self.assertEqual(nans, 1)
        self.assertEqual(non_nans, 4)

        a_min, a_max, mean, var, nans, non_nans = bn.stats(data, None, True)
        self.assertEqual(a_min, 0)
        self.assertEqual(a_max, 2)
        self.assertAlmostEqual(mean, 3/4)
        self.assertEqual(var, (9/16 + 1/16 + 25/16 + 9/16) / 3)
        self.assertEqual(nans, 1)
        self.assertEqual(non_nans, 4)

    def test_int_weight_1d(self):
        data = np.array([0, 1, 1, 2, 0])
        a_min, a_max, mean, var, nans, non_nans = \
            bn.stats(data, [1, 2, 3, 4, 5])
        self.assertEqual(a_min, 0)
        self.assertEqual(a_max, 2)
        self.assertAlmostEqual(mean, 13/15)
        self.assertEqual(var, 0)
        self.assertEqual(nans, 0)
        self.assertEqual(non_nans, 15)

        a_min, a_max, mean, var, nans, non_nans = \
            bn.stats(data, [1, 2, 3, 4, 5], True)
        self.assertEqual(a_min, 0)
        self.assertEqual(a_max, 2)
        self.assertAlmostEqual(mean, 13/15)
        self.assertEqual(var, (169 + 2*4 + 3*4 + 4*17**2 + 5*169) / 225 * 15/170)
        self.assertEqual(nans, 0)
        self.assertEqual(non_nans, 15)

    def test_float_weight_1d(self):
        data = np.array([0, 1, float("nan"), 2, 0], dtype="float")
        a_min, a_max, mean, var, nans, non_nans = \
            bn.stats(data, [1, 2, 3, 4, 5])
        self.assertEqual(a_min, 0)
        self.assertEqual(a_max, 2)
        self.assertAlmostEqual(mean, 10/12)
        self.assertEqual(var, 0)
        self.assertEqual(nans, 3)
        self.assertEqual(non_nans, 12)

        a_min, a_max, mean, var, nans, non_nans = \
            bn.stats(data, [1, 2, 3, 4, 5], True)
        self.assertEqual(a_min, 0)
        self.assertEqual(a_max, 2)
        self.assertAlmostEqual(mean, 10/12)
        self.assertAlmostEqual(var, (100 + 2*4 + 4*196 + 5*100) / 144 * 12/98)
        self.assertEqual(nans, 3)
        self.assertEqual(non_nans, 12)

if __name__ == "__main__":
    unittest.main()
