import unittest
import sys
import bottlechest as bn
import numpy as np


class TestCountNans(unittest.TestCase):
    def test2d(self):
        a = np.ones((4, 5))
        for x, y in ((0, 0), (0, 1), (0, 2), (3, 1)):
            a[x, y] = float("nan")
        self.assertEqual(bn.countnans(a), 4)
        self.assertEqual(bn.slow.countnans(a), 4)
        np.testing.assert_array_equal(bn.countnans(a, axis=0), [1, 2, 1, 0, 0])
        np.testing.assert_array_equal(bn.slow.countnans(a, axis=0), [1, 2, 1, 0, 0])
        np.testing.assert_array_equal(bn.countnans(a, axis=1), [3, 0, 0, 1])
        np.testing.assert_array_equal(bn.slow.countnans(a, axis=1), [3, 0, 0, 1])

    def test2d_w(self):
        a = np.ones((4, 5))
        w = np.random.random((4, 5))
        for x, y in ((0, 0), (0, 1), (0, 2), (3, 1)):
            a[x, y] = float("nan")
        self.assertAlmostEqual(bn.countnans(a, weights=w),
                               bn.slow.countnans(a, weights=w))
        np.testing.assert_almost_equal(bn.countnans(a, weights=[0.1, 0.2, 0.3, 0.4], axis=0), [0.1, 0.5, 0.1, 0, 0])
        np.testing.assert_almost_equal(bn.slow.countnans(a, weights=[0.1, 0.2, 0.3, 0.4], axis=0), [0.1, 0.5, 0.1, 0, 0])
        np.testing.assert_almost_equal(bn.countnans(a, weights=[0.1, 0.2, 0.3, 0.4, 0.5], axis=1), [0.6, 0, 0, 0.2])
        np.testing.assert_almost_equal(bn.slow.countnans(a, weights=[0.1, 0.2, 0.3, 0.4, 0.5], axis=1), [0.6, 0, 0, 0.2])

    def test1d(self):
        a = np.ones(5)
        a[2] = a[4] = float("nan")
        self.assertEqual(bn.countnans(a), 2)
        self.assertEqual(bn.slow.countnans(a), 2)

    def test1d_w(self):
        a = np.ones(5)
        a[2] = a[4] = float("nan")
        self.assertAlmostEqual(bn.countnans(a, weights=[0.1, 0.2, 0.3, 0.4, 0.5]), 0.8)
        self.assertAlmostEqual(bn.slow.countnans(a, weights=[0.1, 0.2, 0.3, 0.4, 0.5]), 0.8)
