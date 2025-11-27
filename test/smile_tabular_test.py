import unittest
import numpy as np
import random

from xwhy import smile_tabular

import unittest
import numpy as np

class TestWassersteinDist(unittest.TestCase):
    def test_distance_calculation(self):
        XX = np.random.normal(1, 1, 1000)
        YY = np.random.normal(3, 1, 1000)
        dist = smile_tabular.Wasserstein_Dist(XX, YY)
        self.assertAlmostEqual(dist, 1.996663761722729, delta=0.1)

    def test_input_vector_lengths(self):
        XX = np.random.normal(1, 1, 1000)
        YY = np.random.normal(2, 1, 2000)
        dist = smile_tabular.Wasserstein_Dist(XX, YY)
        self.assertAlmostEqual(dist, 1.0104185038625486, delta=0.1)

if __name__ == '__main__':
    unittest.main()