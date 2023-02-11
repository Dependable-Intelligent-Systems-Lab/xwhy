import unittest
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from xwhy import smile_tabular

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
        
    def test_WasserstainLIME():
        X_input = np.array([[1, 2]])
        model = LinearRegression()
        num_perturb = 500
        kernel_width2 = 0.2

        X_lime, y_lime2, weights2, y_linmodel2, coef = smile_tabular.WasserstainLIME(X_input, model, num_perturb, kernel_width2)

        # check if X_lime has the correct shape
        assert X_lime.shape == (num_perturb, X_input.shape[1])

        # check if y_lime2 is a 1D numpy array
        assert y_lime2.ndim == 1

        # check if weights2 is a 1D numpy array
        assert weights2.ndim == 1

        # check if y_linmodel2 is a 1D numpy array
        assert y_linmodel2.ndim == 1

        # check if coef is a 1D numpy array
        assert coef.ndim == 1

        # check if the type of y_linmodel2 is bool
        assert isinstance(y_linmodel2[0], np.bool_)

        # check if the return values of the function are not None
        assert X_lime is not None
        assert y_lime2 is not None
        assert weights2 is not None
        assert y_linmodel2 is not None
        assert coef is not None

if __name__ == '__main__':
    unittest.main()
