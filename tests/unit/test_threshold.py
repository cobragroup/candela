import unittest
import numpy as np
import pandas as pd

from src.candela.functions import simple_threshold


class TestThreshold(unittest.TestCase):
    def test_empty (self):
        """Test behaviour with empty input"""
        with self.assertRaises(AssertionError):
            result = simple_threshold([], threshold=2)[0]

    def test_not1D (self):
        """Test behaviour with 2D input"""
        with self.assertRaises(AssertionError):
            result = simple_threshold([[1,2,3],[1,2,4]], threshold=2)[0]

    def test_array(self):
        """
        Test that it can use array input
        """
        data = np.array([1, 1, 1, 4])
        result = simple_threshold(data, threshold=2)[0]
        self.assertEqual(result, 3)

    def test_list(self):
        """
        Test that it can use list input
        """
        data = [1, 1, 1, 4]
        result = simple_threshold(data, threshold=2)[0]
        self.assertEqual(result, 3)

    def test_series(self):
        """
        Test that it can use series input
        """
        data = pd.Series([1, 1, 1, 4], index=[101, 102, 103, 104], name="test")
        result = simple_threshold(data, threshold=2)[0]
        self.assertEqual(result, 3)

    def test_absolute(self):
        """
        Test that it works with absolute values
        """
        data = [1, 1, 1, -4]
        result = simple_threshold(data, threshold=2, absolute=True)[0]
        self.assertEqual(result, 3)

    def test_smaller(self):
        """
        Test that it works with selecting under threshold
        """
        data = [1, 1, 1, -1]
        result = simple_threshold(data, threshold=0, smaller=True)[0]
        self.assertEqual(result, 3)

    def test_number_of_anomalies(self):
        """
        Test that it works with specified number of anomalies
        """
        data = [1, 2, 3, 4]
        result = simple_threshold(data, number=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[-1], 3)

    def test_number_of_anomalies_zero(self):
        """
        Test that it works with specified number of anomalies
        """
        data = [1, 2, 3, 4]
        with self.assertRaises(AssertionError):
            result = simple_threshold(data, number=0)

    def test_fraction_of_anomalies(self):
        """
        Test that it works with specified fraction of anomalies
        """
        data = [1, 2, 3, 4]
        result = simple_threshold(data, fraction=0.5)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[-1], 3)

    def test_fraction_of_anomalies_zero(self):
        """
        Test that it works with specified fraction of anomalies
        """
        data = [1, 2, 3, 4]
        with self.assertRaises(AssertionError):
            result = simple_threshold(data, fraction=0.)
            
    def test_nan(self):
        """
        Test that it can use exclude NaNs
        """
        data = [1, 1, np.nan, 4]
        result = simple_threshold(data, threshold=2)[0]
        self.assertEqual(result, 3)

    def test_number_of_anomalies_smaller(self):
        """
        Test that it works with selecting under threshold
        """
        data = [1, 1, 1, -1]
        result = simple_threshold(data, number=1, smaller=True)[0]
        self.assertEqual(result, 3)


if __name__ == '__main__':
    unittest.main()
