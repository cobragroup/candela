import unittest
import numpy as np
import pandas as pd

from src.candela.functions import tukeys_method


class TestThreshold(unittest.TestCase):
    def test_array(self):
        """
        Test that it can use array input
        """
        data = np.array([3, 3, 4, 4, 5, 5, 6, 8])
        result = tukeys_method(data)[0]
        self.assertEqual(result, 7)

    def test_list(self):
        """
        Test that it can use list input
        """
        data = [3, 3, 4, 4, 5, 5, 6, 8]
        result = tukeys_method(data)[0]
        self.assertEqual(result, 7)

    def test_series(self):
        """
        Test that it can use series input
        """
        data = pd.Series([3, 3, 4, 4, 5, 5, 6, 8], index=[
                         101, 102, 103, 104, 105, 106, 107, 108], name="test")
        result = tukeys_method(data)[0]
        self.assertEqual(result, 7)

    def test_more_anomalies (self):
        """
        Test that it can detect more than one anomaly
        """
        data = [1, 3, 4, 4, 5, 5, 6, 8]
        result = len(tukeys_method(data))
        self.assertEqual(result, 2)


if __name__ == '__main__':
    unittest.main()
