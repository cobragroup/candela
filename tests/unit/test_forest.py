import unittest
import numpy as np
import pandas as pd

from src.candela.functions import isolation_forest


class TestForest(unittest.TestCase):
    def test_array(self):
        """
        Test that it can use array input
        """
        data = np.array([1, 1, 1, 4])
        result = isolation_forest(data)[0]
        self.assertEqual(result, 3)

    def test_list(self):
        """
        Test that it can use list input
        """
        data = [1, 1, 1, 4]
        result = isolation_forest(data)[0]
        self.assertEqual(result, 3)

    def test_series(self):
        """
        Test that it can use series input
        """
        data = pd.Series([1, 1, 1, 4], index=[101, 102, 103, 104], name="test")
        result = isolation_forest(data)[0]
        self.assertEqual(result, 3)

    def test_2D_data(self):
        data = [[5,1],[1,1],[1,1],[1,1],[1,1],[1,5]]
        result = isolation_forest(data)
        self.assertTrue((result==np.array([0,5])).all())

if __name__ == '__main__':
    unittest.main()
