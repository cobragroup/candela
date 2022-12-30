import unittest
import numpy as np

from src.candela.plotting import __to_sequence as to_sequence
from src.candela.plotting import __edges as edges
from src.candela.plotting import __index_from_vec as index_from_vec
from src.candela.plotting import __vec_from_index as vec_from_index

class TestHelpers(unittest.TestCase):
    def test_to_sequence(self):
        res = to_sequence(None)
        self.assertEqual(res, [])
        res = to_sequence({})
        self.assertEqual(res, [])
        res = to_sequence("2")
        self.assertEqual(res[0], "2")
        res = to_sequence([2])
        self.assertEqual(res[0], [2])
        res = to_sequence(2)
        self.assertEqual(res, 2)

    def test_edges(self):
        data = np.array([1,1,1,0,0])
        res = edges(data)
        self.assertEqual(res[1],2)

    def test_index_from_vec(self):
        data = [0,0,1,1,0,0,1,0]
        res = index_from_vec(data)
        self.assertEqual(res[2],6)

    def test_vec_from_index(self):
        data = [2,3,6]
        res = vec_from_index(data, 8)
        self.assertEqual(res[2],1)
        self.assertEqual(len(res),8)
