import unittest
import numpy as np
import pandas as pd

from src.candela.particles2d import preprocess
from src.candela.plotting import plot_features

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv("tests/integration/fixtures/sample_traces.csv")
        SPATIAL_SCALE = 43.3/37
        FRAMES_FREQUENCY = 10
        START_TIME = 1568379600.113061
        KMH_FLAG = True
        self.track = preprocess(self.df[self.df.track_id == 134984])

    def test_only_feature(self):
        fig, ax = plot_features("x","vtot", data=self.track)
        res = ax.get_ylabel()
        self.assertEqual(res, "vtot")
        ha = ax.get_legend()
        res = ha.get_texts()[0].get_text()
        self.assertEqual(res, "vtot")