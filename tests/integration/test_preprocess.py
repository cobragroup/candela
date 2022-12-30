import unittest
import pandas as pd

from src.candela.particles2d import preprocess


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv("tests/integration/fixtures/sample_traces.csv")

    def test_base_preprocess(self):
        data = self.df[self.df.track_id == 14303]
        prep = preprocess(data)
        self.assertEqual(prep.track_id.count(), 1693)
        self.assertEqual(prep.frame_id.min(), 4230)
        self.assertEqual(prep.frame_id.max(), 5922)

    def test_specialised_preprocess(self):
        SPATIAL_SCALE = 43.3/37
        FRAMES_FREQUENCY = 10
        START_TIME = 1568379600.113061
        KMH_FLAG = True
        data = self.df[self.df.track_id == 134984]
        prep = preprocess(data, fps=FRAMES_FREQUENCY, kmh=KMH_FLAG,
                          scale=SPATIAL_SCALE, startTime=START_TIME)
        self.assertEqual(prep.true_time.max(), pd.to_datetime(
            "2019-09-13 13:14:03.813061120"))
        self.assertAlmostEqual(prep.x.max(), 16.43366821)

    def test_wrong_input(self):
        self.assertRaises(AssertionError, preprocess, self.df)
        data = self.df[self.df.track_id == 14303].drop(columns="track_id")
        self.assertRaises(AssertionError, preprocess, data)
