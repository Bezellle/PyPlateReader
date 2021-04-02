import unittest
import numpy as np
from source.tracker import Tracker


class TestTracker(unittest.TestCase):
    def setUp(self) -> None:
        self.m_tracker = Tracker()

    def tearDown(self) -> None:
        pass

    def test_add(self):
        point = np.array([1, 2])
        col1 = 0
        self.m_tracker.add(point, col1)
        self.assertEqual(self.m_tracker.Objects[0][0][0], point[0])
        self.assertEqual(self.m_tracker.Objects[0][0][1], point[1])
        self.assertEqual(len(self.m_tracker.Objects), 1)
        self.assertEqual(self.m_tracker.Objects[0][1], col1)

        point = np.array([4, 5])
        col1 = 2
        self.m_tracker.add(point, col1)
        self.assertEqual(self.m_tracker.Objects[1][0][0], point[0])
        self.assertEqual(self.m_tracker.Objects[1][0][1], point[1])
        self.assertEqual(len(self.m_tracker.Objects), 2)
        self.assertEqual(self.m_tracker.Objects[1][1], col1)

    def test_delOutOfScope(self):
        self.m_tracker.add(np.array([4, 5]), 1)
        self.m_tracker.add(np.array([57, 99]), 1)

        self.m_tracker.delOutOfScope(0)
        self.assertEqual(len(self.m_tracker.Objects), 1)

        self.m_tracker.delOutOfScope(1)
        self.assertEqual(len(self.m_tracker.Objects), 0)

    def test_update(self):
        # [id, (centerX, centerY), col]
        boxes1 = [(20, 20, 40, 30), (50, 80, 90, 120)]
        ret, track_id = self.m_tracker.update(boxes1)
        if ret:
            self.assertEqual(track_id[0][1], 0)
            self.assertEqual(track_id[1][1], 1)
        else:
            self.assertTrue(ret)

        # boxes moved by 10px
        boxes2 = [(30, 30, 50, 40), (60, 90, 100, 130)]
        ret, track_id = self.m_tracker.update(boxes2)
        if ret:
            self.assertEqual(track_id[0][1], 0)
            self.assertEqual(track_id[1][1], 1)
        else:
            self.assertTrue(ret)

        # Box 1 missing
        boxes3 = [(60, 90, 100, 130)]
        ret, track_id = self.m_tracker.update(boxes3)
        if ret:
            self.assertEqual(track_id[0][1], -1)
            self.assertEqual(track_id[1][1], 0)
        else:
            self.assertTrue(ret)

        # Move 1st box's x and y by 20px and add new box
        boxes4 = [(50, 50, 70, 60), (60, 90, 100, 130), (300, 500, 400, 300)]
        ret, track_id = self.m_tracker.update(boxes4)
        if ret:
            self.assertEqual(track_id[0][1], 0)
            self.assertEqual(track_id[1][1], 1)
            self.assertEqual(track_id[2][1], 2)
        else:
            self.assertTrue(ret)

        # change point order
        boxes5 = [(300, 500, 400, 300), (50, 50, 70, 60), (60, 90, 100, 130)]
        ret, track_id = self.m_tracker.update(boxes5)
        if ret:
            self.assertEqual(track_id[0][1], 1)
            self.assertEqual(track_id[1][1], 2)
            self.assertEqual(track_id[2][1], 0)
        else:
            self.assertTrue(ret)



if __name__ == '__main__':
    unittest.main()
