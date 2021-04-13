import unittest
import source.objects
from pathlib import Path
import glob


class TestObjectsSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        #cls.file_path = Path('../video/GPFR3073.MP4')
        files_path = glob.glob('../video/*.MP4')
        cls.ObSet = source.objects.ObjectsSet(files_path)

    def test_init(self):

        self.assertEqual(744, len(self.ObSet.GPS))


if __name__ == '__main__':
    unittest.main()
