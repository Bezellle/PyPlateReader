import unittest
from pathlib import Path
from source.metadataExt import MetaData
import subprocess


class TestMetaData(unittest.TestCase):
    def setUp(self) -> None:
        self.Metadata = MetaData()

    def tearDown(self) -> None:
        pass

    def test_makeBin(self):
        video_path = Path('video/GPFR3074.MP4')
        bin_path = Path('video/GPFR3074.bin')
        self.Metadata.makeBin(video_path)

        self.assertTrue(bin_path.exists())
        # Delete .bin file after test:
        subprocess.run(['rm', str(bin_path)])

if __name__ == '__main__':
    unittest.main()
