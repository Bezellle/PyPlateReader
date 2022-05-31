import unittest
from pathlib import Path
from source.MetaData.metadataFetcher import MetaDataFetcher


class TestMetaData(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.Metadata = MetaDataFetcher(total_frames=500)
        cls.VideoPath = Path('../video/GPFR3073.MP4')
        cls.BinPath = Path('../video/GPFR3073.bin')

    @classmethod
    def tearDownClass(cls) -> None:
        # Delete .bin file after test:
        #subprocess.run(['rm', str(cls.BinPath)])
        del cls.Metadata

    def test_makeBin(self):
        video_path = Path('../video/GPFR3073.MP4')
        bin_path = Path('../video/GPFR3073.bin')
        self.Metadata.makeBin(self.VideoPath)

        self.assertTrue(self.BinPath.exists())

    def test_loadGPSData(self):
        self.Metadata.loadGPSData(self.VideoPath)
        self.assertTrue(len(self.Metadata.GPS) > 0)

    def test_cameraLocation(self):
        self.Metadata.loadGPSData(self.VideoPath)
        location, ort = self.Metadata.getCameraLocation(69)
        print(location)
        self.assertEqual(location[0], 511431789)

if __name__ == '__main__':
    unittest.main()
