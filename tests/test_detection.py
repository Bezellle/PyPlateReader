import unittest
from pathlib import Path
from source.MetaData.metadataFetcher import MetaDataFetcher
from source.detection import Detection


class TestMetaData(unittest.TestCase):

    # TODO Add detecton test for refactor
    @classmethod
    def setUpClass(cls) -> None:
        cls.Detection = Detection()
        cls.IMG = Path('GF213033.JPG')


    @classmethod
    def tearDownClass(cls) -> None:
        del cls.Detection

    def test_makeBin(self):
        self.assertTrue()


if __name__ == '__main__':
    unittest.main()
