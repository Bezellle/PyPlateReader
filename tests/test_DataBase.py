import unittest
from source.DataBase import DBApi
import subprocess


class TestDataBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dbc = DBApi(test=True)

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.dbc
        # subprocess.run(['rm', 'PyPlateReaderDB.db'])

    def test_addValue(self):
        self.dbc.addPlateRecord('test1', 15, 57123, 14898)

        self.dbc.Cursor.execute("SELECT * FROM test WHERE pl_number='test1'")
        result = self.dbc.Cursor.fetchall()
        self.assertEqual(result[0], ('test1', 15, 57123, 14898))

        self.dbc.addPlateRecord('test2', 20, 55666, 19456)

        self.dbc.Cursor.execute("SELECT * FROM test WHERE pl_number='test2'")
        result = self.dbc.Cursor.fetchall()
        self.assertEqual(result[0], ('test2', 20, 55666, 19456))

        self.dbc.addPlateRecord('test3', 9)

        self.dbc.Cursor.execute("SELECT * FROM test WHERE pl_number='test3'")
        result = self.dbc.Cursor.fetchall()
        self.assertEqual(result[0], ('test3', 9, None, None))

    def test_getValue(self):
        pass


if __name__ == '__main__':
    unittest.main()
