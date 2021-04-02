import unittest
from source.DataBase import DBApi
import subprocess


class TestDataBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dbc = DBApi(test=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.dbc.Conn.close()
        del cls.dbc
        # subprocess.run(['rm', 'PyPlateReaderDB.db'])

    def test_addValue(self):
        self.dbc.addValue('test1', 15)

        self.dbc.Cursor.execute("SELECT * FROM test WHERE pl_number='test1'")
        result = self.dbc.Cursor.fetchall()
        self.assertEqual(result[0][0], 'test1')
        self.assertEqual(result[0][1], 15)

        self.dbc.addValue('test2', 20)

        self.dbc.Cursor.execute("SELECT * FROM test WHERE pl_number='test2'")
        result = self.dbc.Cursor.fetchall()
        self.assertEqual(result[0][0], 'test2')
        self.assertEqual(result[0][1], 20)


    def test_getValue(self):
        pass


if __name__ == '__main__':
    unittest.main()
