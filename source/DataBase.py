import sqlite3
from pathlib import Path


class DBApi:
    INSERT_COMMAND = "INSERT INTO {} VALUES (?, ?, ?, ?)"
    CREATE_TABLE_COM = """CREATE TABLE test (pl_number text, qty integer, location1, location2)"""
    DBName = Path('./PyPlateReaderDB.db')

    def __init__(self, test=False):
        # test param used only for unit testing. Do not use in different way!

        if test:
            self.Conn = sqlite3.connect(":memory:")
            self.Cursor = self.Conn.cursor()
            self.Cursor.execute(self.CREATE_TABLE_COM)
        else:
            self.Conn = sqlite3.connect(self.DBName)
            self.Cursor = self.Conn.cursor()

    def __del__(self):
        self.Conn.close()

    def addPlateRecord(self, plate_string, qty, location1=None, location2=None, table='test'):
        with self.Conn:
            self.Cursor.execute(self.INSERT_COMMAND.format(table), (plate_string, qty, location1, location2))

    def getPlate(self, plate_string):
        self.Cursor
