import sqlite3


class DBApi:
    INSERT_COMMAND = "INSERT INTO {} VALUES (?, ?)"
    CREATE_TABLE_COM = """CREATE TABLE test (pl_number text, qty integer)"""

    def __init__(self, test=False):
        # test param used only for unit testing. Don't use in different way!
        self.DBName = './PyPlateReaderDB.db'
        if test:
            self.Conn = sqlite3.connect(":memory:")
            self.Cursor = self.Conn.cursor()
            self.Cursor.execute(self.CREATE_TABLE_COM)
        else:
            self.Conn = sqlite3.connect(self.DBName)
            self.Cursor = self.Conn.cursor()

    def __del__(self):
        self.Conn.close()

    def addValue(self, plate_string, qty, table='test'):
        with self.Conn:
            self.Cursor.execute(self.INSERT_COMMAND.format(table), (plate_string, qty))


    def getPlate(self, plate_string):
        self.Cursor
