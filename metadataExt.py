import subprocess
import struct


def pad(n, base=4):
    i = n

    while i % base != 0:
        i += 1
    return i


class MataData:
    def __init__(self):
        self.BinaryData = bytearray()
        self.MapType = {'c': ['c', 1],
                        'L': ['L', 4],
                        's': ['h', 2],
                        'S': ['H', 2],
                        'f': ['f', 4],
                        'U': ['c', 1],
                        'l': ['l', 4],
                        'B': ['B', 1],
                        'f': ['f', 4],
                        'J': ['Q', 8]}

    def map_type(self, datatype):
        ctype = chr(datatype)
        if ctype in self.MapType.keys():
            return self.MapType[ctype]
        return ctype

    def read_value(self, datatype, size, count, offset):
        if datatype == 0:   # beginning of new stream has no type
            return [[0]]

        ctype, type_size = self.map_type(datatype)
        n_samples = int(size/type_size)
        fmt = '>' + ctype * n_samples
        st = struct.Struct(fmt)
        values = []

        for i in range(count):
            value = st.unpack_from(self.BinaryData, offset=offset+i*size+8)
            values.append(value)
        return values

    def load_bin(self):
        i = 5 #TODO make loading data from binary file

    def make_bin(self, file_name):
        p1 = subprocess.run('dir', shell=True, stdout=subprocess.PIPE, text=True)
        dst = file_name[:-3] + 'bin' #TODO to confirm
        result = subprocess.run(['ffmeg', '-i', file_name, dst])
        #TODO finish creating binary file
