import subprocess
import struct
import os


def pad(n, base=4):
    i = n

    while i % base != 0:
        i += 1
    return i


class MetaData:
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

        self.GPS = []

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

    def load_bin(self, file_name):
        bin_path = file_name[:-3] + 'bin'
        check = os.path.exists(bin_path)
        if not check:
            self.make_bin(file_name)

        try:
            with open(bin_path, 'rb') as f:
                self.BinaryData = f.read()
        except:
            print("Could not load binary file with metadata")

    @staticmethod
    def make_bin(file_name):
        #CMD command: ffmpeg -y -i GPFR1846.MP4 -codec copy -map 0:3 -f rawvideo GPFR1846.bin

        dst = file_name[:-3] + 'bin'
        result = subprocess.run(['ffmpeg', '-y', '-i', file_name, '-codec', 'copy', '-map', '0:3', '-f', 'rawvideo',
                                 dst], shell=True)

    def load_data(self):
        binary_format = '>4sBBH'
        s = struct.Struct(binary_format)
        offset = 0

        while offset < len(self.BinaryData) - 8:
            label, dataType, size, count = s.unpack_from(self.BinaryData, offset=offset)
            length = pad(size*count)

            if label == b'GPS5':
                gps5 = self.read_value(dataType, size, count, offset)

                for point in gps5:
                    self.GPS.append(point)

            offset += 8
            if dataType != 0:
                offset += length




