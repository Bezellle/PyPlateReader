import subprocess
import struct
import os


def pad(n, base=4):
    i = n

    while i % base != 0:
        i += 1
    return i


class MetaData:
    def __init__(self, file_path='GPFR1846.MP4', total_frames=0, FPS=0.0):
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
        self.TotalFrames = total_frames
        if not os.path.exists(file_path):
            raise OSError('file not found. Metadata can not be extracted')
        else:
            self.Path = file_path

    def mapType(self, datatype):
        ctype = chr(datatype)
        if ctype in self.MapType.keys():
            return self.MapType[ctype]
        return ctype

    def readValue(self, datatype, size, count, offset):
        if datatype == 0:   # beginning of new stream has no type
            return [[0]]

        ctype, type_size = self.mapType(datatype)
        n_samples = int(size/type_size)
        fmt = '>' + ctype * n_samples
        st = struct.Struct(fmt)
        values = []

        for i in range(count):
            value = st.unpack_from(self.BinaryData, offset=offset+i*size+8)
            values.append(value)
        return values

    def loadBin(self, file_name):
        bin_path = file_name[:-3] + 'bin'

        if not os.path.exists(bin_path):
            if os.path.exists(file_name):
                self.makeBin(file_name)
            else:
                raise Exception('Could not make binary file with metadata')

        try:
            with open(bin_path, 'rb') as f:
                self.BinaryData = f.read()
        except OSError:
            print("Could not load binary file with metadata")
            raise

    @staticmethod
    def makeBin(file_name):
        #CMD command: ffmpeg -y -i GPFR1846.MP4 -codec copy -map 0:3 -f rawvideo GPFR1846.bin

        dst = file_name[:-3] + 'bin'
        result = subprocess.run(['ffmpeg', '-y', '-i', file_name, '-codec', 'copy', '-map', '0:3', '-f', 'rawvideo',
                                 dst], shell=True)

    def loadData(self):
        binary_format = '>4sBBH'
        s = struct.Struct(binary_format)
        offset = 0

        while offset < len(self.BinaryData) - 8:
            label, data_type, size, count = s.unpack_from(self.BinaryData, offset=offset)
            length = pad(size*count)

            if label == b'GPS5':
                gps5 = self.readValue(data_type, size, count, offset)

                for point in gps5:
                    self.GPS.append(point)

            offset += 8
            if data_type != 0:
                offset += length

    def getCameraLocation(self, frame_number):
        index = round((frame_number * len(self.GPS)) / self.TotalFrames)

        if index >= len(self.GPS):
            index = len(self.GPS) - 1

        return self.GPS[index]




