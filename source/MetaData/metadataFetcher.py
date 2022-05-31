import subprocess
import struct
import os
from math import sqrt
from pathlib import Path


def pad(n, base=4):
    i = n
    while i % base != 0:
        i += 1
    return i


class MetaDataFetcher:
    MapType = {'c': ['c', 1],
               'L': ['L', 4],
               's': ['h', 2],
               'S': ['H', 2],
               'f': ['f', 4],
               'U': ['c', 1],
               'l': ['l', 4],
               'B': ['B', 1],
               'f': ['f', 4],
               'J': ['Q', 8]}

    def __init__(self, file_path=None, total_frames=0):
        self.BinaryData = bytearray()
        self.GPS = []
        self.Path = Path()
        self.TotalFrames = total_frames

        if file_path is not None:
            if not os.path.exists(file_path):
                raise OSError('File not found. Metadata can not be extracted')
            else:
                self.loadBin(file_path)

    def __del__(self):
        cmd = 'rm {}'
        # subprocess.run(cmd.format(self.Path), shell=True)

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
        #bin_path = file_name[:-3] + 'bin'
        bin_path = file_name.with_suffix('.bin')

        if not os.path.exists(bin_path):
            if os.path.exists(file_name):
                self.makeBin(file_name)
            else:
                raise Exception('Could not make binary file with metadata')

        try:
            with open(bin_path, 'rb') as f:
                self.BinaryData = f.read()
                self.Path = bin_path
        except OSError:
            print("Could not load binary file with metadata")
            raise

    @staticmethod
    def makeBin(file_name):
        # CMD command for extracting meeta data with ffmpeg:
        # ffmpeg -y -i GPFR1846.MP4 -codec copy -map 0:3 -f rawvideo GPFR1846.bin

        #dst = file_name[:-3] + 'bin'
        dst = file_name.with_suffix('.bin')
        cmd = 'ffmpeg -y -i ./{} -codec copy -map 0:3 -f rawvideo ./{}'

        result = subprocess.run(cmd.format(str(file_name), str(dst)), shell=True, check=True)

    def loadGPSData(self, path):
        if len(self.BinaryData) == 0:
            self.loadBin(path)

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
            index = len(self.GPS) - 2
        sin_dir = 0
        #x_len = self.GPS[index+1][0] - self.GPS[index][0]
        #y_len = self.GPS[index+1][1] - self.GPS[index][1]

        #dist = sqrt(x_len ** 2 + y_len ** 2)
        #if dist == 0:
        #    sin_dir = 1
        #else:
        #   sin_dir = y_len/dist

        return self.GPS[index], sin_dir
