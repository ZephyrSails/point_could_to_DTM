import csv
import math
from fileHelper import *


F   = 1.0 / 298.257224
RAD = 6378137.0
H   = 0.0


def latLonZ2XYZ(line):
    lat, lon, alt, intensity = map(float, line)

    cosLat = math.cos(lat * math.pi / 180.0)
    sinLat = math.sin(lat * math.pi / 180.0)
    cosLon = math.cos(lon * math.pi / 180.0)
    sinLon = math.sin(lon * math.pi / 180.0)

    C = 1.0 / math.sqrt(cosLat * cosLat + (1 - F) * (1 - F) * sinLat * sinLat)
    S = (1.0 - F) * (1.0 - F) * C

    x = (RAD * C + H) * cosLat * cosLon
    y = (RAD * C + H) * cosLat * sinLon
    z = alt

    return map(str, [x, y, z, intensity])


if __name__ == '__main__':
    # print read('final_project_data/final_project_point_cloud.fuse')[0:2]
    # convertedList = map(latLonZ2XYZ, read('final_project_data/final_project_point_cloud.fuse'))
    convertedList = map(latLonZ2XYZ, read('xyzi.dat'))
    writeToFile(convertedList, 'xyzi.dat')
