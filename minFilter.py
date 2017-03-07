import numpy as np
from fileHelper import *


N = 100.0


def findRectangle(lines):
    """
    find the rectangle edge
    """
    X, Y, Z, I = np.array(lines).T

    return min(X), max(X), min(Y), max(Y)


def findMinAndStride(lines):
    """
    find the rectangle and stride
    """
    xMin, xMax, yMin, yMax = findRectangle(lines)
    xStride, yStride = (xMax - xMin) / N, (yMax - yMin) / N
    NY = int((yMax - yMin) / xStride)
    return xStride, xMin, yMin, NY


def splitToMatrix(lines):
    """
    map points to matrix, use grid in matrix to re-groupe the data
    """

    xStride, xMin, yMin, NY = findMinAndStride(lines)

    matrix = [[[] for _ in xrange(int(NY))] for _ in xrange(int(N))]

    for x, y, z, i in lines:
        xi = int((x - xMin) / xStride)
        yi = int((y - yMin) / xStride)
        # print xStride, yStride, xMin, xMax, yMin, yMax, xi, yi, x - xMin, y - yMin

        matrix[xi-1][yi-1].append((x, y, z, i))

    return matrix


def saveMatrix(matrix, fileName):
    """
    save the Matrix, only keep the min value in each group
    """
    with open(fileName, 'wb') as f:
        for i in xrange(len(matrix)):
            for j in xrange(len(matrix[0])):
                if matrix[i][j]:
                    matrix[i][j].sort(key=lambda n: n[2])
                    f.write('%s\n' % ' '.join(map(str, matrix[i][j][0])))


def minFilter():
    # lines = read('final_project_data/final_project_point_cloud.fuse')
    lines = read('xyzi.dat')
    matrix = splitToMatrix(lines)

    saveMatrix(matrix, 'minFilter.dat')


if __name__ == '__main__':
    minFilter()
