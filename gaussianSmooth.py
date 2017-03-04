from minFilter import *
import cv2
import matplotlib.pyplot as plt


M = 40


def recSmooth(I, J, gMatrix, n):
    # print n
    count = 0
    hSum = 0
    hMin = float('inf')
    for i in xrange(max(I - n, 0), min(I + n, len(gMatrix))):
        for j in xrange(max(J - n, 0), min(J + n, len(gMatrix[0]))):
            if gMatrix[i][j] is not None:
                hMin = min(hMin, gMatrix[i][j])
                count += 1
                # hSum += gMatrix[i][j]

    if count:
        # gMatrix[i][j] = hSum / count
        return hMin
        # return hSum / count
    else:
        return None


def toGaussianMatrix(matrix):
    X, Y = len(matrix), int(len(matrix[0]) * (900 / 1741.0))

    gMatrix = [[min(matrix[i][j], key=lambda n: n[2])[2] if matrix[i][j] else None for j in xrange(Y)] for i in xrange(X)]

    while True:
        countNone = 0
        # print X, Y
        for i in xrange(X):
            print i
            for j in xrange(Y):
                if gMatrix[i][j] is None:
                    # recSmooth(i, j, gMatrix, X / M)
                    gMatrix[i][j] = recSmooth(i, j, gMatrix, X / M)
                    # print gMatrix[i][j]
                    if gMatrix[i][j] is None:
                        countNone += 1
        if countNone == 0: break
        print '-------', countNone

    return gMatrix
    # 458101
    # 458101


def gaussianSmooth():
    lines = read('xyzi.dat')
    matrix = splitToMatrix(lines)
    gMatrix = np.array(toGaussianMatrix(matrix))

    np.save('minSmoothed.npy', gMatrix)
    print gMatrix.dtype

    cv2.imwrite('color_img.jpg', gMatrix)

    blur = cv2.GaussianBlur(gMatrix, (49, 49), 0)

    plt.imshow(blur)
    plt.colorbar()
    plt.show()


def readAndGaussian():
    # cv2.imwrite('color_img.jpg', gMatrix)
    # image = cv2.imread('color_img.jpg', 0)
    image = np.load('minSmoothed.npy')
    # image = np.load('smmothed.npy')
    blur = cv2.GaussianBlur(image, (99, 99), 0)

    plt.imshow(blur)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    # gaussianSmooth()
    readAndGaussian()
