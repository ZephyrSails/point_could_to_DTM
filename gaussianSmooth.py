from minFilter import *
import cv2
import matplotlib.pyplot as plt


M = 40  # M will influence recSmooth, if we make M smaller
        # recSmooth will search through larger area for each missing grid.


def recSmooth(I, J, gMatrix, n):
    """
    use the nearby grid to predict the missing grid
    """
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
    """
    transform the point cloud to grid
    which can be used to deploy gaussian blur later
    """
    # (900 / 1741.0) is hardcoded to cut the image, remove the sparse area.
    X, Y = len(matrix), len(matrix[0])

    # init the gaussian matrix, currently just keeping the min value for each grid.
    gMatrix = [[min(matrix[i][j], key=lambda n: n[2])[2] if matrix[i][j] else None for j in xrange(Y)] for i in xrange(X)]
    # hardcoded to cut the image, remove the sparse area.
    gMatrix = np.array(gMatrix)[200:1000, 400:1200]

    while True: # Some of the grid in the matrix is missing, we need to put some value into them
        countNone = 0
        for i in xrange(len(gMatrix)):
            print i
            for j in xrange(len(gMatrix[0])):
                if gMatrix[i][j] is None:
                    # gMatrix[i][j] = 0
                    gMatrix[i][j] = recSmooth(i, j, gMatrix, X / M)
                    if gMatrix[i][j] is None:
                        countNone += 1
        if countNone == 0: break
        print '-------', countNone

    return gMatrix


def gaussianSmooth():
    """
    Main function, the output is saved in 'minSmoothed.npy', used later.
    """
    lines = read('xyzi.dat')
    matrix = splitToMatrix(lines)
    gMatrix = np.array(toGaussianMatrix(matrix))

    np.save('minSmoothed2.npy', gMatrix)
    print gMatrix.dtype
    gMatrix = map(lambda n: map(float, n), gMatrix)

    cv2.imwrite('color_img.jpg', gMatrix)

    blur = cv2.GaussianBlur(gMatrix, (49, 49), 0)

    plt.imshow(blur)
    plt.colorbar()
    plt.show()


def saveToPointCloud(image, fileName):
    """
    save the grid matrix as point cloud
    """
    lines = read('xyzi.dat')
    stride, xMin, yMin, _ = findMinAndStride(lines)
    xMin, yMin = xMin + 200 * stride, yMin + 400 * stride
    with open(fileName, 'wb') as f:
        for i in xrange(len(image)):
            for j in xrange(len(image[0])):
                f.write('%f %f %f\n' % ((xMin + i * stride), (yMin + j * stride), image[i][j]))


def readAndGaussian():
    """
    Read saved grid image, and deploy Gaussian.
    """
    # cv2.imwrite('color_img.jpg', gMatrix)
    # image = cv2.imread('color_img.jpg', 0)
    image = np.load('minSmoothed2.npy')
    image = np.array(map(lambda n: map(float, n), image))
    # image = np.load('smmothed.npy')
    image = cv2.GaussianBlur(image, (149, 149), 0)

    saveToPointCloud(image, 'GaussianPointCloud.dat')

    plt.imshow(image)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    # gaussianSmooth()
    readAndGaussian()
