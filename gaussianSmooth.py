from minFilter import *
import cv2
import matplotlib.pyplot as plt
import copy


INPUT_POINT_CLOUD = 'xyzi.dat'
# INPUT_POINT_CLOUD = 'de_spike.txt'
GRID_REPRESENTATION = 'minSmoothed4.npy'
M = 10  # M will influence recSmooth, if we make M smaller
        # recSmooth will search through larger area for each missing grid.
G_BLUR1=17
G_BLUR2=7

M_KERNEL1=6
M_KERNEL2=10

HEIGHT_MIN=221
HEIGHT_MID=223
HEIGHT_MAX=224.5
HEIGHT_LIMIT=222.34

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
                hSum += gMatrix[i][j]

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
    gMatrix = np.array([[min(matrix[i][j], key=lambda n: n[2])[2] if matrix[i][j] else HEIGHT_LIMIT for j in xrange(Y)] for i in xrange(X)])
    # hardcoded to cut the image, remove the sparse area.
    # gMatrix = np.array(gMatrix)[200:1000, 400:1200]
    # print gMatrix[80][96], gMatrix[89][166]

    kernel = np.ones((M_KERNEL1, M_KERNEL1), np.uint8)
    mask = cv2.erode(gMatrix, kernel, iterations = 1)

    mask = np.array([[mask[i][j] if mask[i][j] != HEIGHT_MID else HEIGHT_MID for j in xrange(Y)] for i in xrange(X)])
    # kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 3)

    mask = cv2.erode(mask, kernel, iterations = 1)

    mask = np.array([[1 if mask[i][j] >= HEIGHT_MAX else 0 for j in xrange(Y)] for i in xrange(X)])

    kernel = np.ones((M_KERNEL2, M_KERNEL2), np.uint8)
    gMatrix = cv2.erode(gMatrix, kernel, iterations = 1)
    # while True: # Some of the grid in the matrix is missing, we need to put some value into them
    #     countNone = 0
    #     cpMatrix = copy.deepcopy(gMatrix)
    #     for i in xrange(len(gMatrix)):
    #         print i
    #         for j in xrange(len(gMatrix[0])):
    #             # if gMatrix[i][j] is None:
    #             if True:
    #                 # gMatrix[i][j] = 0
    #                 # gMatrix[i][j] = recSmooth(i, j, cpMatrix, X / M)
    #                 gMatrix[i][j] = recSmooth(i, j, cpMatrix, 20)
    #                 if gMatrix[i][j] is None:
    #                     countNone += 1
    #     if countNone == 0: break
    #     print '-------', countNone

    return gMatrix, mask


def gaussianSmooth():
    """
    Main function, the output is saved in 'minSmoothed.npy', used later.
    """
    lines = read(INPUT_POINT_CLOUD)
    matrix = splitToMatrix(lines)
    gMatrix, mask = toGaussianMatrix(matrix)
    # gMatrix = np.array()

    np.save(GRID_REPRESENTATION, gMatrix)
    print gMatrix.dtype
    gMatrix = np.array(map(lambda n: map(float, n), gMatrix))

    # cv2.imwrite('color_img.jpg', gMatrix)

    blur = cv2.GaussianBlur(gMatrix, (G_BLUR1, G_BLUR1), 0)

    mean = gMatrix[mask == 1].mean()

    image = np.array([[mean if mask[i][j] else blur[i][j] for j in xrange(len(gMatrix[0]))] for i in xrange(len(gMatrix))])

    image = cv2.GaussianBlur(image, (G_BLUR2, G_BLUR2), 0)

    # image[image == 223] = 0

    plt.imshow(image)
    plt.colorbar()
    plt.show()
    saveToPointCloud(image,'test.txt')


def saveToPointCloud(image, fileName):
    """
    save the grid matrix as point cloud
    """
    lines = read(INPUT_POINT_CLOUD)
    stride, xMin, yMin, _ = findMinAndStride(lines)
    # xMin, yMin = xMin + 200 * stride, yMin + 400 * stride
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
    image = np.load(GRID_REPRESENTATION)
    image = np.array(map(lambda n: map(float, n), image))
    # image = np.load('smmothed.npy')
    blur = cv2.GaussianBlur(image, (G_BLUR2, G_BLUR2), 0)

    blur = np.array(map(lambda n: map(int, n), blur))

    minBlur = np.array([[min(image[i][j], blur[i][j]) for j in xrange(len(image[0]))] for i in xrange(len(image))])

    # laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    # edges = cv2.Canny(blur, 100, 200)
    # minBlurBlur = cv2.GaussianBlur(image, (49, 49), 0)
    # minBlur

    saveToPointCloud(minBlur, 'GaussianMinPointCloud.dat')

    plt.imshow(image)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    gaussianSmooth()
    # readAndGaussian()
