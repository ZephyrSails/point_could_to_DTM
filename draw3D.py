from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from latLonZ2XYZ import *
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def plotTrisurf(x, y, z):
    print np.shape(x), np.shape(y), np.shape(z)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    plt.show()


def plotSurface(x, y, z):
    print np.shape(x), np.shape(y), np.shape(z)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plotContour(x, y, z):
    print np.shape(x), np.shape(y), np.shape(z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cset = ax.contour(X, Y, Z, cmap=cm.coolwarm)
    ax.clabel(cset, fontsize=9, inline=1)

    plt.show()


def plotPointCloud(x, y, z, i):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=i)

    plt.show()


if __name__ in '__main__':
    X, Y, Z, I = np.array(read('final_project_data/final_project_point_cloud.fuse')).T
    # Z = np.zeros_like(X)
    # X, Y, Z = axes3d.get_test_data(0.05)

    plotPointCloud(X, Y, Z, I)
    # plotTrisurf(X, Y, Z)
    # plotSurface(X, Y, Z)
    # plotContour(X, Y, Z)
