"""
Code by: Michael Katsoulis

This code generates points withing a 2D/3D enviroment. 
It takes in the size of the mesh and how many points need to be placed
and then randomly places the points.

Something like:
+--------------------------------------------------+
|          xx        x                             |
|    x        x               x   x                |
|      x  x               x                        |
|       x    x x     x xx                x         |
|         x                  x                x    |
+--------------------------------------------------+

"""
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import GenerateMesh






if __name__ == "__main__":
    np.random.seed(seed=0)
    num_pts = 50
    xlim, ylim = 500, 500
    zlim = 1
    z_axis_scale_factor = 3

    # pts = GenerateMesh.generatePoints(num_pts, xlim, ylim)
    pts = GenerateMesh.generatePoints(num_pts, xlim, ylim, zlim)


    # print(pts)
    fig0,ax0 = plt.subplots(1,1, figsize=(10,10))
    ax0.plot(pts[:,0],pts[:,1], 'o')
    fig0.show()
    

    # tri = Delaunay(pts)
    # fig,ax2d = plt.subplots(1,1, figsize=(10,10))
    # ax2d.triplot(pts[:,0], pts[:,1], tri.simplices.copy())
    # ax2d.plot(pts[:,0],pts[:,1], 'o')
    # fig.show()

    fig3d = plt.figure(figsize=(10,10))
    ax3d = plt.axes(projection='3d')
    # pts3d = GenerateMesh.generatePoints(num_pts, xlim, ylim, zlim)
    tri = Delaunay(pts[:,:2])
    ax3d.plot_trisurf(pts[:,0], pts[:,1], pts[:,2], triangles=tri.simplices.copy(),  cmap=plt.cm.Spectral)
    ax3d.plot(pts[:,0],pts[:,1],pts[:,2], 'o')
    ax3d.set_zlim([-z_axis_scale_factor*zlim-0.5*zlim, z_axis_scale_factor*zlim+0.5*zlim])
    fig3d.show()

    



