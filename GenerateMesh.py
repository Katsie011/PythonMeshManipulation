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



def makeGrid(points, xdim, ydim, zdim =None, stepSize = 0.5):
    """
    Returns an array of sampled points all spaced out and structured

    """
    pass


def generatePoints(total_num, x_dim, y_dim, z_dim =np.nan, stepSize = 0.5):
    """
    Returns the locations of the random points

    
    """


    if np.isnan(z_dim):
        x_pts = np.random.randint(x_dim+1, size=total_num)
        y_pts = np.random.randint(y_dim+1, size=total_num)
        pts = np.vstack((x_pts, y_pts)).T

    else:
        x_pts = np.random.randint(x_dim+1, size=total_num)
        y_pts = np.random.randint(y_dim+1, size=total_num)
        z_pts = np.random.rand(total_num) * z_dim
        pts = np.vstack((x_pts, y_pts, z_pts)).T

    return pts




def viewPoints(points):
    """
    Turns the points into something nice to look at
    """
    pass





if __name__ == "__main__":
    print("Test generate points 2D")
    print("---------------------------------")
    print(generatePoints(20, 100, 100))
    print()

    print("Test generatePoints 3D")
    print("---------------------------------")
    print(generatePoints(20, 100, 100, 100))
    print()




