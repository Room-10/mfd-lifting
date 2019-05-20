
import numpy as np

from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_curves(curves, mfd, subgrid=None):
    if mfd.nembdim == 3 or hasattr(mfd, "embed"):
        plot_surface_curves(curves, mfd, subgrid=subgrid)
    else:
        plot_curves_2d(curves, mfd, subgrid=subgrid)

def plot_surface_curves(curves, mfd, subgrid=None):
    """ Plot curves on a triangulated surface embedded in R^3 """
    from mayavi import mlab
    verts = mfd.verts
    if hasattr(mfd, "embed"):
        verts, subgrid, *curves = map(mfd.embed, [verts,subgrid] + curves)

    x,y,z = np.hsplit(verts, 3)
    mlab.triangular_mesh(x, y, z, mfd.simplices)
    mlab.triangular_mesh(x, y, z, mfd.simplices,
        representation='wireframe', color=(0,0,0))


    if subgrid is not None:
        mlab.points3d(*np.hsplit(subgrid,3), scale_factor=.02)

    for i, crv in enumerate(curves):
        rgb = (1,0,0) if i%2 == 0 else (0,0,1)
        mlab.plot3d(*np.hsplit(crv,3), color=rgb, tube_radius=.01)

    for crv in np.stack(curves, axis=1):
        mlab.plot3d(*np.hsplit(crv,3), color=(0.5,0.5,0.5), tube_radius=.005)

    mlab.show()

def plot_curves_2d(curves, tri, subgrid=None):
    """ Plot curves on a triangulated area in the plane """
    plot_polys(tri.verts, tri.simplices)

    if subgrid is not None:
        plt.scatter(*np.hsplit(subgrid,2), c='#808080', s=10.0, marker='x')

    for i,crv in enumerate(curves):
        plt.plot(*np.hsplit(crv,2), c="r" if i%2 == 0 else "b")

    for crv in np.stack(curves, axis=1):
        plt.plot(*np.hsplit(crv,2), c='#A0A0A0', linestyle="--")

    plt.axis('equal')
    plt.show()

def plot_polys(verts, polys, facecolors=(1,1,1)):
    collection = PolyCollection([verts[p,::-1] for p in polys])
    collection.set_facecolor(facecolors)
    collection.set_edgecolor((0.5,0.5,0.5))
    ax = plt.gca()
    ax.add_collection(collection)
    xmin, xmax = np.amin(verts[:,1]), np.amax(verts[:,1])
    ymin, ymax = np.amin(verts[:,0]), np.amax(verts[:,0])
    ax.set_xlim([xmin-1,xmax+1])
    ax.set_ylim([ymin-1,ymax+1])

def plot_trifuns(subpoints, grid):
    M_tris, m_sublabels, _ = subpoints.shape
    nrows, ncols = len(grid), len(grid[0])
    fig = plt.figure(figsize=(13,8), dpi=100)
    c = [colors.rgb2hex(np.random.rand(3)) for t in range(M_tris)]
    offset = 100*nrows + 10*ncols + 1
    for i,row in enumerate(grid):
        for j,(fun,funmask) in enumerate(row):
            ax = fig.add_subplot(offset + (i*ncols + j), projection='3d')
            for k,vk in enumerate(subpoints):
                mask = funmask[k]
                plot_trifun(ax, vk[mask], fun[k][mask], c=c[k])
            set_plot_lims(ax, subpoints.reshape(-1,2), fun.ravel())
    fig.show()

def plot_trifun(ax, verts, vals, c=None):
    graph = np.vstack((verts[:,::-1].T, vals)).T
    for simplex in Delaunay(verts).simplices:
        polygon = Poly3DCollection([graph[simplex]], alpha=1.0)
        if c is not None: polygon.set_color(c)
        ax.add_collection3d(polygon)

def set_plot_lims(ax, verts, vals):
    xmin, xmax = np.amin(verts[:,1]), np.amax(verts[:,1])
    ymin, ymax = np.amin(verts[:,0]), np.amax(verts[:,0])
    zmin, zmax = np.amin(vals), np.amax(vals)
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.set_zlim([zmin,zmax])
