
import numpy as np

from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_polys(verts, polys, facecolors=(1,1,1)):
    collection = PolyCollection([verts[p,::-1] for p in polys])
    collection.set_facecolor(facecolors)
    collection.set_edgecolor((0,0,0))
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
