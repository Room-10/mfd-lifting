
import itertools
import numpy as np

from scipy.io import loadmat
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import colors, cm, rc
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_hue_images(Is, filename=None):
    rc('font', size=7, family='serif')
    rc('xtick.major', size=2.5, width=0.5)
    rc('ytick.major', size=2.5, width=0.5)
    fig = plt.figure(figsize=(10.7,5), dpi=100)

    for i,I in enumerate(Is):
        ax = fig.add_subplot(100 + 10*len(Is) + (i+1))
        ax.imshow(I, vmin=0, vmax=2*np.pi, cmap='hsv')

    fig.subplots_adjust(left=0.04, bottom=0.05, right=0.99, top=0.99, wspace=0.07, hspace=0)
    if filename is None:
        plt.show()
    else:
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(filename)
        plt.close(fig)

def plot_terrain_maps(Is, dt, filename=None):
    rc('grid', linestyle=':')
    rc('axes', linewidth=0.5)
    rc('font', size=7, family='serif')
    rc('xtick', top=True, direction='in')
    rc('xtick.major', size=2.5, width=0.5)
    rc('ytick', right=True, direction='in')
    rc('ytick.major', size=2.5, width=0.5)
    fig = plt.figure(figsize=(17, 4), dpi=100)

    imagedims = Is[0].shape[:-1]
    ccm = loadmat("data/cm_mountain.mat")['ccm']
    X, Y = np.split(dt['normalscoords'], 2, axis=0)
    X = X.reshape(imagedims, order='F')
    Y = Y.reshape(imagedims, order='F')
    Xuu, Yuu = [np.arange(s) for s in dt['uu'].shape]
    Xuu, Yuu = np.meshgrid(Xuu, Yuu)
    uuh = dt['uuh']
    uuh_col = ccm[np.int64(255*(uuh - uuh.min())/(uuh.max() - uuh.min()))]

    lightnormal = np.array([1,-1,-1.5], dtype=np.float64)
    lightnormal /= np.linalg.norm(lightnormal)
    for i,I in enumerate(Is):
        Iabs = np.fmax(0, np.einsum('k,ijk->ij', -lightnormal, I))
        col = uuh_col*Iabs[:,:,None]

        ax = fig.add_subplot(100 + 20*len(Is) + (0*len(Is)+i+1))
        xstride = np.int(np.ceil(np.array(imagedims[0])/40))
        ystride = np.int(np.ceil(np.array(imagedims[1])/40))
        ax.quiver(X[::ystride,::xstride], Y[::ystride,::xstride],
                  -I[::ystride,::xstride,1], -I[::ystride,::xstride,0],
                  color='#1f77b4', # constant color
                  # optionally: colors by third component
                  #I[::ystride,::xstride,2], cmap='autumn',
                  headwidth=2.5, headlength=4.2, headaxislength=4.2,
                  width=0.0032, scale=27, minshaft=2)
        ax.set_xlim((0.7,imagedims[1]+0.1))
        ax.set_ylim((0.7,imagedims[0]+0.4))
        ax.set_aspect(1.0)
        ax.invert_xaxis()
        ax.invert_yaxis()

        ax = fig.add_subplot(100 + 20*len(Is) + (1*len(Is)+i+1), projection='3d')
        ax.plot_surface(Xuu, Yuu, dt['uu'], facecolors=col, shade=False)
        ax.view_init(elev=75., azim=105)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_xlim((0,dt['uu'].shape[0]))
        ax.set_ylim((0,dt['uu'].shape[1]))
        ax.set_zlim((dt['uu'].min()-0.5, dt['uu'].max()-1))
        ztickmin = int(10*np.ceil(dt['uu'].min()/10))
        ztickmax = int(10*np.floor(dt['uu'].max()/10))
        ztickstride = int(np.ceil((ztickmax - ztickmin)/3))
        ax.set_zticks(range(ztickmin, ztickmax+ztickstride, ztickstride))

    fig.subplots_adjust(left=0.02, bottom=0.1, right=1, top=0.9, wspace=0.05, hspace=0)
    if filename is None:
        plt.show()
    else:
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(filename)
        plt.close(fig)

def plot_curves(curves, mfd, subgrid=None, filename=None):
    if mfd.nembdim == 3 or hasattr(mfd, "embed"):
        plot_surface_curves(curves, mfd, subgrid=subgrid, filename=filename)
    else:
        plot_curves_2d(curves, mfd, subgrid=subgrid, filename=filename)

def plot_surface_curves(curves, mfd, subgrid=None, filename=None):
    """ Plot curves on a triangulated surface embedded in R^3 """
    from mayavi import mlab
    if filename is not None:
        mlab.options.offscreen = True
    mfig = mlab.figure(size=(1024, 1024), bgcolor=(1,1,1))

    verts, simplices = mfd.mesh(0.2)
    verts = mfd.embed(verts) if hasattr(mfd, "embed") else verts
    x,y,z = np.hsplit(verts, 3)
    mlab.triangular_mesh(x, y, z, simplices, color=(.9,.9,.9), opacity=0.8)

    verts, simplices = mfd.verts, mfd.simplices
    has_edge = np.zeros((verts.shape[0], verts.shape[0]), dtype=bool)
    for tri in simplices:
        for e in itertools.combinations(tri, 2):
            e = sorted(e)
            if not has_edge[e[0],e[1]]:
                has_edge[e[0],e[1]] = True
                crv = mfd.embed(mfd.geodesic(verts[e[0]],verts[e[1]], 20))
                mlab.plot3d(*np.hsplit(crv,3),
                    color=(.2,.2,.2), tube_radius=.015)

    pointcurves = []
    orthcurves = []
    if len(curves) > 2:
        pointcurves = curves[:-1]
        curves = curves[-1:]
    else:
        for crv in np.stack(curves, axis=1):
            orthcurves.append(mfd.geodesic(crv[0], crv[1], 10))

    for k in range(len(curves)):
        crv = []
        for i in range(curves[k].shape[0] - 1):
            crv.append(mfd.geodesic(curves[k][i], curves[k][i+1], 10))
        curves[k] = np.concatenate(crv, axis=0)

    if hasattr(mfd, "embed"):
        subgrid = None if subgrid is None else mfd.embed(subgrid)
        curves = [mfd.embed(c) for c in curves]
        orthcurves = [mfd.embed(c) for c in orthcurves]
        pointcurves = [mfd.embed(c) for c in pointcurves]

    if subgrid is not None:
        mlab.points3d(*np.hsplit(subgrid,3), scale_factor=.02)

    if len(curves) > 1:
        mlab.plot3d(*np.hsplit(curves[-2],3), color=(1,0,0), tube_radius=.03)
    for c in pointcurves:
        mlab.points3d(*np.hsplit(c,3), color=(1,0,0), scale_factor=.1)

    mlab.plot3d(*np.hsplit(curves[-1],3), color=(0,0,1), tube_radius=.03)

    for c in orthcurves:
        mlab.plot3d(*np.hsplit(c,3), color=(0.5,0.5,0.5), tube_radius=.005)

    if filename is None:
        mlab.show()
    else:
        #mlab.savefig(filename, figure=mfig, magnification=2)
        pass

def plot_curves_2d(curves, tri, subgrid=None, filename=None):
    """ Plot curves on a triangulated area in the plane """
    plt.figure(figsize=(12, 10), dpi=100)
    plot_polys(tri.verts, tri.simplices)

    if subgrid is not None:
        plt.scatter(*np.hsplit(subgrid,2), c='#808080', s=10.0, marker='x')

    for i,crv in enumerate(curves):
        col = "r" if i%2 == 0 else "b"
        plt.plot(*np.hsplit(crv,2), c=col, linewidth=2)

    for crv in np.stack(curves, axis=1):
        plt.plot(*np.hsplit(crv,2), c='#A0A0A0', linestyle="--", linewidth=0.5)

    plt.axis('equal')

    if filename is None:
        plt.show()
    else:
        fig = plt.gcf()
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(filename)
        plt.close(fig)

def plot_polys(verts, polys, facecolors=(1,1,1)):
    collection = PolyCollection([verts[p,::-1] for p in polys])
    collection.set_facecolor(facecolors)
    collection.set_edgecolor((0.8,0.8,0.8))
    collection.set_linewidth(0.15)
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
