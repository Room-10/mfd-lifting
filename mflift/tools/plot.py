
import itertools
import numpy as np

from scipy.io import loadmat
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import colors, cm, rc
from matplotlib.collections import PolyCollection, EllipseCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from mflift.tools.linalg import quaternion_so3

def plot_so3(I, mask=None, shape="tric", filename=None):
    """ Plot (masked) image of SO(3) rotation matrices """
    import vtk

    imagedims = I.shape[:-1]
    if mask is None:
        mask = np.zeros(imagedims, dtype=bool)
    M = quaternion_so3(I)

    if shape == "piano":
        # https://www.thingiverse.com/thing:148696
        # CC-BY-NC-SA kazysd
        stl_file = "data/piano_assembly.stl"
        steps = 200
        pos = (150,-20,0)
    else:
        # https://www.thingiverse.com/thing:3313805
        # CC-BY-NC-SA BillyOceansBlues
        stl_file = "data/Triceratops_plane_cut.stl"
        steps = 100
        pos = (0,0,0)

    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    camzoom = 100 # simulates parallel projection
    camdist = camzoom*2*steps*max(*imagedims)
    campos = (0.5*steps*(imagedims[1]-1),-camdist,-0.5*steps*(imagedims[0]-1))
    camera = vtk.vtkCamera()
    camera.SetPosition(*campos)
    camera.SetFocalPoint(campos[0]+1,0,campos[2]-1) # bug in VTK (polar coords?)
    camera.Roll(45)
    camera.Zoom(camzoom)
    camera.SetClippingRange(0.5*camdist, 2*camdist)
    ren = vtk.vtkRenderer()
    ren.SetActiveCamera(camera)
    ren.SetBackground(vtk.vtkNamedColors().GetColor3d("White"))
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    matrix = vtk.vtkMatrix4x4()

    for i in range(imagedims[0]):
        for j in range(imagedims[1]):
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            col = (1.0,0.4,0.5) if mask[i,j] else (0.7,0.7,0.7)
            actor.GetProperty().SetColor(*col)
            actor.SetPosition(*pos)

            T = np.eye(4)
            T[:3,3] = np.array([steps*j,0,-steps*i])
            T[:3,:3] = M[i,j]
            [matrix.SetElement(k,l,T[k,l]) for k in range(4) for l in range(4)]
            T = vtk.vtkTransform()
            T.SetMatrix(matrix)
            actor.SetUserTransform(T)

            ren.AddActor(actor)

    if filename is None:
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.Initialize()
        renWin.Render()
        iren.Start()
    else:
        filename = "%s.png" % filename[:-4]
        renWin.SetOffScreenRendering(True)
        renWin.Render()
        win2img = vtk.vtkWindowToImageFilter()
        win2img.SetInput(renWin)
        win2img.SetScale(10)
        win2img.SetInputBufferTypeToRGBA()
        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(win2img.GetOutputPort())
        writer.SetFileName(filename)
        writer.Write()

def plot_spd2(Is, filename=None):
    rc('axes', linewidth=0.5)
    rc('font', size=7, family='serif')
    rc('xtick', top=True, direction='in')
    rc('xtick.major', size=2.5, width=0.5)
    rc('ytick', right=True, direction='in')
    rc('ytick.major', size=2.5, width=0.5)
    fig = plt.figure(figsize=(len(Is)*5, 5), dpi=100)

    for k,I in enumerate(Is):
        ax = fig.add_subplot(100 + len(Is)*10 + (k + 1))

        imagedims = I.shape[:2]
        vals, vecs = np.linalg.eig(I)

        FA = np.sqrt(0.5*(vals[...,0]**2 + vals[...,1]**2) - vals[...,0]*vals[...,1])
        FA /= np.linalg.norm(Is[0], axis=(-2,-1))

        lvals = np.log(vals)
        GA = np.sqrt(0.5*(lvals[...,0]**2 + lvals[...,1]**2) - lvals[...,0]*lvals[...,1])
        GA /= 1 + GA

        angles = 180 + 180*np.arctan2(vecs[...,0,0], vecs[...,1,0])/np.pi
        vals /= 0.5*vals.max()

        X, Y = np.meshgrid(np.arange(imagedims[0]), np.arange(imagedims[1]))
        XY = np.vstack((X.ravel(), Y.ravel())).T
        ec = EllipseCollection(vals[...,0], vals[...,1], angles, units='x',
                               offsets=XY, transOffset=ax.transData,
                               edgecolors=0.8*cm.hsv(GA.ravel())[:,:-1],
                               facecolors=1.0*cm.hsv(GA.ravel())[:,:-1],
                               linewidths=0.5)
        ax.add_collection(ec)
        ax.autoscale_view()
        ax.invert_yaxis()
        ax.axis("equal")

    if filename is None:
        plt.show()
    else:
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(filename)
        plt.close(fig)

def plot_elevation(elev, insar, filename=None):
    rc('grid', linestyle=':')
    rc('axes', linewidth=0.5)
    rc('font', size=7, family='serif')
    rc('xtick', top=True, direction='in')
    rc('xtick.major', size=2.5, width=0.5)
    rc('ytick', right=True, direction='in')
    rc('ytick.major', size=2.5, width=0.5)
    fig = plt.figure(figsize=(10, 5), dpi=100)

    X, Y = [np.arange(s) for s in elev.shape]
    X, Y = np.meshgrid(X, Y, indexing='ij')

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, elev, cmap='gray', shade=False)
    ax.view_init(elev=75., azim=105)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlim((0,elev.shape[0]))
    ax.set_ylim((0,elev.shape[1]))

    if filename is None:
        plt.show()
    else:
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(filename)
        plt.close(fig)

def plot_rcom(result, data, filename=None):
    rc('grid', linestyle=':')
    rc('font', size=19)
    rc('font', family='serif')
    rc('text', usetex=True)
    fig = plt.figure(figsize=(10,3.7), dpi=200)

    vlocal = 3.4834278686879987
    vglobal = result[0,0]

    points = data.points
    points2d = np.vstack((np.cos(points[:,0]), np.sin(points[:,0]))).T
    weights = data.weights
    X = np.linspace(-0.1, 2*np.pi + 0.1, 300)
    X2d = np.vstack((np.cos(X), np.sin(X))).T
    distX = data.rho(X[:,None])[0]
    R = data.R.reshape(data.M_tris, data.N_image, -1)[:,0].ravel()
    Rbase = data.Rbase.reshape(data.M_tris, data.N_image, -1)[:,0].ravel()
    S, R = data.S.ravel()[Rbase], R[Rbase]
    S[np.argmin(S)] = 2*np.pi
    ordered = S.argsort()
    S, R = S[ordered], R[ordered]
    T = np.concatenate((data.T,data.T + 2*np.pi,data.T - 2*np.pi), axis=0)[:,0]
    T = T[(T < 2*np.pi+0.1) & (T > -0.1)]

    ax = fig.add_subplot(121)
    ax.plot(X, distX)
    ax.axvline(vglobal, color='#AA0000', linewidth=2)
    ax.axvline(vlocal, color='#00AA00')
    [ax.axvline(t, color='#EE9900', linewidth=1) for t in T]
    ax.plot(S, R, color='#EE9900')
    ax.set_ylim((0.85*distX.min(),1.05*distX.max()))
    ax.set_xlim((-0.1,2*np.pi+0.1))
    ax.grid(True)
    ax.set_aspect(2.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax.xaxis.set_tick_params(which='major', pad=10)
    ax.yaxis.set_tick_params(which='major', pad=10)

    ax = fig.add_subplot(122)
    ax.plot(X2d[:,0], X2d[:,1], c='k')
    for pt,w in zip(points2d, weights):
        ln = np.vstack((1.05*pt,(1.1+5*w)*pt))
        ax.plot(ln[:,0],ln[:,1], linewidth=7, c='#1f77b4')
    w = 0.2
    pt = np.array([np.cos(vlocal), np.sin(vlocal)])
    ln = np.vstack((1.05*pt,(1.1+5*w)*pt))
    ax.plot(ln[:,0],ln[:,1], linewidth=5, c='#00AA00')
    pt = np.array([np.cos(vglobal), np.sin(vglobal)])
    ln = np.vstack((1.05*pt,(1.1+5*w)*pt))
    ax.plot(ln[:,0],ln[:,1], linewidth=5, c='#AA0000')
    ax.set_aspect(1.0)
    ax.axis('off')

    fig.subplots_adjust(left=0.06, bottom=0, right=1, top=1.1, wspace=0, hspace=0)
    if filename is None:
        plt.show()
    else:
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(filename)
        plt.close(fig)

def plot_hue_images(Is, filename=None):
    rc('font', size=12, family='serif')
    rc('xtick.major', size=2.5, width=0.5)
    rc('ytick.major', size=2.5, width=0.5)
    fig = plt.figure(figsize=(10.7,5), dpi=100)

    for i,I in enumerate(Is):
        ax = fig.add_subplot(100 + 10*len(Is) + (i+1))
        ax.imshow(I, vmin=0, vmax=2*np.pi, cmap='hsv')

    fig.subplots_adjust(left=0.045, bottom=0.05, right=0.99, top=0.98, wspace=0.085, hspace=0)
    if filename is None:
        plt.show()
    else:
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(filename)
        plt.close(fig)

def plot_terrain_maps(Is, dt, filename=None):
    rc('grid', linestyle=':')
    rc('axes', linewidth=0.5)
    rc('font', size=11, family='serif')
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
        ztickstride = int(np.ceil((ztickmax - ztickmin)/2))
        ax.set_zticks(range(ztickmin, ztickmax+ztickstride, ztickstride))

    fig.subplots_adjust(left=0.02, bottom=0.12, right=1, top=0.9, wspace=0.02, hspace=0)
    if filename is None:
        plt.show()
    else:
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(filename)
        plt.close(fig)

def plot_curves(curves, mfd, subgrid=None, filename=None):
    if mfd.ndim == 3:
        if type(mfd).__name__ == "SO3":
            plot_so3(np.stack(curves, axis=0), filename=filename)
        else:
            plot_curves_3d(curves, mfd, subgrid=subgrid, filename=filename)
    elif mfd.nembdim == 3:
        plot_surface_curves(curves, mfd, subgrid=subgrid, filename=filename)
    else:
        plot_curves_2d(curves, mfd, subgrid=subgrid, filename=filename)

def plot_surface_curves(curves, mfd, subgrid=None, filename=None):
    """ Plot curves on a triangulated surface embedded in R^3 """
    assert len(curves) == 2
    from mayavi import mlab
    if filename is not None:
        mlab.options.offscreen = True
        # don't do anything, because we can't avoid an annoying dialog
        return
    mfig = mlab.figure(size=(1024, 1024), bgcolor=(1,1,1))

    verts, simplices = mfd.mesh(0.2)
    verts = mfd.embed(verts)
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

    orthcurves = []
    for crv in np.stack(curves, axis=1):
        orthcurves.append(mfd.geodesic(crv[0], crv[1], 10))
    pointcurves = curves[:-1]
    curves = curves[-1:]

    for k in range(len(curves)):
        crv = []
        for i in range(curves[k].shape[0] - 1):
            crv.append(mfd.geodesic(curves[k][i], curves[k][i+1], 10))
        curves[k] = np.concatenate(crv, axis=0)

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
        mlab.savefig(filename, figure=mfig, magnification=2)
        pass

def plot_curves_2d(curves, tri, subgrid=None, filename=None):
    """ Plot curves on a triangulated area in the plane """
    fig = plt.figure(figsize=(12, 10), dpi=100)
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
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(filename)
        plt.close(fig)

def plot_curves_3d(curves, tri, subgrid=None, filename=None):
    """ Plot curves in a triangulated volume in R^3 """
    rc('grid', linestyle=':')
    rc('axes', linewidth=0.5)
    rc('font', size=7, family='serif')

    fig = plt.figure(figsize=(12, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plot_polys_3d(tri.verts, tri.simplices)

    if subgrid is not None:
        ax.scatter(subgrid[:,0], subgrid[:,1], subgrid[:,2],
                    c='#808080', s=10.0, marker='x')

    for i,crv in enumerate(curves):
        col = "r" if i%2 == 0 else "b"
        ax.plot(crv[:,0], crv[:,1], crv[:,2], c=col, linewidth=2)

    for crv in np.stack(curves, axis=1):
        ax.plot(crv[:,0], crv[:,1], crv[:,2],
                c='#A0A0A0', linestyle="--", linewidth=0.5)

    if filename is None:
        plt.show()
    else:
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(filename)
        plt.close(fig)

def plot_polys_3d(verts, tetrahedra):
    lines = []
    has_edge = np.zeros((verts.shape[0], verts.shape[0]), dtype=bool)
    for t in tetrahedra:
        for e in itertools.combinations(t, 2):
            e = sorted(e)
            if not has_edge[e[0],e[1]]:
                has_edge[e[0],e[1]] = True
                lines.append(verts[[e[0],e[1]]])
    collection = Line3DCollection(lines)
    collection.set_edgecolor((0.8,0.8,0.8))
    collection.set_linewidth(0.15)
    ax = plt.gca()
    ax.add_collection(collection)
    xmin, xmax = np.amin(verts[:,0]), np.amax(verts[:,0])
    ymin, ymax = np.amin(verts[:,1]), np.amax(verts[:,1])
    zmin, zmax = np.amin(verts[:,2]), np.amax(verts[:,2])
    ax.set_xlim([xmin-0.1,xmax+0.1])
    ax.set_ylim([ymin-0.1,ymax+0.1])
    ax.set_zlim([zmin-0.1,zmax+0.1])

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

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter
