
import numpy as np
from scipy.spatial import Delaunay

from mflift.manifolds import DiscretizedManifold

class FlatManifold(DiscretizedManifold):
    """ Flat convex n-D manifold embedded in n-D euclidean space

    >>> mfd = FlatManifold(verts)
    >>> # verts : ndarray of floats, shape (nverts,ndim)
    """
    def __init__(self, verts, simplices=None):
        self.verts = np.ascontiguousarray(verts)
        if self.verts.ndim < 2:
            self.verts = self.verts[:,None]
        self.nverts, self.ndim = self.verts.shape

        self.simplices = simplices
        if self.ndim > 1:
            self._delaunay = Delaunay(self.verts)
            if self.simplices is None:
                self.simplices = self._delaunay.simplices
        elif self.simplices is None:
            idx = np.argsort(self.verts[:,0])
            self.simplices = [idx[i:i+2] for i in range(self.nverts-1)]
        self.simplices = np.ascontiguousarray(self.simplices)

        DiscretizedManifold.__init__(self, 1)

    def mesh(self, h):
        """ dummy function for compatibility """
        return self.verts, self.simplices

    def _log(self, location, pfrom, out):
        out[:] = pfrom[:,None,:,:] - location[:,:,None,:]

    def _exp(self, location, vfrom, out):
        out[:] = location[:,:,None,:] + vfrom[:,None,:,:]

    def _mean(self, points, weights, out):
        np.einsum("ikm,ilmt->ilkt", weights, points, out=out)

    def _dist(self, x, y, out):
        # The assumption of convexity is crucial here!
        out[:] = np.linalg.norm(x[:,:,None] - y[:,None], axis=-1)

class Cube(FlatManifold):
    """ Evenly triangulate a cubical area around the origin

    >>> mfd = Cube(width, l)
    >>> # width : width of the cubical area
    >>> # l : number of grid points per direction
    """
    def __init__(self, width, l):
        assert l > 1
        self.n_dimlabels = l
        self.delta = width/(l-1)
        verts = np.zeros((l*l*l, 3), dtype=np.float64)
        verts[:] = np.mgrid[0:l,0:l,0:l].transpose((1,2,3,0)).reshape(-1,3)
        verts[:] = self.delta*verts - width/2

        nsimplices = 5*(l - 1)**3
        tris = np.zeros((nsimplices, 4), dtype=np.int64, order='C')
        i = np.arange(l-1)
        i1, i2, i3 = i[None,None,:], i[None,:,None], i[:,None,None]
        k = (4*((l-1)**2*i1 + (l-1)*i2 + i3)).ravel()
        tris[k + 0,:] = np.vstack([idx.ravel() for idx in
            [l**2* i1    + l* i2    +  i3   , l**2*(i1+1) + l* i2    +  i3   ,
             l**2* i1    + l*(i2+1) +  i3   , l**2* i1    + l* i2    + (i3+1),]]).T
        tris[k + 1,:] = np.vstack([idx.ravel() for idx in
            [l**2*(i1+1) + l*(i2+1) +  i3   , l**2*(i1+1) + l*(i2+1) + (i3+1),
             l**2* i1    + l*(i2+1) +  i3   , l**2*(i1+1) + l* i2    +  i3   ,]]).T
        tris[k + 2,:] = np.vstack([idx.ravel() for idx in
            [l**2* i1    + l*(i2+1) + (i3+1), l**2*(i1+1) + l*(i2+1) + (i3+1),
             l**2* i1    + l*(i2+1) +  i3   , l**2* i1    + l* i2    + (i3+1),]]).T
        tris[k + 3,:] = np.vstack([idx.ravel() for idx in
            [l**2*(i1+1) + l* i2    + (i3+1), l**2*(i1+1) + l*(i2+1) + (i3+1),
             l**2*(i1+1) + l* i2    +  i3   , l**2* i1    + l* i2    + (i3+1),]]).T
        tris[k + 4,:] = np.vstack([idx.ravel() for idx in
            [l**2* i1    + l* i2    + (i3+1), l**2*(i1+1) + l*(i2+1) + (i3+1),
             l**2*(i1+1) + l* i2    +  i3   , l**2* i1    + l*(i2+1) +  i3   ,]]).T

        FlatManifold.__init__(self, verts, simplices=tris)

class Square(FlatManifold):
    """ Evenly triangulate a quadratic area around the origin

    >>> mfd = Square(width, l)
    >>> # width : width of the quadratic area
    >>> # l : number of grid points per direction
    """
    def __init__(self, width, l):
        assert l > 1
        self.n_dimlabels = l
        self.delta = width/(l-1)
        verts = np.zeros((l*l, 2), dtype=np.float64)
        verts[:] = np.mgrid[0:l,0:l].transpose((1,2,0)).reshape(-1,2)
        verts[:] = self.delta*verts - width/2

        nsimplices = 2*(l - 1)**2
        tris = np.zeros((nsimplices, 3), dtype=np.int64, order='C')
        i, j = np.arange(l-1)[None,:], np.arange(l-1)[:,None]
        k = (2*(l-1)*i + 2*j).ravel()
        tris[k + 0,:] = np.vstack([idx.ravel() for idx in
            [l*i + j, l*(i+1) +  j   , l*(i+1) + (j+1)]]).T
        tris[k + 1,:] = np.vstack([idx.ravel() for idx in
            [l*i + j, l*(i+1) + (j+1), l* i    + (j+1)]]).T

        FlatManifold.__init__(self, verts, simplices=tris)

class Disk(FlatManifold):
    """ Evenly triangulate a circular area around the origin

    >>> mfd = Disk(width, l)
    >>> # width : diameter of the largest disk that fits completely inside
    >>> #         the triangulated area (2*apothem)
    >>> # l : approximate number of grid points per direction
    """
    def __init__(self, width, l):
        delta_i = 4
        i0 = 4 if l%2 == 0 else 1
        imax = (l+1)//2 - 1
        nmax = i0 + imax*delta_i
        diameter = width/np.cos(np.pi/nmax)
        delta_r = diameter/(l-1)
        r0 = delta_r/2 if l%2 == 0 else 0
        verts_x, verts_y = [], []
        for i in range(imax + 1):
            n = i0 + i*delta_i
            n -= n%2 if i > 0 else 0
            r = r0 + i*delta_r
            phi = np.arange(n)/n
            verts_x.append(r*np.cos(2*np.pi*phi))
            verts_y.append(r*np.sin(2*np.pi*phi))
        verts = np.vstack([np.concatenate(verts_x),np.concatenate(verts_y)]).T
        verts = np.ascontiguousarray(verts)
        FlatManifold.__init__(self, verts)

class Interval(FlatManifold):
    """ Evenly partition an interval around the origin

    >>> mfd = Interval(width, l)
    >>> # width : width of the interval
    >>> # l : number of grid points
    """
    def __init__(self, width, l):
        assert l > 1
        self.n_dimlabels = l
        self.delta = width/(l-1)
        FlatManifold.__init__(self, self.delta*np.mgrid[0:l] - width/2)
