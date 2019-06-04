
import numpy as np
from scipy.spatial import SphericalVoronoi

from mflift.manifolds import DiscretizedManifold
from mflift.tools.linalg import normalize

class Sphere(DiscretizedManifold):
    """ 2-dimensional sphere embedded into 3-dimensional euclidean space """
    ndim = 2

    def __init__(self, h, verts=None):
        """ Setup a triangular grid on the 2-sphere.

        Args:
            h : maximal length of edges in the triangulation
            verts : ndarray of floats, shape (npoints, 3)
                or one of ["icosahedron", "octahedron", "tetrahedron"]
                or `None` to automatically find best fit for given `h`
        """
        if verts is None:
            if h >= (8/3)**0.5:
                verts = "tetrahedron"
            elif h >= 2**0.5:
                verts = "octahedron"
            else:
                verts = "icosahedron"

        if verts == "icosahedron":
            # h = 1.0514...
            self.verts, self.simplices = sphmesh_icosahedron()
        elif verts == "octahedron":
            # h = 2**0.5 = 1.4142...
            self.verts, self.simplices = sphmesh_octahedron()
        elif verts == "tetrahedron":
            # h = (8/3)**0.5 = 1.6329...
            self.verts, self.simplices = sphmesh_tetrahedron()
        else:
            assert verts.shape[1] == 3
            self.verts = normalize(verts)
            sv = SphericalVoronoi(self.verts)
            sv.sort_vertices_of_regions()
            self.simplices = np.ascontiguousarray(sv._tri.simplices)

        DiscretizedManifold.__init__(self, h)

    def mesh(self, h):
        triverts = self.verts[self.simplices]
        diffs = np.stack((triverts[:,1] - triverts[:,0],
                          triverts[:,2] - triverts[:,0],
                          triverts[:,2] - triverts[:,1],), axis=1)
        maxedgelen = np.linalg.norm(diffs, axis=-1).max()
        rep = max(0, np.ceil(np.log2(maxedgelen/h)))
        return sphmesh_refine(self.verts, self.simplices, repeat=rep)

    def _log(self, location, pfrom, out):
        """ exp_l^{-1}(p) = d(l,p)*(p - <p,l>l)/|p - <p,l>l| """
        # pl : <p,l>
        # fc : d(l,p)/|p - <p,l>l| = arccos(<p,l>)/sqrt(1 - <p,l>^2)
        # out : fc*(p - <p,l>*l)
        pl = np.clip(np.einsum('ilm,ikm->ikl', pfrom, location), -1.0, 1.0)
        pl = pl[:,:,:,None]
        fc = np.arccos(pl)/np.fmax(np.spacing(1),np.sqrt(1 - pl**2))
        out[:] = fc*(pfrom[:,None,:,:] - pl*location[:,:,None,:])

    def _exp(self, location, vfrom, out):
        """ exp_l(v) = cos(|v|) * l  + sin(|v|) * v/|v| """
        vn = np.sqrt(np.einsum('ikm,ikm->ik', vfrom, vfrom))
        vnm = np.fmax(np.spacing(1),vn[:,None,:,None])
        out[:] = np.cos(vn[:,None,:,None])*location[:,:,None,:]
        out += np.sin(vn[:,None,:,None])/vnm*vfrom[:,None,:,:]
        # normalizing prevents errors from accumulating
        return normalize(out)

    def _dist(self, x, y, out):
        np.einsum('ikm,ilm->ikl', x, y, out=out)
        out[:]= np.arccos(np.clip(out, -1.0, 1.0))

    def embed(self, x):
        return x

    def geodesic(self, x, y, N):
        t = np.linspace(0.0, 1.0, N)
        return normalize((1 - t[:,None])*x[None] + t[:,None]*y[None])

def sphmesh_tetrahedron():
    """ Spherical regular tetrahedron (4 vertices, 4 triangles).

    Returns:
        verts : ndarray of floats, shape (4,3)
            Each row corresponds to a point on the unit sphere.
        tris : ndarray of floats, shape (4,3)
            Each row defines a triangle through indices into `verts`.
    """
    Y = 2**0.5
    X = 3**0.5

    verts = np.array([
        [   0,      0,    1],
        [   0,  2/3*Y, -1/3],
        [ Y/X, -1/3*Y, -1/3],
        [-Y/X, -1/3*Y, -1/3],
    ], dtype=np.float64)

    tris = np.array([
        [ 0,  1,  2],
        [ 0,  2,  3],
        [ 0,  3,  1],
        [ 1,  2,  3],
    ])

    return normalize(verts), tris

def sphmesh_octahedron():
    """ Spherical regular octahedron (6 vertices, 8 triangles).

    Returns:
        verts : ndarray of floats, shape (6,3)
            Each row corresponds to a point on the unit sphere.
        tris : ndarray of floats, shape (8,3)
            Each row defines a triangle through indices into `verts`.
    """
    X = 0.5*(2**0.5)

    verts = np.array([
        [ 0,  0,  1],
        [ 0,  0, -1],
        [ X,  X,  0],
        [-X,  X,  0],
        [-X, -X,  0],
        [ X, -X,  0],
    ], dtype=np.float64)

    tris = np.array([
        [ 0,  2,  3],
        [ 0,  3,  4],
        [ 0,  4,  5],
        [ 0,  5,  2],
        [ 1,  2,  3],
        [ 1,  3,  4],
        [ 1,  4,  5],
        [ 1,  5,  2],
    ])

    return normalize(verts), tris

def sphmesh_icosahedron():
    """ Spherical regular icosahedron (12 vertices, 20 triangles).

    Returns:
        verts : ndarray of floats, shape (12,3)
            Each row corresponds to a point on the unit sphere.
        tris : ndarray of floats, shape (20,3)
            Each row defines a triangle through indices into `verts`.
    """
    # (X, Z) : solution to X/Z = Z/(X + Z) and 1 = X^2 + Z^2
    X = ((5 - 5**0.5)/10)**0.5
    Z = (1 - X**2)**0.5

    verts = np.array([
        [ -X, 0.0,   Z],
        [  X, 0.0,   Z],
        [ -X, 0.0,  -Z],
        [  X, 0.0,  -Z],
        [0.0,   Z,   X],
        [0.0,   Z,  -X],
        [0.0,  -Z,   X],
        [0.0,  -Z,  -X],
        [  Z,   X, 0.0],
        [ -Z,   X, 0.0],
        [  Z,  -X, 0.0],
        [ -Z,  -X, 0.0]
    ])

    tris = np.array([
        [ 0,  4,  1],
        [ 0,  9,  4],
        [ 9,  5,  4],
        [ 4,  5,  8],
        [ 4,  8,  1],
        [ 8, 10,  1],
        [ 8,  3, 10],
        [ 5,  3,  8],
        [ 5,  2,  3],
        [ 2,  7,  3],
        [ 7, 10,  3],
        [ 7,  6, 10],
        [ 7, 11,  6],
        [11,  0,  6],
        [ 0,  1,  6],
        [ 6,  1, 10],
        [ 9,  0, 11],
        [ 9, 11,  2],
        [ 9,  2,  5],
        [ 7,  2, 11]
    ])

    return normalize(verts), tris

def sphmesh_refine(verts, tris, repeat=1):
    """ Refine a sphere triangulation

    The algorithm applies the following procedure to a given triangulation:
    Each triangular face is split into four triangles using the triangle's edge
    centers as new vertices.

    Args:
        verts : ndarray of floats, shape (nverts,3)
        tris : ndarray of floats, shape (ntris,3)
        repeat : int
            The refinement procedure is iterated `repeat` times.
            If `repeat` is 0, the input is returned unchanged.

    Returns:
        verts : ndarray of floats, shape (nverts,3)
        tris : ndarray of floats, shape (ntris,3)
    """
    if repeat == 0: return verts, tris

    nverts = verts.shape[0]
    edgecenters = np.zeros((nverts, nverts), dtype=np.int64)

    newverts = [v for v in verts]
    for tri in tris:
        for e in [[tri[0],tri[1]],[tri[1],tri[2]],[tri[2],tri[0]]]:
            if edgecenters[e[0],e[1]] == 0:
                newverts.append(normalize(0.5*(verts[e[0]] + verts[e[1]])))
                edgecenters[e[0],e[1]] = nverts
                edgecenters[e[1],e[0]] = nverts
                nverts += 1
    verts = np.asarray(newverts)

    newtris = []
    for tri in tris:
        a = edgecenters[tri[1],tri[2]]
        b = edgecenters[tri[2],tri[0]]
        c = edgecenters[tri[0],tri[1]]
        assert a*b*c > 0
        newtris.extend([
            [tri[0], c, b],
            [tri[1], a, c],
            [tri[2], b, a],
            [     a, b, c]
        ])
    tris = np.asarray(newtris)

    return sphmesh_refine(verts, tris, repeat=repeat-1)
