
import logging
import itertools
import numpy as np

from scipy.spatial import Delaunay, Voronoi

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mflift.util import cached_property

class Triangulation(object):
    """ Triangulation of an n-D point set

    >>> tri = Triangulation(verts)
    >>> # verts : ndarray of floats, shape (nverts,ndim)
    >>> # tri.simplices : ndarray of ints, shape (nsimplices,ndim+1)
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
        self.nsimplices = self.simplices.shape[0]

    def subgrid(self, nse):
        """ Create subgrid for this triangulation

        Args:
            nse : int, at least 2
                Number of subpoints per (onedimensional) edge.

        Returns:
            subverts : ndarray of floats, shape (ntotalsubs,ndim)
            subidx : ndarray of ints, shape (nsimplices,nsubs)
                Each row specifies the indices into `subverts` that belong to
                subpoints contained in the corresponding triangle.
        """
        assert nse >= 2

        subverts = barygrid(self.ndim, nse, boundary=True)[:,:-1]
        subverts = subverts[np.all(subverts != 1, axis=1)]
        subverts = np.concatenate((np.eye(self.ndim), subverts))
        subidx = np.arange(subverts.shape[0])[None,:]

        subverts = np.array(subverts, dtype=np.float64, order='C')
        subidx = np.array(subidx, dtype=np.int64, order='C')
        return subverts, subidx

    def proj(self, points):
        """ Project points onto triangulated area

        Args:
            points : ndarray of floats, shape (npoints, ndim)

        Returns:
            ppoints : ndarray of floats, shape (npoints, ndim)
        """
        import cvxpy as cp

        A, b = self.convex_hull
        x0 = cp.Constant(points)
        x = cp.Variable(points.shape)
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(cp.vec(x - x0))),
            [A*x[i] <= b for i in range(x.shape[0])])
        prob.solve(verbose=False, solver='MOSEK')
        return x.value

    def embed(self, points):
        """ Compute location of points relative to the triangulation

        Args:
            points : ndarray of floats, shape (npoints, ndim)

        Returns:
            containing_simplices : ndarray of ints, shape (npoints,)
                point `i` is contained in simplex `containing_simplices[i]`
            coords : ndarray of floats, shape (npoints, nverts)
                `coords[i,containing_simplices[i]]` are the barycentric
                coordinates of point i relative to the containing simplex.
                Moreover: `coords.dot(verts) == points`.
        """
        assert points.shape[1] == self.ndim
        npoints = points.shape[0]
        tol = 1e-14

        M = self.barycentric_coordinate_matrices
        lifted = -np.ones((npoints, self.ndim+1))
        lifted[:,0:-1] = points
        tcoords = np.einsum('jkl,il->ijk', M, lifted)
        mask_01 = np.logical_and(tcoords > -tol, tcoords < 1.0+tol)
        mask_01 = np.all(mask_01, axis=2)
        indices_01 = [w.nonzero()[0] for w in mask_01]
        outsiders = np.array([i.size == 0 for i in indices_01], dtype=bool)
        if np.any(outsiders):
            logging.info("Warning: Projecting points onto triangulated area.")
            ppoints = points.copy()
            ppoints[outsiders] = self.proj(points[outsiders])
            return self.embed(ppoints)
        idx_01 = np.array([i[0] for i in indices_01], dtype=np.int64, order='C')
        coords = np.zeros((npoints, self.nverts), order='C')
        for i,j in enumerate(idx_01):
            coords[i,self.simplices[j]] = tcoords[i,j]
        return idx_01, coords

    def mean(self, points, weights=None):
        multi_p = points.ndim > 2
        points = points[None] if not multi_p else points
        npoints = points.shape[1]

        weights = np.ones(npoints)/npoints if weights is None else weights
        multi_w = weights.ndim > 1
        weights = weights[None] if not multi_w else weights

        result = np.einsum("kl,jlt->jkt", weights, points)
        result = result if multi_w else result[:,0,:]
        result = result if multi_p else result[0]
        return result

    @cached_property
    def convex_hull(self):
        """ Equations describing facets of the triangulation's convex hull

        Returns:
            normals : ndarray of floats, shape (nfacets, ndim)
                Outer normals
            offsets : ndarray of floats, shape (nfacets,)
                Points x inside the convex hull satisfy `normals*x <= offsets`
        """
        if self.ndim == 1:
            ch_idx = np.array([[self.verts.argmin()],[self.verts.argmax()]])
        else:
            ch_idx = self._delaunay.convex_hull
        ch_verts = self.verts[ch_idx]
        p0 = ch_verts.sum(axis=(0,1))/(self.ndim*ch_verts.shape[0])
        normals = hyperplane_normals(ch_verts[:,1:,:] - ch_verts[:,0:1,:])
        normals /= np.linalg.norm(normals, axis=1)[:,None]
        orientation = np.einsum("ik,ik->i", normals, p0 - ch_verts[:,0,:])
        normals[orientation > 0] *= -1
        offsets = np.einsum("ik,ik->i", normals, ch_verts[:,0,:])
        return normals, offsets

    @cached_property
    def barycentric_coordinate_matrices(self):
        """ Matrices for euclidean to barycentric coordinate transformation

        For a simplex (v[0],...,v[d]) each point x inside of it can be
        described via unique "barycentric" coordinates (a[0],...,a[d]) as

            x = sum_i a[i]*v[i]   s.t.  sum_i a[i] = 1

        For a fixed simplex, the transformation from euclidean coordinates to
        barycentric coordinates is an affine linear map.

        Returns:
            mats : ndarray of floats, shape (nsimplices, ndim+1, ndim+1)
                The matrix `M = mats[j,:,:]` describes the transformation from
                euclidean coordinates `x` to barycentric coordinates `a`
                relative to triangle `j` as

                    a[k] = sum(M[k,:-1]*x[:]) - M[k,-1]
        """
        mats = np.zeros((self.nsimplices, self.ndim+1, self.ndim+1), order='C')
        mats[:] = np.eye(self.ndim+1)[None]
        mats[:,-1,:] = -1
        return mats
        #mats = -np.ones((self.nsimplices, self.ndim+1, self.ndim+1), order='C')
        #mats[:,:,0:-1] = self.verts[self.simplices]
        #return np.linalg.inv(mats).transpose(0,2,1)

    @cached_property
    def derivative_matrices(self):
        """ Matrices for linear gradient computation

        Given values on the vertices of a simplex, there is a unique affine
        map x -> <g,x> + b that attains these values at the resp. vertices.

        For each simplex, there is a matrix mapping tuples `a` of real
        values to the corresponding gradient vector `g` in the above sense.

        Implicitely, if v[i] are the vertices of simplex `j`, we have

            <v[i]-v[0],g> = a[i] - a[0] for i=1,...,d

        or, in short: Ad[j].dot(g) = B[j].dot(a) for some matrices Ad[j] and
        B[j]. This method computes these matrices.

        Returns:
            B : ndarray of floats, shape (nsimplices, ndim, ndim+1)
            Ad : ndarray of floats, shape (nsimplices, ndim, ndim)
        """
        B = -np.ones((self.nsimplices, self.ndim, self.ndim+1), order='C')
        B[:,:,1:] = np.eye(self.ndim)[None]
        Ad = np.zeros((self.nsimplices, self.ndim, self.ndim), order='C')
        v = self.verts[self.simplices]
        Ad[:] = v[:,1:] - v[:,0:1]
        return B, Ad

    @cached_property
    def derivative_matrices_inv(self):
        """ B[j] is replaced by inv(Ad[j]).dot(B[j]) and Ad[j] by Id """
        B, Ad = [M.copy() for M in self.derivative_matrices]
        B[:] = np.einsum('jkm,jml->jkl', np.linalg.inv(Ad), B)
        Ad[:] = np.eye(self.ndim)[None]
        return B, Ad

    @cached_property
    def concavity_matrices(self):
        """ Matrices for second order derivatives across edges

        Let t1 and t2 be neighboring simplices with common face f and let vn be
        orthogonal to f pointing from t1 to t2.
        Let a[i] be the values of a piecewise linear function on the
        vertices of t1 and t2 with gradient vectors g1 and g2.

        Then, the expression

            <g2-g1,vn>

        is a second order finite difference along vn.

        The mapping from a to (g2 - g1) is linear, hence there
        is a vector b such that

            <g2-g1,vn> = <a,b>

        This method computes the vectors b = B2[j] for each pair of neighboring
        simplices whose vertices are stored in P2[j].

        Returns:
            bdry : ndarray of bools, shape (nsimplices,)
            P2 : ndarray of ints, shape (nnbsimplices, ndim + 2)
            B2 : ndarray of floats, shape (nnbsimplices, 1, ndim + 2)
        """
        B = -np.ones((self.ndim, self.ndim+1))
        B[:,:self.ndim] = np.eye(self.ndim)
        mat = np.zeros((self.ndim,self.ndim+2))
        bdry = np.zeros((self.nsimplices,), dtype=np.int64, order='C')
        P2 = []
        B2 = []
        for j1,sim1 in enumerate(self.simplices):
            for j2 in range(j1 + 1, self.nsimplices):
                sim2 = self.simplices[j2]

                # continue if there's no common face
                e = np.intersect1d(sim1, sim2)
                if e.size != self.ndim: continue

                # register neighboring pair
                k3, k4 = np.setdiff1d(sim1, e)[0], np.setdiff1d(sim2, e)[0]
                P2.append(np.hstack((e, [k3, k4])))
                bdry[[j1,j2]] += 1

                # compute interface normal vector vn pointing from sim1 to sim2
                v = self.verts[e]
                vn = hyperplane_normals(v[None,1:] - v[None,0:1])[0]
                if vn.dot(self.verts[k4] - self.verts[k3]) < 0:
                    vn *= -1

                # determine B2[j]
                mat.fill(0.0)
                Ad1 = v - self.verts[k3][None,:]
                Ad2 = v - self.verts[k4][None,:]
                idx = list(range(self.ndim))
                mat[:,idx + [-1]] += np.linalg.inv(Ad2).dot(B)
                mat[:,idx + [-2]] -= np.linalg.inv(Ad1).dot(B)
                B2.append(mat.T.dot(vn))

        bdry = np.ascontiguousarray((bdry != self.ndim + 1))
        P2 = np.array(P2, dtype=np.int64, order='C')
        B2 = np.array(B2, dtype=np.float64, order='C')[:,None,:]
        return bdry, P2, B2

    @cached_property
    def sim_adjacency(self):
        """ (Symmetric) adjacency matrix for simplices

        Two simplices are neighboring (adjacent) if they have exactly one
        common face. A simplex is not adjacent to itself.

        Returns:
            M : ndarray of bools, shape (nsimplices, nsimplices)
        """
        M = np.zeros((self.nsimplices, self.nsimplices), dtype=np.uint8)
        for j1,sim1 in enumerate(self.simplices):
            nb_count = 0
            for j2 in range(j1 + 1, self.nsimplices):
                sim2 = self.simplices[j2]
                f = np.intersect1d(sim1, sim2)
                if f.size != self.ndim: continue
                M[j1,j2] = 1
                if nb_count >= self.ndim: break
                nb_count += 1
        return (M + M.T).astype(bool)

    @cached_property
    def vert_adjacency(self):
        """ (Symmetric) adjacency matrix for vertices

        Two vertices are neighboring (adjacent) if there is an edge in the
        triangulation connecting the two. A vertex is not adjacent to itself.

        Returns:
            M : ndarray of bools, shape (nverts, nverts)
        """
        M = np.zeros((self.nverts, self.nverts), dtype=np.uint8)
        for j,sim in enumerate(self.simplices):
            for e in itertools.combinations(sim, 2):
                e = sorted(e)
                M[e[0],e[1]] = 1
        return (M + M.T).astype(bool)

    @cached_property
    def bdry_sims(self):
        """ Indicate if a simplex is at the boundary

        Returns:
            bdry_sims : ndarray of bools, shape (nsimplices,)
        """
        counts = np.count_nonzero(self.sim_adjacency, axis=1)
        return np.ascontiguousarray(counts < 3)

    @cached_property
    def bdry_verts_mask(self):
        """ Indicate if a vertex is at the boundary

        Returns:
            mask : ndarray of bools, shape (nverts,)
        """
        mask = np.zeros((self.nverts,), dtype=bool)
        mask[self.bdry_verts] = True
        return mask

    @cached_property
    def bdry_verts(self):
        """ Indices of vertices at the boundary

        In 2D, result is in counterclockwise order.

        Returns:
            bdry_verts : ndarray of ints, shape (nbdryverts,)
        """
        if self.ndim == 1:
            bdry_verts = [self.verts.argmin(),self.verts.argmax()]
            return np.array(bdry_verts, dtype=np.int64)
        else:
            bdry_verts = np.unique(self._delaunay.convex_hull.ravel())
            if self.ndim == 2:
                bdry_verts = bdry_verts[polygon_order(self.verts[bdry_verts])]
            return bdry_verts

    @cached_property
    def vert2sims(self):
        """ Indices of simplices neighboring each vertex

        Returns:
            vert_sims : list of lists of indices into self.simplices
        """
        vert_sims = [np.any(self.simplices == k, axis=1) for k in range(self.nverts)]
        return [np.where(sims)[0] for sims in vert_sims]

    @cached_property
    def sim_vols(self):
        """ Volume of each simplex in the triangulation

        Returns:
            vol : ndarray of floats, shape (nsimplices,)
        """
        M = np.ones((self.nsimplices,self.ndim+1,self.ndim+1))
        M[:,:,0:-1] = self.verts[self.simplices]
        return np.abs(np.linalg.det(M))/np.prod(1 + np.arange(self.ndim))

    @cached_property
    def fem_vols(self):
        """ Volume elements of piecewise linear finite elements on vertices

        Returns:
            vol : ndarray of floats, shape (nverts,)
        """
        vol = np.array([np.sum(self.sim_vols[t]) for t in self.vert2sims])
        return vol/(self.ndim + 1)

class Triangulation2D(Triangulation):
    """ Triangulation of a 2D point set """

    @cached_property
    def voronoi_regions(self):
        """ Description of Voronoi regions corresponding to the triangulation

        Returns:
            vor : scipy.spatial.Voronoi object
                Doesn't contain regions for boundary points.
                `vor.vertices` is shorter than `vertices` (see below)!
            regions : list of lists of ints, shape (nregions,*)
                Number of regions is number of points in the triangulation.
                The indices are with respect to `vertices` (see below).
            vertices : ndarray of floats, shape (nverts, 2)
                The vertices of the Voronoi polygons.
                Not to be confused with the vertices/points of the triangulation.
        """
        vor = Voronoi(self.verts)
        regions = []
        vertices = vor.vertices.tolist()
        ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            ridges.setdefault(p1, []).append((p2, v1, v2))
            ridges.setdefault(p2, []).append((p1, v1, v2))

        for p1, region in enumerate(vor.point_region):
            rverts = vor.regions[region]
            if all(v >= 0 for v in rverts):
                regions.append(rverts)
                continue
            assert self.bdry_verts_mask[p1]

            rridges = ridges[p1]
            rverts = [v for v in rverts if v >= 0]
            assert len(rridges) == len(rverts) + 1
            for p2, v1, v2 in rridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0: continue
                rverts.append(len(vertices))
                vertices.append(vor.points[[p1,p2]].mean(axis=0).tolist())
            assert len(rridges) == len(rverts) - 1
            rverts.append(len(vertices))
            vertices.append(vor.points[p1])
            poly_verts = np.array([vertices[i] for i in rverts])
            regions.append([rverts[i] for i in polygon_order(poly_verts)])

        return vor, regions, np.array(vertices)

    @cached_property
    def vert_areas(self):
        """ Areas of Voronoi regions associated with each vertex

        Returns:
            vol : ndarray of floats, shape (nverts,)
        """
        _, regions, vertices = self.voronoi_regions
        return np.array([polygon_area(vertices[r]) for r in regions])

    @cached_property
    def area(self):
        """ Total area of triangulated region

        Returns:
            area : float
        """
        return polygon_area(self.verts[self.bdry_verts])

class SquareTriangulation(Triangulation2D):
    """ Evenly triangulate a quadratic area around the origin

    >>> tri = SquareTriangulation(width, l)
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

        Triangulation2D.__init__(self, verts, simplices=tris)

class DiskTriangulation(Triangulation2D):
    """ Evenly triangulate a circular area around the origin

    >>> tri = DiskTriangulation(width, l)
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
        Triangulation2D.__init__(self, verts)

class Triangulation1D(Triangulation):
    """ Evenly partition an interval around the origin

    >>> tri = Triangulation1D(width, l)
    >>> # width : width of the interval
    >>> # l : number of grid points
    """
    def __init__(self, width, l):
        assert l > 1
        self.n_dimlabels = l
        self.delta = width/(l-1)
        Triangulation.__init__(self, self.delta*np.mgrid[0:l] - width/2)

def barygrid(ndim, nse, boundary=True):
    """ Regular grid of interior barycentric (simplex) coordinates

    Args:
        ndim : int
        nse : int, at least 2
            Number of subpoints per (onedimensional) edge.
        boundary : boolean
            Optionally skip points on the boundaries.

    Returns:
        ndarray of floats, shape (ngridpoints, ndim+1)
    """
    assert nse >= 2
    grid = []
    for coord in itertools.product(range(nse), repeat=ndim):
        scoord = sum(coord)
        if scoord >= nse:
            continue
        coord += (nse - 1 - scoord,)
        if not boundary and 0 in coord:
            continue
        grid.append(coord)
    return np.array(grid, dtype=np.float64)/(nse-1)

def hyperplane_normals(planes):
    """ Compute normals of hyperplanes spanned by (d-1) vectors

    Args:
        planes : ndarray of floats, shape (nplanes, ndim-1, ndim)
            For each hyperplane ndim-1 vectors spanning the hyperplane.

    Returns:
        ndarray of floats, shape (nplanes, ndim)
    """
    nplanes, _, ndim = planes.shape
    assert planes.shape[1] == ndim-1
    tmp = np.zeros((nplanes, ndim, ndim, ndim))
    tmp[:,:,:-1,:] = planes[:,None,:,:]
    tmp[:,:,-1,:] = np.eye(ndim)[None,:,:]
    normals = np.linalg.det(tmp)
    return normals/np.linalg.norm(normals, axis=1)[:,None]

def polygon_order(verts):
    """ Sort vertices of a polygon in counterclockwise order around center

    Args:
        order : ndarray of ints, shape (nverts,)
    """
    center = verts.mean(axis=0)
    angles = np.arctan2(verts[:,1] - center[1], verts[:,0] - center[0])
    return np.argsort(angles)

def polygon_area(verts):
    """ Compute area of a (simple) polygon using the shoelace formula

    Args:
        verts : ndarray of floats, shape (nverts, 2)
            vertices in sequential (counter-)clockwise order
    """
    extra = verts[-1,0]*verts[0,1] - verts[0,0]*verts[-1,1]
    main = np.dot(verts[:-1,0], verts[1:,1]) - np.dot(verts[1:,0], verts[:-1,1])
    return 0.5*np.abs(main + extra)
