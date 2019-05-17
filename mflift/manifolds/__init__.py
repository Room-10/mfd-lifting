
import logging
import numpy as np

from mflift.tools.linalg import gramschmidt
from mflift.util import cached_property, broadcast_io

class DiscretizedManifold(object):
    """ Base class describing a triangulated manifold of dimension `ndim`
        embeded into `nembdim`-dimensional euclidean space. """

    ndim = None
    verts = None
    simplices = None

    def __init__(self):
        self.nverts, self.nembdim = self.verts.shape
        self.nsimplices = self.simplices.shape[0]
        assert self.simplices.shape[1] == self.ndim + 1

    @broadcast_io(1,1)
    def log(self, location, pfrom, out=None):
        """ Inverse exponential map at `location` evaluated at `pfrom`

        Args:
            location : ndarray of floats, shape (nembdim,)
            pfrom : ndarray of floats, shape (nembdim,)

        Returns:
            ndarray of floats, shape (nembdim,)
        """
        nbatch, nlocations, nembdim = location.shape
        npoints = pfrom.shape[1]
        if out is None:
            out = np.zeros((nbatch, nlocations, npoints, nembdim))
        self._log(location, pfrom, out)
        return out

    @broadcast_io(1,1)
    def exp(self, location, vfrom, out=None):
        """ Inverse exponential map at `location` evaluated at `vfrom`

        Args:
            location : ndarray of floats, shape (nembdim,)
            vfrom : ndarray of floats, shape (nembdim,)

        Returns:
            ndarray of floats, shape (nembdim,)
        """
        nbatch, nlocations, nembdim = location.shape
        nvectors = vfrom.shape[1]
        if out is None:
            out = np.zeros((nbatch, nlocations, nvectors, nembdim))
        self._exp(location, vfrom, out)
        return out

    @broadcast_io((2,1),1)
    def mean(self, points, weights, out=None):
        """ Calculate arithmetic (geodesic) means of points on the manifold.

        Args:
            points : ndarray of floats, shape (npoints, nembdim)
                The first axis can be omitted.
            weights : ndarray of floats, shape (npoints,)
                The first axis can be omitted.
        Returns:
            ndarray of floats, shape (nembdim,)
        """
        nbatch, npointsets, npoints, nembdim = points.shape
        nweights = weights.shape[1]
        if out is None:
            out = np.zeros((nbatch, npointsets, nweights, nembdim))
        self._mean(points, weights, out)
        return out

    def _mean(self, points, weights, out, max_iter=200):
        nbatch, npointsets, npoints, nembdim = points.shape
        nweights = weights.shape[1]

        if nbatch*npointsets*nweights*npoints > 100:
            # Log only if it will take a bit longer...
            logging.info(("Computing {N} means of {npoints} "
                  "points in at most {maxiter} steps...").format(
                    N=nbatch*npointsets*nweights,
                    npoints=npoints, maxiter=max_iter))

        for i in range(nbatch):
            out[i] = points[i,:,np.argmax(weights[i], axis=-1),:].transpose(1,0,2)

        out_flat = out.reshape((nbatch*npointsets, nweights, nembdim))
        out_flat2 = out.reshape((nbatch*npointsets*nweights, nembdim))
        tmean = out.copy()
        tmean_flat = tmean.reshape((nbatch*npointsets, nweights, nembdim))
        tmean_flat2 = tmean.reshape((nbatch*npointsets*nweights, nembdim))
        tpoints = np.zeros((nbatch, npointsets, nweights, npoints, nembdim))
        tpoints_flat = tpoints.reshape((nbatch*npointsets, nweights, npoints, nembdim))
        points_flat = points.reshape((nbatch*npointsets, npoints, nembdim))

        w_sum_inv = 1.0/np.einsum('ikl->ik', weights)
        for _iter in range(max_iter):
            self.log(out_flat, points_flat, out=tpoints_flat)
            np.einsum('ikm,ilkmt->ilkt', weights, tpoints, out=tmean)
            tmean *= w_sum_inv[:,None,:,None]
            out_flat2[:] = self.exp(out_flat2, tmean_flat2)

    @broadcast_io(1,0)
    def dist(self, x, y, out=None):
        """ Compute geodesic distance of points `x` and `y` on the manifold

        Args:
            x : ndarray of floats, shape (nembdim,)
            y : ndarray of floats, shape (nembdim,)

        Returns:
            ndarray of floats, shape ()
        """
        nbatch, nx, nembdim = x.shape
        ny = y.shape[1]
        if out is None:
            out = np.zeros((nbatch, nx, ny))
        self._dist(x, y, out)
        return out

    @cached_property
    def mean_derivative_matrices(self):
        """ Matrices for mean gradient computation

        Given values on the vertices of a simplex, there is a unique affine
        map in the tangent space of the simplex's mean point whose log-pullback
        to the manifold attains these values at the resp. vertices.

        For each simplex, there is a matrix mapping tuples `a` of real
        values to the corresponding tangential gradient vector `g` at the
        simplex's mean point in the above sense.

        Implicitely, if v[i] are the vertices of simplex `j`, we have

            <log(v[i])-log(v[0]),g> = a[i] - a[0] for i=1,...,ndim

        or, in short: Ad[j].dot(g) = B[j].dot(a) for some matrices Ad[j] and
        B[j]. This method computes these matrices.

        Returns:
            B : ndarray of floats, shape (nsimplices, ndim, ndim+1)
            Ad : ndarray of floats, shape (nsimplices, ndim, ndim)
        """
        B = -np.ones((self.nsimplices, self.ndim, self.ndim+1), order='C')
        B[:,:,1:] = np.eye(self.ndim)[None]
        Ad = np.zeros((self.nsimplices, self.ndim, self.ndim), order='C')
        v = self.log(self.sim_means[:,None,:], self.verts[self.simplices])[:,0]
        bases = v[:,1:] - v[:,:1]
        Ad[:] = np.einsum('jkm,jlm->jlk', gramschmidt(bases), bases)
        return B, Ad

    @cached_property
    def mean_derivative_matrices_inv(self):
        """ B[j] is replaced by inv(Ad[j]).dot(B[j]) and Ad[j] by Id """
        B, Ad = [M.copy() for M in self.mean_derivative_matrices]
        B[:] = np.einsum('jkm,jml->jkl', np.linalg.inv(Ad), B)
        Ad[:] = np.eye(self.ndim)[None]
        return B, Ad

    @cached_property
    def sim_means(self):
        """ Each row is the mean point of a simplex in the triangulation. """
        return self.mean(self.verts[self.simplices])

    @cached_property
    def sim_tangent_bases(self):
        """ Orthonormal basis vectors spanning the triangulation's simplices

        Returns:
            ndarray of floats, shape (nsimplices,ndim,nembdim)
        """
        v = self.verts[self.simplices]
        bases = np.zeros((self.nsimplices, self.ndim, self.nembdim), order='C')
        bases[:] = v[:,1:] - v[:,:1]
        return gramschmidt(bases)

    @cached_property
    def sim_derivative_matrices(self):
        """ Matrices for linear gradient computation

        Given values on the vertices of a simplex, there is a unique affine
        map x -> <Qg,x-v> + b that attains these values at the resp. vertices.
        Here, Q is an orthonormal basis matrix of the simplex and v is one of
        the simplex's vertices.

        For each simplex, there is a matrix mapping tuples `a` of real
        values to the corresponding tangential gradient vector `g` in the above
        sense.

        Implicitely, if v[i] are the vertices of simplex `j`, we have

            <v[i]-v[0],Q[j]g> = a[i] - a[0] for i=1,...,ndim

        or, in short: Ad[j].dot(g) = B[j].dot(a) for some matrices Ad[j] and
        B[j]. This method computes these matrices.

        Returns:
            B : ndarray of floats, shape (nsimplices, ndim, ndim+1)
            Ad : ndarray of floats, shape (nsimplices, ndim, ndim)
        """
        B = -np.ones((self.nsimplices, self.ndim, self.ndim+1), order='C')
        B[:,:,1:] = np.eye(self.ndim)[None]
        Ad = np.zeros((self.nsimplices, self.ndim, self.ndim), order='C')
        v = self.verts[self.simplices[:,1:]] - self.verts[self.simplices[:,:1]]
        Ad[:] = np.einsum('jkm,jlm->jlk', self.sim_tangent_bases, v)
        return B, Ad

    @cached_property
    def sim_derivative_matrices_inv(self):
        """ B[j] is replaced by inv(Ad[j]).dot(B[j]) and Ad[j] by Id """
        B, Ad = [M.copy() for M in self.sim_derivative_matrices]
        B[:] = np.einsum('jkm,jml->jkl', np.linalg.inv(Ad), B)
        Ad[:] = np.eye(self.ndim)[None]
        return B, Ad
