
import itertools
import numpy as np

from mflift.manifolds import DiscretizedManifold
from mflift.tools.linalg import expm_sym, logm_spd

class P2(DiscretizedManifold):
    """ Manifold P(2) of 2-d symmetric positive definite matrices """
    ndim = 3
    nembdim = 4

    def __init__(self, stretch, h):
        """ Setup a simplicial grid on P(2).

        Args:
            stretch : maximal eigenvalue of a matrix in the triangulation.
            h : maximal length of edges in the triangulation
        """
        self.stretch = stretch
        DiscretizedManifold.__init__(self, h)

    def mesh(self, h):
        """ By Cholesky decomposition, A = L D L.T, where D has positive
            diagonal and L is lower triangular with 1 on the diagonal.
            Contrary to the SVD, this representation is unique.
            In 2D it follows, that

                A = [[   b,         a*b],
                     [ a*b,  b*a**2 + c]]

             for unique scalars a,b,c with b,c > 0.
        """
        l = int(np.ceil(1 + self.stretch/h))
        delta = self.stretch/(l-1)
        abc = np.zeros((l*l*l, 3), dtype=np.float64)
        abc[:] = np.mgrid[0:l,0:l,0:l].transpose((1,2,3,0)).reshape(-1,3)
        abc[:,0] = delta*abc[:,0] - self.stretch/2
        abc[:,1:] = 1.0/self.stretch + delta*abc[:,1:]
        a, b, c = np.hsplit(abc, 3)
        verts = np.hstack(( b, a*b, a*b, b*a**2 + c))

        nsimplices = 5*(l - 1)**3
        sims = np.zeros((nsimplices, 4), dtype=np.int64, order='C')
        i = np.arange(l-1)
        i1, i2, i3 = i[None,None,:], i[None,:,None], i[:,None,None]
        k = (5*((l-1)**2*i1 + (l-1)*i2 + i3)).ravel()
        sims[k + 0,:] = np.vstack([idx.ravel() for idx in
            [l**2* i1    + l* i2    +  i3   , l**2*(i1+1) + l* i2    +  i3   ,
             l**2* i1    + l*(i2+1) +  i3   , l**2* i1    + l* i2    + (i3+1),]]).T
        sims[k + 1,:] = np.vstack([idx.ravel() for idx in
            [l**2*(i1+1) + l*(i2+1) +  i3   , l**2*(i1+1) + l*(i2+1) + (i3+1),
             l**2* i1    + l*(i2+1) +  i3   , l**2*(i1+1) + l* i2    +  i3   ,]]).T
        sims[k + 2,:] = np.vstack([idx.ravel() for idx in
            [l**2* i1    + l*(i2+1) + (i3+1), l**2*(i1+1) + l*(i2+1) + (i3+1),
             l**2* i1    + l*(i2+1) +  i3   , l**2* i1    + l* i2    + (i3+1),]]).T
        sims[k + 3,:] = np.vstack([idx.ravel() for idx in
            [l**2*(i1+1) + l* i2    + (i3+1), l**2*(i1+1) + l*(i2+1) + (i3+1),
             l**2*(i1+1) + l* i2    +  i3   , l**2* i1    + l* i2    + (i3+1),]]).T
        sims[k + 4,:] = np.vstack([idx.ravel() for idx in
            [l**2* i1    + l* i2    + (i3+1), l**2*(i1+1) + l*(i2+1) + (i3+1),
             l**2*(i1+1) + l* i2    +  i3   , l**2* i1    + l*(i2+1) +  i3   ,]]).T

        return verts, sims

    def _log(self, location, pfrom, out):
        """ exp_l^{-1}(p) = l^(1/2) logm(l^(-1/2) p l^(-1/2)) l^(1/2) """
        location = location.reshape(location.shape[:-1] + (2,2))
        pfrom = pfrom.reshape(pfrom.shape[:-1] + (2,2))
        out = out.reshape(out.shape[:-1] + (2,2))
        location = np.broadcast_to(location[:,:,None], out.shape)
        pfrom = np.broadcast_to(pfrom[:,None], out.shape)

        U, S = np.linalg.svd(location)[:2]
        S = np.sqrt(np.fmax(S, 0))
        sL = np.einsum('...ik, ...k, ...jk -> ...ij', U, S, U)
        sLinv = np.einsum('...ik, ...k, ...jk -> ...ij', U, 1.0/S, U)
        A = np.einsum("...ik,...kl,...lj->...ij", sLinv, pfrom, sLinv)
        S, U = np.linalg.eig(0.5*(A.swapaxes(-2,-1) + A))
        S = np.log(np.fmax(np.spacing(1), S))
        np.einsum("...ab,...bc,...c,...dc,...de->...ae", sL, U, S, U, sL, out=out)
        out[:] = 0.5*(out.swapaxes(-2,-1) + out)

    def _exp(self, location, vfrom, out):
        """ exp_l(v) = l^(1/2) expm(l^(-1/2) v l^(-1/2)) l^(1/2) """
        location = location.reshape(location.shape[:-1] + (2,2))
        vfrom = vfrom.reshape(vfrom.shape[:-1] + (2,2))
        out = out.reshape(out.shape[:-1] + (2,2))
        location = np.broadcast_to(location[:,:,None], out.shape)
        vfrom = np.broadcast_to(vfrom[:,None], out.shape)

        U, S = np.linalg.svd(location)[:2]
        S = np.sqrt(np.fmax(S, 0))
        sL = np.einsum('...ik, ...k, ...jk -> ...ij', U, S, U)
        sLinv = np.einsum('...ik, ...k, ...jk -> ...ij', U, 1.0/S, U)
        A = np.einsum("...ik,...kl,...lj->...ij", sLinv, vfrom, sLinv)
        S, U = np.linalg.eig(0.5*(A.swapaxes(-2,-1) + A))
        np.einsum("...ab,...bc,...c,...dc,...de->...ae", sL, U, np.exp(S), U, sL, out=out)
        out[:] = 0.5*(out.swapaxes(-2,-1) + out)

    def _dist(self, x, y, out):
        x = x.reshape(x.shape[:-1] + (2,2))
        y = y.reshape(y.shape[:-1] + (2,2))
        x = np.broadcast_to(x[:,:,None], out.shape + (2,2))
        y = np.broadcast_to(y[:,None], out.shape + (2,2))
        S = logm_spd(np.linalg.solve(x, y))
        out[:] = np.sqrt(np.einsum("...ij,...ij->...", S, S))

    def embed(self, x):
        return x
