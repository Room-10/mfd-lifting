
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
        lphi = max(3, int(np.ceil(2*np.pi*self.stretch/h)))
        hphi = 2*np.pi/lphi
        lstretch = max(2, int(np.ceil(1 + self.stretch/h)))
        hstretch = self.stretch/(lstretch - 1)
        nsimplices = 5*lphi*(lstretch - 1)**2

        verts = np.zeros((lphi, lstretch, lstretch, 3), dtype=np.float64)
        verts[:] = np.mgrid[0:lphi,0:lstretch,0:lstretch].transpose((1,2,3,0))
        verts = verts.reshape(-1,3)
        verts[:,0] *= hphi
        verts[:,1:] = hstretch*(1 + verts[:,1:])
        angles = verts[:,0]
        vals = verts[:,1:]
        V = np.hstack((np.cos(verts[:,:1]), -np.sin(verts[:,:1]),
                       np.sin(verts[:,:1]),  np.cos(verts[:,:1]),)).reshape(-1,2,2)
        verts = np.einsum("kij,kj,kmj->kim", V, verts[:,1:], V)
        verts = np.ascontiguousarray(verts.reshape(-1,4))

        sims = np.zeros((nsimplices, 4), dtype=np.int64, order='C')
        iphi = np.arange(lphi)
        istretch = np.arange(lstretch-1)
        i1, i2, i3 = iphi[None,None,:], istretch[None,:,None], istretch[:,None,None]
        l = lstretch
        k = (5*((l-1)**2*i1 + (l-1)*i2 + i3)).ravel()
        i1p1 = np.roll(i1.ravel(), -1).reshape(i1.shape)
        sims[k + 0,:] = np.vstack([idx.ravel() for idx in
            [l**2* i1    + l* i2    +  i3   , l**2*(i1p1) + l* i2    +  i3   ,
             l**2* i1    + l*(i2+1) +  i3   , l**2* i1    + l* i2    + (i3+1),]]).T
        sims[k + 1,:] = np.vstack([idx.ravel() for idx in
            [l**2*(i1p1) + l*(i2+1) +  i3   , l**2*(i1p1) + l*(i2+1) + (i3+1),
             l**2* i1    + l*(i2+1) +  i3   , l**2*(i1p1) + l* i2    +  i3   ,]]).T
        sims[k + 2,:] = np.vstack([idx.ravel() for idx in
            [l**2* i1    + l*(i2+1) + (i3+1), l**2*(i1p1) + l*(i2+1) + (i3+1),
             l**2* i1    + l*(i2+1) +  i3   , l**2* i1    + l* i2    + (i3+1),]]).T
        sims[k + 3,:] = np.vstack([idx.ravel() for idx in
            [l**2*(i1p1) + l* i2    + (i3+1), l**2*(i1p1) + l*(i2+1) + (i3+1),
             l**2*(i1p1) + l* i2    +  i3   , l**2* i1    + l* i2    + (i3+1),]]).T
        sims[k + 4,:] = np.vstack([idx.ravel() for idx in
            [l**2* i1    + l* i2    + (i3+1), l**2*(i1p1) + l*(i2+1) + (i3+1),
             l**2*(i1p1) + l* i2    +  i3   , l**2* i1    + l*(i2+1) +  i3   ,]]).T

        # discard cases where eigenvalues are descending
        vertkeep = (vals[:,0] - vals[:,1] <= 0)
        vertdiscard = np.logical_not(vertkeep)
        newinds = np.zeros((verts.shape[0],), dtype=np.int64)
        newinds[vertkeep] = np.arange(np.count_nonzero(vertkeep))
        newinds[vertdiscard] = -1
        simkeep = np.all(newinds[sims] >= 0, axis=-1)
        verts = verts[vertkeep]
        vals = vals[vertkeep]
        angles = angles[vertkeep]
        sims = newinds[sims[simkeep]]

        # identify cases where eigenvalues are equal
        vertid = (vals[:,0] - vals[:,1] == 0)
        angles0 = vertid & (angles == 0)
        vertkeep = np.logical_not(vertid) | angles0
        newinds = np.zeros((verts.shape[0],), dtype=np.int64)
        newinds[vertkeep] = np.arange(np.count_nonzero(vertkeep))
        for i in range(verts.shape[0]):
            if vals[i,0] == vals[i,1] and angles[i] > 0:
                ind = (angles0 & (vals[:,0] == vals[i,0])).nonzero()[0][0]
                newinds[i] = newinds[ind]
        verts = verts[vertkeep]
        sims = np.unique(np.sort(newinds[sims], axis=-1), axis=0)

        return verts, sims

    def _log(self, location, pfrom, out):
        """ exp_l^{-1}(p) = x^(1/2) logm(x^(-1/2) y x^(-1/2)) x^(1/2) """
        location = location.reshape(location.shape[:-1] + (2,2))
        pfrom = pfrom.reshape(pfrom.shape[:-1] + (2,2))
        out = out.reshape(out.shape[:-1] + (2,2))
        location = np.broadcast_to(location[:,:,None], out.shape)
        pfrom = np.broadcast_to(pfrom[:,None], out.shape)
        S = logm_spd(np.linalg.solve(location, pfrom))
        S = np.einsum("...ij,...jk->...ik", location, S)
        out[:] = 0.5*(S.transpose(0,1,2,4,3) + S)

    def _exp(self, location, vfrom, out):
        """ exp_l(v) = x^(1/2) expm(x^(-1/2) y x^(-1/2)) x^(1/2) """
        location = location.reshape(location.shape[:-1] + (2,2))
        vfrom = vfrom.reshape(vfrom.shape[:-1] + (2,2))
        out = out.reshape(out.shape[:-1] + (2,2))
        location = np.broadcast_to(location[:,:,None], out.shape)
        vfrom = np.broadcast_to(vfrom[:,None], out.shape)
        S = expm_sym(np.linalg.solve(location, vfrom))
        S = np.einsum("...ij,...jk->...ik", location, S)
        out[:] = 0.5*(S.transpose(0,1,2,4,3) + S)

    def _dist(self, x, y, out):
        x = x.reshape(x.shape[:-1] + (2,2))
        y = y.reshape(y.shape[:-1] + (2,2))
        x = np.broadcast_to(x[:,:,None], out.shape + (2,2))
        y = np.broadcast_to(y[:,None], out.shape + (2,2))
        S = logm_spd(np.linalg.solve(x, y))
        out[:] = np.sqrt(np.einsum("...ij,...ij->...", S, S))

    def embed(self, x):
        return x
