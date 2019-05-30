
import numpy as np

from mflift.manifolds import DiscretizedManifold

class KleinBottle(DiscretizedManifold):
    """ Flat 2-dimensional Klein bottle """
    ndim = 2

    def mesh(self, h):
        """ Return a triangular grid on the Klein bottle.

        Args:
            h : float or pair of floats
                Step widths in radial and axial components of the strip.
        """
        if np.isscalar(h):
            h = (h,h)
        assert len(h) == 2
        tres = max(2, int(np.ceil(2*np.pi/h[0])))
        phires = max(2, int(np.ceil(2*np.pi/h[1])))

        th = 2*np.pi/tres
        phih = 2*np.pi/phires
        t, phi = np.meshgrid(np.arange(0, 2*np.pi, th),
                             np.arange(0, 2*np.pi, phih))
        v = np.vstack((phi.ravel(order='C'), t.ravel(order='C'))).T
        verts = np.ascontiguousarray(v)

        ti, phii = np.meshgrid(np.arange(0, tres), np.arange(0, phires))
        phi00, t00 = klein_idx_normalize(phii + 0, ti + 0, phires, tres)
        phi10, t10 = klein_idx_normalize(phii + 1, ti + 0, phires, tres)
        phi01, t01 = klein_idx_normalize(phii + 0, ti + 1, phires, tres)
        phi11, t11 = klein_idx_normalize(phii + 1, ti + 1, phires, tres)

        M_tris = 2*phires*tres
        simplices = np.zeros((M_tris, 3), order='C', dtype=np.int64)
        simplices[:] = np.concatenate((
            np.vstack(((phi00*tres + t00).ravel(order='C'),
                       (phi01*tres + t01).ravel(order='C'),
                       (phi11*tres + t11).ravel(order='C'),)).T,
            np.vstack(((phi00*tres + t00).ravel(order='C'),
                       (phi10*tres + t10).ravel(order='C'),
                       (phi11*tres + t11).ravel(order='C'),)).T,
            ), axis=0)
        return verts, simplices

    def _log(self, location, pfrom, out):
        [klein_normalize(v.reshape(-1,2)) for v in [location, pfrom]]
        diff = out.copy()
        curd = np.zeros(out.shape[:-1])
        bestd = curd.copy()
        bestd.fill(np.inf)
        for tofs in [-1,0,1]:
            for phiofs in [2*np.pi,0,2*np.pi]:
                xk = pfrom.copy()
                if tofs != 0:
                    xk[:,:,0] += tofs*2*np.pi
                    xk[:,:,1] *= -1
                xk[:,:,1] += phiofs
                diff[:] = xk[:,None] - location[:,:,None]
                curd[:] = np.linalg.norm(diff, axis=-1)
                ind = (curd < bestd)
                out[ind,:] = diff[ind,:]
                bestd[:] = np.fmin(bestd, curd)

    def _exp(self, location, vfrom, out):
        out[:] = location[:,:,None] + vfrom[:,None]
        klein_normalize(out.reshape(-1,2))

    def _dist(self, x, y, out):
        out[:] = np.linalg.norm(self.log(x, y), axis=-1)

    def embed(self, x):
        """ Convert Klein bottle coordinates to cartesian coordinates in R^3

        Args:
            x : ndarray of floats, shape (npoints, 2)

        Returns:
            ndarray of floats, shape (npoints, 3)
        """
        multi = (x.ndim == 2)
        x = x if multi else x[None]
        klein_normalize(x)

        v, u = 2*x[:,0], x[:,1] + 0.5*np.pi
        ibot = np.nonzero(np.logical_and(v >=       0, v <   np.pi))[0]
        imid = np.nonzero(np.logical_and(v >=   np.pi, v < 2*np.pi))[0]
        itop = np.nonzero(np.logical_and(v >= 2*np.pi, v < 3*np.pi))[0]
        ihan = np.nonzero(np.logical_and(v >= 3*np.pi, v < 4*np.pi))[0]
        result = np.zeros((x.shape[0], 3), dtype=np.float64)
        result[ibot,:] = np.vstack((
            (2.5 + 1.5*np.cos(np.pi - v[ibot]))*np.cos(u[ibot]),
            (2.5 + 1.5*np.cos(np.pi - v[ibot]))*np.sin(u[ibot]),
                  -2.5*np.sin(np.pi - v[ibot])                   ,)).T
        result[imid,:] = np.vstack((
            (2.5 + 1.5*np.cos(v[imid] - np.pi))*np.cos(u[imid]),
            (2.5 + 1.5*np.cos(v[imid] - np.pi))*np.sin(u[imid]),
                     3*      (v[imid] - np.pi)                   ,)).T
        result[itop,:] = np.vstack((
                  2 + (2 + np.cos(np.pi - u[itop]))*np.cos(3*np.pi - v[itop]),
                           np.sin(np.pi - u[itop])                             ,
            3*np.pi + (2 + np.cos(np.pi - u[itop]))*np.sin(3*np.pi - v[itop]),)).T
        result[ihan,:] = np.vstack((
            2 - 2*np.cos(4*np.pi - v[ihan]) + np.sin(u[ihan] - np.pi/2),
                                                np.cos(u[ihan] - np.pi/2),
                3*      (4*np.pi - v[ihan])                              ,)).T
        return result if multi else result[0]

def klein_normalize(x):
    """ Normalizes values (phi, t) such that always 0 <= phi,t < 2*pi (in place)

        (phi, t) -> (phi - 2*pi, -t)
        (phi, t) -> (phi + 2*pi, -t)
        t %= 2*pi

    Args:
        x : ndarray of floats, shape (npoints, 2)
    """
    multi = (x.ndim == 2)
    x = x if multi else x[None]
    while True:
        ind1 = np.nonzero(x[:,0] >= 2*np.pi)
        x[ind1,0] -= 2*np.pi
        x[ind1,1] *= -1
        ind2 = np.nonzero(x[:,0] < 0)
        x[ind2,0] += 2*np.pi
        x[ind2,1] *= -1
        if ind1[0].size == 0 and ind2[0].size == 0:
            break
    x[:,1] %= 2*np.pi
    return x if multi else x[0]

def klein_idx_normalize(phiind, tind, phires, tres):
    """ Normalizes indices (phii, ti) such that always 0 <= phii < phires
        and 0 <= ti < tres (in place)

        ti %= tres
        (phii, ti) -> (phii - phires, tres - 1 - ti)
        (phii, ti) -> (phii + phires, tres - 1 - ti)

    Args:
        phiind : ndarray of ints, shape (npoints,)
        tind : ndarray of ints, shape (npoints,)
        phires : int
        tres : int

    Returns:
        phiind : ndarray of ints, shape (npoints,)
        tind : ndarray of ints, shape (npoints,)
    """
    tind %= tres
    while True:
        ind1 = np.nonzero(phiind >= phires)
        phiind[ind1] -= phires
        tind[ind1] = tres - 1 - tind[ind1]
        ind2 = np.nonzero(phiind < 0)
        phiind[ind2] += phires
        tind[ind2] = tres - 1 - tind[ind2]
        if ind1[0].size == 0 and ind2[0].size == 0:
            break
    return phiind, tind