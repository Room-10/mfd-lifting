
import numpy as np

from mflift.manifolds import DiscretizedManifold

class Moebius(DiscretizedManifold):
    """ Flat 2-dimensional Moebius strip """
    ndim = 2

    def __init__(self, tmax, tres, phires):
        """ Setup a triangular grid on the Moebius strip.

        Args:
            tmax : half the width of the strip
            tres : resolution in width component
            phires : resolution in angle component
        """
        th = 2*tmax/(tres - 1)
        phih = 2*np.pi/phires
        t, phi = np.meshgrid(np.linspace(-tmax, tmax, tres),
                             np.arange(0, 2*np.pi, phih))
        v = np.vstack((phi.ravel(order='C'), t.ravel(order='C'))).T
        self.verts = np.ascontiguousarray(v)

        ti, phii = np.meshgrid(np.arange(0, tres - 1), np.arange(0, phires))
        phi00, t00 = moeb_idx_normalize(phii + 0, ti + 0, phires, tres)
        phi10, t10 = moeb_idx_normalize(phii + 1, ti + 0, phires, tres)
        phi01, t01 = moeb_idx_normalize(phii + 0, ti + 1, phires, tres)
        phi11, t11 = moeb_idx_normalize(phii + 1, ti + 1, phires, tres)

        M_tris = 2*phires*(tres - 1)
        self.simplices = np.zeros((M_tris, 3), order='C', dtype=np.int64)
        self.simplices[:] = np.concatenate((
            np.vstack(((phi00*tres + t00).ravel(order='C'),
                       (phi01*tres + t01).ravel(order='C'),
                       (phi11*tres + t11).ravel(order='C'),)).T,
            np.vstack(((phi00*tres + t00).ravel(order='C'),
                       (phi10*tres + t10).ravel(order='C'),
                       (phi11*tres + t11).ravel(order='C'),)).T,
            ), axis=0)

        DiscretizedManifold.__init__(self)

    def _log(self, location, pfrom, out):
        out[:] = pfrom[:,None] - location[:,:,None]
        fact1 = np.heaviside(out[...,0] - np.pi, 0)
        fact2 = np.heaviside(-out[...,0] - np.pi, 0)
        out[...,0] -= (fact1 - fact2)*2*np.pi
        out[...,1] -= (fact1 + fact2)*pfrom[:,None,:,1]

    def _exp(self, location, vfrom, out):
        out[:] = location[:,:,None] + vfrom[:,None]
        moeb_normalize(out.reshape(-1,2))

    def _dist(self, x, y, out):
        out[:] = np.linalg.norm(self.log(x, y), axis=-1)

    def embed(self, x):
        """ Convert Moebius coordinates to cartesian coordinates in R^3

        Args:
            x : ndarray of floats, shape (npoints, 2)

        Returns:
            ndarray of floats, shape (npoints, 3)
        """
        multi = (x.ndim == 2)
        x = x if multi else x[None]
        moeb_normalize(x)
        result = np.vstack((np.cos(x[:,0]) + x[:,1]*np.cos(x[:,0]/2.0),
                            np.sin(x[:,0]) + x[:,1]*np.cos(x[:,0]/2.0),
                                             x[:,1]*np.sin(x[:,0]/2.0),)).T
        return result if multi else result[0]

def moeb_normalize(x):
    """ Normalizes values (phi, t) such that always 0 <= phi < 2 PI (in place)

        (phi, t) -> (phi - 2 PI, -t)
        (phi, t) -> (phi + 2 PI, -t)

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
    return x if multi else x[0]

def moeb_idx_normalize(phiind, tind, phires, tres):
    """ Normalizes indices (phii, ti) such that always 0 <= phii < phires (in place)

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
