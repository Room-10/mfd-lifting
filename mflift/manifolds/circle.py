
import numpy as np

from mflift.manifolds import DiscretizedManifold

class Circle(DiscretizedManifold):
    """ Flat 1-dimensional circle S^1 """
    ndim = 1

    def mesh(self, h):
        """ Return a discretization of the interval [0,2*pi]

        Args:
            h : float
                Step width in angular component (radians).
        """
        L_labels = M_tris = max(2, int(np.ceil(2*np.pi/h)))
        verts = np.linspace(0, 2*np.pi, L_labels+1)[:-1,None]
        verts = np.array(verts, dtype=np.float64, order='C')
        num = np.arange(L_labels)
        simplices = np.zeros((M_tris, 2), order='C', dtype=np.int64)
        simplices[:] = np.vstack((num, np.roll(num, -1))).T
        return verts, simplices

    def _log(self, location, pfrom, out):
        out[:] = pfrom[:,None] - location[:,:,None]
        fact1 = np.heaviside(out[...,0] - np.pi, 0)
        fact2 = np.heaviside(-out[...,0] - np.pi, 0)
        out[...,0] -= (fact1 - fact2)*2*np.pi

    def _exp(self, location, vfrom, out):
        out[:] = location[:,:,None] + vfrom[:,None]
        circ_normalize(out.reshape(-1))

    def _dist(self, x, y, out):
        out[:] = np.abs(self.log(x, y))[...,0]

    def embed(self, x):
        """ Convert angular coordinates to cartesian coordinates in R^2

        Args:
            x : ndarray of floats, shape (npoints, 1)

        Returns:
            ndarray of floats, shape (npoints, 2)
        """
        multi = (x.ndim == 2)
        x = x if multi else x[None]
        circ_normalize(x.reshape(-1))
        result = np.vstack((np.cos(x[:,0]), np.sin(x[:,0]))).T
        return result if multi else result[0]

def circ_normalize(x):
    """ Normalizes values such that always 0 <= x < 2*π (in place)

        (phi) -> (phi ± 2*π)

    Args:
        x : ndarray of floats, shape (npoints,)
    """
    while True:
        ind1 = np.nonzero(x >= 2*np.pi)
        x[ind1] -= 2*np.pi
        ind2 = np.nonzero(x < 0)
        x[ind2] += 2*np.pi
        if ind1[0].size == 0 and ind2[0].size == 0:
            break
    return x
