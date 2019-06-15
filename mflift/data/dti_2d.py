
import numpy as np

from mflift.data import ManifoldValuedData
from mflift.manifolds.p2 import P2
from mflift.tools.image import cell_centered_grid

class Data(ManifoldValuedData):
    name = "dti-2d"
    imageh = (1.0,1.0)
    rhoDomain = np.array([[0.0,1.0],[0.0,1.0]])
    l_dimsubls = 6

    def __init__(self, *args, **kwargs):
        self.I = synthetic_spd_img(10)
        self.imagedims = self.I.shape[:-1]
        self.d_image = len(self.imagedims)
        self.N = self.N_image = np.prod(self.imagedims)
        self.rhoResolution = self.imagedims
        self.rhoGrid, h = cell_centered_grid(self.rhoDomain, self.rhoResolution)
        s = 1
        h = 2*np.pi*s/5
        ManifoldValuedData.__init__(self, P2(s, h), *args, **kwargs)

    def rho(self, z):
        I = self.I.reshape(-1,4)
        return 0.5*self.mfd.dist(I[None], z[None])[0]**2

def synthetic_spd_img(res, s=8.0):
    """ Create a 2-d image of SPD-valued points of shape res*res.

    Args:
        res : int
            Resolution in each coordinate direction
        s : float
            Higher means more anisotrope ellipses

    Returns:
        ndarray of floats, shape (res, res, 4)
    """
    grid1 = np.arange(-np.floor((res-1)/2), np.ceil((res-1)/2) + 1)
    X, Y =  np.meshgrid(grid1, grid1)

    angles = np.arctan2(X, Y)
    radii = np.sqrt(Y**2 + X**2)

    U = np.zeros((res,res,2,2))
    U[:,:,0,0] =  np.cos(angles)
    U[:,:,1,1] =  np.cos(angles)
    U[:,:,1,0] =  np.sin(angles)
    U[:,:,0,1] = -np.sin(angles)
    circ = np.zeros((res,res,2,2))
    D = np.ones((res,res,2))
    D[:,:,1] += s*radii/res
    circ[:] = np.einsum("ijmk,ijm,ijml->ijkl", U, D, U)

    result = circ.copy()
    midl, midr = int(np.floor(res/2)), int(np.floor((res+1)/2))
    result[:midl,:midr] = circ[:midl,midl:]
    result[:midl,midl:] = circ[:midl,:midr]
    return result.reshape(res,res,4)
