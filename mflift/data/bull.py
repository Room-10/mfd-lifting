
import numpy as np
from scipy.io import loadmat

from mflift.data import ManifoldValuedData
from mflift.manifolds.sphere import Sphere
from mflift.tools.image import cell_centered_grid

class Data(ManifoldValuedData):
    name = "bull"
    imageh = (1.0,1.0)
    rhoDomain = np.array([[0.0,1.0],[0.0,1.0]])
    l_dimsubls = 10
    filename = "data/bull-nn.mat"

    def __init__(self, *args, **kwargs):
        self.extra = loadmat(self.filename)
        self.imagedims = self.extra['uuh'].shape
        self.d_image = len(self.imagedims)
        self.I = self.extra['nn'].reshape((3,) + self.imagedims)
        self.I = self.I.transpose(2,1,0)
        self.N = self.N_image = np.prod(self.imagedims)
        self.rhoResolution = self.imagedims
        self.rhoGrid, h = cell_centered_grid(self.rhoDomain, self.rhoResolution)
        ManifoldValuedData.__init__(self, Sphere(2*np.pi/5), *args, **kwargs)

    def rho(self, z):
        I = self.I.reshape(-1,3)
        return 0.5*self.mfd.dist(I[None], z[None])[0]**2
