
import numpy as np

from mflift.data import ManifoldValuedData
from mflift.manifolds.circle import Circle
from mflift.tools.image import cell_centered_grid

class Data(ManifoldValuedData):
    name = "rcom" # Riemannian center of mass
    imageh = (1.0,)
    imagedims = (3,)
    rhoDomain = np.array([[0.0,1.0]])
    l_dimsubls = 20

    def __init__(self, *args, **kwargs):
        np.random.seed(134182)
        self.points = 2*np.pi*np.random.rand(10,1)
        self.weights = np.random.rand(10)
        self.weights /= self.weights.sum()
        self.initializer = np.array([np.pi])

        self.d_image = len(self.imagedims)
        self.N = self.N_image = np.prod(self.imagedims)
        self.rhoResolution = self.imagedims
        self.rhoGrid, h = cell_centered_grid(self.rhoDomain, self.rhoResolution)
        ManifoldValuedData.__init__(self, Circle(2*np.pi/3), *args, **kwargs)

    def rho(self, z):
        dist = self.mfd.dist(self.points[None], z[None])[0]
        val = np.sum(self.weights[:,None]*dist**2, axis=0)
        return np.tile(val, (self.N_image,1))
