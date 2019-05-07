
import numpy as np

from mflift.tools.tri import SquareTriangulation
from mflift.data import ManifoldValuedData

from mflift.tools.image import cell_centered_grid

class Data(ManifoldValuedData):
    name = "flat-1d"
    d_image = 1
    imagedims = (10,)
    imageh = (1.0,)
    l_dimsubls = 5
    rhoDomain = np.array([[0.0,1.0]])

    def __init__(self, *args, **kwargs):
        self.N = self.N_image = np.prod(self.imagedims)
        self.rhoResolution = (self.N_image,)
        self.rhoGrid, h = cell_centered_grid(self.rhoDomain, self.rhoResolution)
        tri = SquareTriangulation(2.1, 5)
        ManifoldValuedData.__init__(self, tri, *args, **kwargs)

    def curve(self, t):
        return np.hstack((t*np.cos(4*t*np.pi), t*np.sin(4*t*np.pi)))

    def rho_x(self, x, z):
        return np.sum(0.5*(self.curve(x)[:,None,:] - z)**2, axis=-1)
