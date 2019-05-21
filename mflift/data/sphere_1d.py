
import numpy as np

from mflift.data import ManifoldValuedData
from mflift.manifolds.sphere import Sphere
from mflift.tools.image import cell_centered_grid

class Data(ManifoldValuedData):
    name = "sphere-1d"
    d_image = 1
    imagedims = (50,)
    imageh = (1.0,)
    l_dimsubls = 10
    rhoDomain = np.array([[0.0,1.0]])

    def __init__(self, *args, **kwargs):
        self.N = self.N_image = np.prod(self.imagedims)
        self.rhoResolution = (self.N_image,)
        self.rhoGrid, h = cell_centered_grid(self.rhoDomain, self.rhoResolution)
        ManifoldValuedData.__init__(self, Sphere(2*np.pi/5), *args, **kwargs)

    def curve(self, t):
        return np.hstack((np.cos(3*t*np.pi)*np.sin(0.5*t*np.pi),
                          np.sin(3*t*np.pi)*np.sin(0.5*t*np.pi),
                          np.cos(0.5*t*np.pi)))

    def rho_x(self, x, z):
        return 0.5*self.mfd.dist(self.curve(x)[None], z[None])[0]**2
