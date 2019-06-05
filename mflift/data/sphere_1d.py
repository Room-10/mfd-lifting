
import numpy as np

from mflift.data import ManifoldValuedData
from mflift.manifolds.sphere import Sphere
from mflift.tools.image import cell_centered_grid

class Data(ManifoldValuedData):
    name = "sphere-1d"
    d_image = 1
    imagedims = (50,)
    imageh = (1.0,)
    rhoDomain = np.array([[0.0,1.0]])

    def __init__(self, *args, dimsubls=10, dimres=45, **kwargs):
        self.l_dimsubls = dimsubls
        sph_h = 2*np.pi/np.ceil(dimres/(dimsubls - 1))

        self.N = self.N_image = np.prod(self.imagedims)
        self.rhoResolution = (self.N_image,)
        self.rhoGrid, h = cell_centered_grid(self.rhoDomain, self.rhoResolution)
        self.I = [self.curve(self.rhoGrid)]
        ManifoldValuedData.__init__(self, Sphere(sph_h), *args, **kwargs)

    def curve(self, t):
        t = 4/3*(1 - (1 - 0.5*t)**2) + 0.3
        return np.hstack((np.cos(3*t*np.pi)*np.sin(0.5*t*np.pi),
                          np.sin(3*t*np.pi)*np.sin(0.5*t*np.pi),
                          np.cos(0.5*t*np.pi)))

    def rho(self, z):
        return 0.5*self.mfd.dist(self.I[0][None], z[None])[0]**2
