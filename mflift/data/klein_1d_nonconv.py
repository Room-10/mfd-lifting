
import numpy as np

from mflift.data import ManifoldValuedData
from mflift.manifolds.klein import KleinBottle, klein_normalize
from mflift.tools.image import cell_centered_grid

class Data(ManifoldValuedData):
    name = "klein-1d-non-convex"
    d_image = 1
    imagedims = (50,)
    imageh = (1.0,)
    rhoDomain = np.array([[0.0,1.0]])

    def __init__(self, *args, dimsubls=4, dimres=45, **kwargs):
        self.l_dimsubls = dimsubls
        phi_h = 2*np.pi/np.ceil(dimres/(dimsubls - 1))

        self.N = self.N_image = np.prod(self.imagedims)
        self.rhoResolution = (self.N_image,)
        self.rhoGrid, h = cell_centered_grid(self.rhoDomain, self.rhoResolution)
        self.I = [self.curve(self.rhoGrid)]
        ManifoldValuedData.__init__(self, KleinBottle(phi_h), *args, **kwargs)

    def curve(self, t):
        vals = np.hstack(((-1.4 + 0.4*t)*np.pi,
                          (-1.15 + 0.3*t)*np.pi,))
        return klein_normalize(vals)

    def rho(self, z):
        return -0.5*self.mfd.dist(self.I[0][None], z[None])[0]**2
