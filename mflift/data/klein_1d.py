
import numpy as np

from mflift.data import ManifoldValuedData
from mflift.manifolds.klein import KleinBottle, klein_normalize
from mflift.tools.image import cell_centered_grid

class Data(ManifoldValuedData):
    name = "klein-1d"
    d_image = 1
    imagedims = (50,)
    imageh = (1.0,)
    rhoDomain = np.array([[0.0,1.0]])

    def __init__(self, *args, dimsubls=10, dimres=45, **kwargs):
        self.l_dimsubls = dimsubls
        phi_h = 2*np.pi/np.ceil(dimres/(dimsubls - 1))

        self.N = self.N_image = np.prod(self.imagedims)
        self.rhoResolution = (self.N_image,)
        self.rhoGrid, h = cell_centered_grid(self.rhoDomain, self.rhoResolution)
        ManifoldValuedData.__init__(self, KleinBottle(phi_h), *args, **kwargs)

    def curve(self, t):
        vals = np.hstack(((-1.3 + 1.2*t)*np.pi,
                          (-0.3 + 0.5*t),))
        noise = 0.05*np.random.normal(size=vals.shape)
        return klein_normalize(vals + noise)

    def rho_x(self, x, z):
        return 0.5*self.mfd.dist(self.curve(x)[None], z[None])[0]**2
