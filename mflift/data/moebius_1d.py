
import numpy as np

from mflift.data import ManifoldValuedData
from mflift.manifolds.moebius import Moebius, moeb_normalize
from mflift.tools.image import cell_centered_grid

class Data(ManifoldValuedData):
    name = "moebius-1d"
    d_image = 1
    imagedims = (50,)
    imageh = (1.0,)
    l_dimsubls = 10
    rhoDomain = np.array([[0.0,1.0]])

    def __init__(self, *args, **kwargs):
        self.N = self.N_image = np.prod(self.imagedims)
        self.rhoResolution = (self.N_image,)
        self.rhoGrid, h = cell_centered_grid(self.rhoDomain, self.rhoResolution)
        ManifoldValuedData.__init__(self, Moebius(0.5, 2*np.pi/5), *args, **kwargs)

    def curve(self, t):
        vals = np.hstack(((-1.3 + 1.2*t)*np.pi,
                          (-0.3 + 0.5*t),))
        noise = 0.05*np.random.normal(size=vals.shape)
        return moeb_normalize(vals + noise)

    def rho_x(self, x, z):
        return 0.5*self.mfd.dist(self.curve(x)[None], z[None])[0]**2
