
import numpy as np

from mflift.data import ManifoldValuedData
from mflift.manifolds.moebius import Moebius, moeb_normalize
from mflift.tools.image import cell_centered_grid

class Data(ManifoldValuedData):
    name = "moebius-1d"
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
        self.I = [self.curve(self.rhoGrid) for k in range(5)]
        ManifoldValuedData.__init__(self, Moebius(phi_h), *args, **kwargs)

    def curve(self, t):
        vals = np.hstack(((-1.1 + 0.7*t)*np.pi, (-0.1 + 0.2*t),))
        noise = 0.2*np.random.normal(size=vals.shape)
        return moeb_normalize(vals + noise)

    def rho(self, z):
        return -sum([0.5*self.mfd.dist(I[None], z[None])[0]**2 for I in self.I])
