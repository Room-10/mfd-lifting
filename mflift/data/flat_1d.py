
import numpy as np

from mflift.data import ManifoldValuedData
from mflift.manifolds.flat import Square
from mflift.tools.image import cell_centered_grid

class Data(ManifoldValuedData):
    name = "flat-1d"
    d_image = 1
    imagedims = (50,)
    imageh = (1.0,)
    rhoDomain = np.array([[0.0,1.0]])

    def __init__(self, *args, labels=(2, 20), **kwargs):
        assert len(labels) == 2
        dimls, self.l_dimsubls = labels

        self.N = self.N_image = np.prod(self.imagedims)
        self.rhoResolution = (self.N_image,)
        self.rhoGrid, h = cell_centered_grid(self.rhoDomain, self.rhoResolution)
        #self.initializer = self.curve(self.rhoGrid)
        ManifoldValuedData.__init__(self, Square(2.1, dimls), *args, **kwargs)
        self.data_b = self.curve(self.rhoGrid)

    def curve(self, t):
        # Spiral:
        return (0.8 - 0.5*t)*np.hstack((np.cos(4*t*np.pi), np.sin(4*t*np.pi)))
        # Straight line:
        #return np.hstack((-0.4 + 0.7*t, -0.9 + 1.5*t))

    def rho_x(self, x, z):
        return 0.5*self.mfd.dist(self.curve(x)[None], z[None])[0]**2
