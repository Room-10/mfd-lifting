
import numpy as np
from skimage.data import astronaut
from skimage.color import rgb2hsv

from mflift.data import ManifoldValuedData
from mflift.manifolds.circle import Circle
from mflift.tools.image import cell_centered_grid

class Data(ManifoldValuedData):
    name = "hue"
    imageh = (1.0,1.0)
    rhoDomain = np.array([[0.0,1.0],[0.0,1.0]])
    l_dimsubls = 6

    def __init__(self, *args, **kwargs):
        self.I = np.array(rgb2hsv(astronaut())[:80,:80,0], dtype=np.float64)
        self.I *= 2*np.pi # I comes with hue normalized to [0,1]
        self.imagedims = self.I.shape
        self.d_image = len(self.imagedims)
        self.N = self.N_image = np.prod(self.imagedims)
        self.rhoResolution = self.imagedims
        self.rhoGrid, h = cell_centered_grid(self.rhoDomain, self.rhoResolution)
        ManifoldValuedData.__init__(self, Circle(2*np.pi/5), *args, **kwargs)

    def rho(self, z):
        I = self.I.reshape(-1,1)
        return 0.5*self.mfd.dist(I[None], z[None])[0]**2
