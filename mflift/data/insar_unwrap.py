
import numpy as np
from skimage.io import imread

from mflift.data import ManifoldValuedData
from mflift.manifolds.flat import Interval
from mflift.tools.image import cell_centered_grid

class Data(ManifoldValuedData):
    name = "insar_unwrap"
    imageh = (1.0,1.0)
    rhoDomain = np.array([[0.0,1.0],[0.0,1.0]])
    l_dimsubls = 6

    def __init__(self, *args, **kwargs):
        self.I = np.float64(imread("data/vesuflat.gif"))/255.0
        self.I *= 2*np.pi # I comes with hue normalized to [0,1]
        self.imagedims = self.I.shape
        self.d_image = len(self.imagedims)
        self.N = self.N_image = np.prod(self.imagedims)
        self.rhoResolution = self.imagedims
        self.rhoGrid, h = cell_centered_grid(self.rhoDomain, self.rhoResolution)
        ManifoldValuedData.__init__(self, Interval(0,200,50), *args, **kwargs)

    def rho(self, z):
        I = self.I.reshape(-1)
        diff = I[None] - z%(2*np.pi)
        return 0.5*np.fmin(np.abs(diff), np.abs(2*np.pi - diff))**2
