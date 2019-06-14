
import numpy as np

from mflift.data import ManifoldValuedData
from mflift.manifolds.so3 import SO3
from mflift.tools.image import cell_centered_grid
from mflift.tools.linalg import quaternion_prod, normalize

class Data(ManifoldValuedData):
    name = "cam-1d-inpaint"
    d_image = 1
    imagedims = (13,)
    imageh = (1.0,)
    rhoDomain = np.array([[0.0,1.0]])

    def __init__(self, *args, dimsubls=6, dimres=50, **kwargs):
        self.l_dimsubls = dimsubls
        so3_h = 2*np.pi/np.ceil(dimres/(dimsubls - 1))

        self.N = self.N_image = np.prod(self.imagedims)
        self.rhoResolution = (self.N_image,)
        self.rhoGrid, h = cell_centered_grid(self.rhoDomain, self.rhoResolution)
        q1 = np.array([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0])
        q2 = np.array([np.cos(np.pi/2), np.sin(np.pi/2), 0, 0])
        qbase = quaternion_prod(q2, q1)
        hh = np.linspace(0, 2*np.pi, self.N_image)
        q3 = np.vstack((np.cos(hh/2), np.sin(hh/2), -np.sin(hh/2), 0*hh)).T
        q = quaternion_prod(q3[None], qbase[None,None])[0,:,0]
        self.I = [normalize(q)]
        self.inpaint_msk = np.ones(self.imagedims, dtype=bool)
        self.inpaint_msk[1:6] = False
        self.inpaint_msk[7:-1] = False
        self.I[0][np.logical_not(self.inpaint_msk)] = [[1,0,0,0]]
        self.constraints = [self.inpaint_msk, self.I[0]]
        ManifoldValuedData.__init__(self, SO3(so3_h), *args, **kwargs)

    def rho(self, z):
        dt = 0.5*self.mfd.dist(self.I[0][None], z[None])[0]**2
        dt[self.inpaint_msk,:] = 0
        return dt
