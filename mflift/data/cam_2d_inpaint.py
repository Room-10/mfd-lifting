
import numpy as np

from mflift.data import ManifoldValuedData
from mflift.manifolds.so3 import SO3
from mflift.tools.image import cell_centered_grid
from mflift.tools.linalg import quaternion_prod, normalize

class Data(ManifoldValuedData):
    name = "cam-2d-inpaint"
    d_image = 2
    imagedims = (5,5)
    imageh = (1.0,1.0)
    rhoDomain = np.array([[0.0,1.0],[0.0,1.0]])

    def __init__(self, *args, dimsubls=6, dimres=50, **kwargs):
        self.l_dimsubls = dimsubls
        so3_h = 2*np.pi/np.ceil(dimres/(dimsubls - 1))

        self.N = self.N_image = np.prod(self.imagedims)
        self.rhoResolution = self.imagedims
        self.rhoGrid, h = cell_centered_grid(self.rhoDomain, self.rhoResolution)

        q1 = np.array([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0])
        q2 = np.array([np.cos(np.pi/2), np.sin(np.pi/2), 0, 0])
        qbase = quaternion_prod(q2, q1)
        hh = np.linspace(0, 2*np.pi, 20)
        q3 = np.vstack((np.cos(hh/2), np.sin(hh/2), -np.sin(hh/2), 0*hh)).T
        q = normalize(quaternion_prod(q3[None], qbase[None,None])[0,:,0])

        self.I = np.zeros(self.imagedims + (4,), dtype=np.float64, order='C')
        self.I[ 0, 0] = q[0]
        self.I[-1, 0] = q[6]
        self.I[ 0,-1] = q[12]
        self.I[-1,-1] = q[18]
        #self.I[ 0] = q[:5]
        #self.I[-1] = q[15:20]
        self.inpaint_msk = np.ones(self.imagedims, dtype=bool)
        self.inpaint_msk[[0,0,-1,-1],[0,-1,0,-1]] = False
        #self.inpaint_msk[[0,-1],:] = False
        self.constr_msk = np.zeros(self.imagedims, dtype=bool)
        self.constr_msk[[0,0,-1,-1],[0,-1,0,-1]] = True
        #self.constr_msk[[0,-1],:] = True

        self.inpaint_msk = self.inpaint_msk.reshape(-1)
        self.constr_msk = self.constr_msk.reshape(-1)
        self.I = self.I.reshape(-1,4)
        self.I[self.inpaint_msk] = [[1,0,0,0]]
        self.constraints = [self.constr_msk, self.I]
        ManifoldValuedData.__init__(self, SO3(so3_h), *args, **kwargs)

    def rho(self, z):
        dt = 0.5*self.mfd.dist(self.I[None], z[None])[0]**2
        dt[self.inpaint_msk,:] = 0
        return dt
