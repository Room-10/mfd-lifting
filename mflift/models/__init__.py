
import logging
import numpy as np

from repyducible.model import PDBaseModel

from opymize.functionals import EpigraphSupportFct
from opymize.linear import IndexedMultAdj, MatrixMultR

class SublabelModel(PDBaseModel):
    def __init__(self, *args):
        PDBaseModel.__init__(self, *args)
        self.linblocks = {}

    def setup_solver(self, *args):
        self.setup_dataterm_blocks()
        self.initialize()
        PDBaseModel.setup_solver(self, *args)

    def setup_dataterm_blocks(self):
        if hasattr(self, 'epifct'):
            return
        N_image = self.data.N_image
        L_labels = self.data.L_labels
        self.epifct = EpigraphSupportFct(self.data.Rbase, self.data.Rfaces,
                                         self.data.Q, self.data.S, self.data.R)
        self.linblocks.update({
            'PAb': IndexedMultAdj(L_labels, N_image, self.data.P, self.data.Ab_mats),
            'S': MatrixMultR(N_image, np.ones((L_labels, 1), order='C')),
        })

    def initialize(self):
        self.state = (self.x.new(), self.y.new())
        x = self.x.vars(self.state[0], named=True)
        if self.data.initializer is None:
            x['u'][:] = 1.0/self.data.L_labels
        else:
            s = self.data.initializer.shape[-1]
            uproj = np.zeros((self.data.N_image,s))
            if self.data.initializer.ndim == 1:
                uproj[:] = [self.data.initializer]
            else:
                uproj[:] = self.data.initializer
            utris, coords = self.data.mfd.embed(uproj)
            x['u'][:] = coords
            for i,tr in enumerate(utris):
                x['w12'][tr,i,:-1] = coords[i,self.data.P[tr,:-1]]
                x['w12'][tr,i,-1] = -1.0

    def proj(self, u):
        tmp_u = u.copy()
        np.clip(tmp_u, 0.0, 1.0, out=tmp_u)
        tmp_u /= tmp_u.sum(axis=1)[:,None]
        return self.data.mfd.mean(self.data.T, weights=tmp_u)
