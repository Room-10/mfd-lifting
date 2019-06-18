
import logging
import numpy as np

from repyducible.data import Data as BaseData
from opymize.quickhull import piecewise_convexify

from mflift.tools.linalg import barygrid

class ManifoldValuedData(BaseData):
    rho = None
    N = 0
    initializer = None
    constraints = None

    def __init__(self, mfd, *args, **kwargs):
        BaseData.__init__(self, *args, **kwargs)

        self.mfd = mfd
        self.s_gamma = self.mfd.ndim

        # T (L_labels, nintdim)
        # P (M_tris, s_gamma+1)
        self.T, self.P = self.mfd.verts, self.mfd.simplices
        self.L_labels = self.mfd.nverts
        self.M_tris = self.mfd.nsimplices

        # B (M_tris, s_gamma, s_gamma+1)
        # Ad (M_tris, s_gamma, s_gamma)
        self.B, self.Ad = self.mfd.mean_derivative_matrices

        # Qbary (1, m_sublabels)
        # Sbary (m_sublabels, s_gamma)
        # S (M_tris, m_sublabels, nintdim)
        self.init_subgrid()
        self.m_sublabels = self.Sbary.shape[0]
        self.l_sublabels = self.S.shape[0]*self.S.shape[1]

        logging.info("Label space: %d labels, %d tris, %d sublabels (%d per tri)"
            % (self.L_labels, self.M_tris, self.l_sublabels, self.m_sublabels))

        # R (M_tris*N, m_sublabels)
        # Rbase (M_tris*N, m_sublabels)
        npoints = self.N*self.l_sublabels
        nhulls = self.N*self.M_tris
        M, N, m = self.M_tris, self.N_image, self.m_sublabels
        logging.info("Evaluating function on %d points..." % npoints)
        result = self.rho(self.S.reshape(-1, self.mfd.nintdim))
        self.R = np.zeros((M*N,m), dtype=np.float64, order='C')
        self.R[:] = result.reshape((N,M,m)).transpose(1,0,2).reshape(-1,m)
        logging.info("Computing %d convex hulls..." % nhulls)
        self.Rbase, self.Rfaces = piecewise_convexify(self.Sbary, self.R, self.Qbary)

    def init_subgrid(self):
        if self.l_dimsubls == 2:
            bary = np.eye(self.s_gamma+1)
            subgrid = self.T[self.P]
        else:
            bary = barygrid(self.s_gamma, self.l_dimsubls, boundary=True)
            bary = bary[np.all(bary != 1, axis=1)]
            bary = np.concatenate((np.eye(self.s_gamma+1), bary), axis=0)
            subgrid = self.mfd.mean(self.T[self.P][None], bary[None])[0]
        self.Sbary = np.array(bary[:,:-1], dtype=np.float64, order='C')
        self.Qbary = np.arange(bary.shape[0])[None,:]
        self.Qbary = np.array(self.Qbary, dtype=np.int64, order='C')
        self.S = np.array(subgrid, dtype=np.float64, order='C')

    def rho(self, z):
        return self.rho_x(self.rhoGrid, z)
