
import logging
import numpy as np

from opymize import Variable
from opymize.functionals import SplitSum, ZeroFct, IndicatorFct, \
                                PositivityFct, L1Norms, AffineFct
from opymize.linear import BlockOp, IdentityOp, GradientOp, \
                           IndexedMultAdj, MatrixMultR, MatrixMultRBatched

from mflift.models import SublabelModel

class Model(SublabelModel):
    name = "tv"

    def __init__(self, *args, lbd=1.0, **kwargs):
        SublabelModel.__init__(self, *args, **kwargs)
        self.lbd = lbd
        logging.info("Init model '%s' (lambda=%.2e)" % (self.name, self.lbd))

        imagedims = self.data.imagedims
        N_image = self.data.N_image
        L_labels = self.data.L_labels
        M_tris = self.data.M_tris
        s_gamma = self.data.s_gamma
        d_image = self.data.d_image

        xvars = [('u', (N_image, L_labels)),
                 ('w12', (M_tris, N_image, s_gamma+1)),
                 ('w', (M_tris, N_image, d_image, s_gamma))]
        yvars = [('p', (N_image, d_image, L_labels)),
                 ('q', (N_image, L_labels)),
                 ('v12', (M_tris, N_image, s_gamma+1)),
                 ('v3', (N_image,)),
                 ('g', (M_tris, N_image, d_image, s_gamma))]

        if self.data.R.shape[-1] == s_gamma+1:
            del xvars[1]
            del yvars[2]

        self.x = Variable(*xvars)
        self.y = Variable(*yvars)

    def setup_solver(self, *args):
        imagedims = self.data.imagedims
        N_image = self.data.N_image
        L_labels = self.data.L_labels
        d_image = self.data.d_image
        self.linblocks.update({
            'Grad': GradientOp(imagedims, L_labels),
            'PB': IndexedMultAdj(L_labels, d_image*N_image, self.data.P, self.data.B),
            'Ad': MatrixMultRBatched(N_image*d_image, self.data.Ad),
        })
        SublabelModel.setup_solver(self, *args)

    def setup_dataterm_blocks(self):
        if hasattr(self, 'epifct') or hasattr(self, 'rho'):
            return
        N_image = self.data.N_image
        L_labels = self.data.L_labels
        M_tris = self.data.M_tris
        s_gamma = self.data.s_gamma
        P = self.data.P

        if self.data.R.shape[-1] == s_gamma+1:
            logging.info("Setup for model without sublabels...")
            R = self.data.R.reshape(M_tris, N_image, s_gamma+1)
            self.rho = np.zeros((N_image, L_labels), order='C')
            for j in range(M_tris):
                for m in range(s_gamma+1):
                    self.rho[:,P[j,m]] = R[j,:,m]
            self.rho = self.rho.ravel()
        else:
            logging.info("Setup for sublabel-accurate model...")
            self.epifct = EpigraphSupportFct(self.data.Rbase, self.data.Rfaces,
                                             self.data.Qbary, self.data.Sbary,
                                             self.data.R)

            # Ab (M_tris, s_gamma+1, s_gamma+1)
            Ab = np.zeros((M_tris, s_gamma+1, s_gamma+1),
                           dtype=np.float64, order='C')
            Ab[:] = np.eye(s_gamma+1)[None]
            Ab[...,-1] = -1
            self.linblocks['PAb'] = IndexedMultAdj(L_labels, N_image, P, Ab)

        self.linblocks.update({
            'S': MatrixMultR(N_image, np.ones((L_labels, 1), order='C')),
        })

    def setup_solver_pdhg(self):
        x, y = self.x.vars(named=True), self.y.vars(named=True)
        imagedims = self.data.imagedims
        N_image = self.data.N_image
        L_labels = self.data.L_labels
        M_tris = self.data.M_tris
        s_gamma = self.data.s_gamma
        d_image = self.data.d_image

        S_u_k = self.linblocks['S']
        GradOp = self.linblocks['Grad']
        PBLinOp = self.linblocks['PB']
        AdMult = self.linblocks['Ad']

        l1norms = L1Norms(M_tris*N_image, (d_image, s_gamma), self.lbd, "nuclear")

        Id_u = IdentityOp(x['u']['size'])

        if self.data.constraints is not None:
            constrmask, constru = self.data.constraints
            constru_lifted = self.data.mfd.embed_barycentric(constru)[1]
            Gu = ConstrainFct(constrmask, constru_lifted)
        else:
            Gu = PositivityFct(x['u']['size'])

        if hasattr(self, 'epifct'):
            PAbOp = self.linblocks['PAb']
            Id_w12 = IdentityOp(x['w12']['size'])

            G_summands = [
                Gu,                         # \delta_{u >= 0} or constraints
                ZeroFct(x['w12']['size']),  # 0
                ZeroFct(x['w']['size']),    # 0
            ]

            F_summands = [
                IndicatorFct(y['p']['size']),         # \delta_{p = 0}
                IndicatorFct(y['q']['size']),         # \delta_{q = 0}
                self.epifct,                          # \max_{v \in epi(rho*)} <v12,v>
                IndicatorFct(y['v3']['size'], c1=1), # \delta_{v3^i = 1}
                l1norms,                              # lbd*\sum_ji |g[j,i,:,:]|_nuc
            ]

            op_blocks = [
                [GradOp,       0,  PBLinOp], # p = Du - P'B'w
                [  Id_u,   PAbOp,        0], # q = u - P'Ab'w12
                [     0,  Id_w12,        0], # v12 = w12
                [ S_u_k,       0,        0], # v3^i = sum_k u[i,k]
                [     0,       0,   AdMult], # g = A'w
            ]
        else:
            G_summands = [
                Gu,                         # \delta_{u >= 0} or constraints
                ZeroFct(x['w']['size']),    # 0
            ]

            F_summands = [
                IndicatorFct(y['p']['size']),          # \delta_{p = 0}
                AffineFct(y['q']['size'], c=self.rho), # <q,rho>
                IndicatorFct(y['v3']['size'], c1=1),  # \delta_{v3^i = 1}
                l1norms,                               # lbd*\sum_ji |g[j,i,:,:]|_nuc
            ]

            op_blocks = [
                [GradOp, PBLinOp], # p = Du - P'B'w
                [  Id_u,       0], # q = u
                [ S_u_k,       0], # v3^i = sum_k u[i,k]
                [     0,  AdMult], # g = A'w
            ]

        self.pdhg_G = SplitSum(G_summands)
        self.pdhg_F = SplitSum(F_summands)
        self.pdhg_linop = BlockOp(op_blocks)
