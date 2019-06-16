
import logging
import numpy as np

from opymize import Variable
from opymize.functionals import SplitSum, ZeroFct, IndicatorFct, \
                                PositivityFct, ConstrainFct, \
                                QuadEpiSupp, HuberPerspective
from opymize.linear import BlockOp, IdentityOp, GradientOp, \
                           IndexedMultAdj, MatrixMultR, MatrixMultRBatched

from mflift.models import SublabelModel

class Model(SublabelModel):
    name = "quadratic"

    def __init__(self, *args, lbd=5.0, alph=np.inf, fdscheme="centered", **kwargs):
        SublabelModel.__init__(self, *args, **kwargs)
        self.lbd = lbd
        self.alph = alph
        self.fdscheme = fdscheme
        logging.info("Init model '%s' (lambda=%.2e, alpha=%.2e, fdscheme=%s)" \
                     % (self.name, self.lbd, self.alph, self.fdscheme))

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
                 ('g12', (M_tris, N_image, d_image*s_gamma+1)),]

        self.x = Variable(*xvars)
        self.y = Variable(*yvars)

    def setup_solver(self, *args):
        imagedims = self.data.imagedims
        N_image = self.data.N_image
        L_labels = self.data.L_labels
        d_image = self.data.d_image
        M_tris = self.data.M_tris
        s_gamma = self.data.s_gamma

        Id_w2 = np.zeros((s_gamma+1,d_image*s_gamma+1), order='C')
        Id_w2[-1,-1] = 1.0

        Adext = np.zeros((M_tris,d_image*s_gamma,d_image*s_gamma+1), order='C')
        Adext[:,:,:-1] = np.kron(np.eye(d_image), self.data.Ad)

        self.linblocks.update({
            'Grad': GradientOp(imagedims, L_labels, scheme=self.fdscheme),
            'PB': IndexedMultAdj(L_labels, d_image*N_image, self.data.P, self.data.B),
            'Adext': MatrixMultRBatched(N_image, Adext),
            'Id_w2': MatrixMultR(M_tris*N_image, Id_w2),
        })
        SublabelModel.setup_solver(self, *args)

    def setup_solver_pdhg(self):
        x, y = self.x.vars(named=True), self.y.vars(named=True)
        imagedims = self.data.imagedims
        N_image = self.data.N_image
        L_labels = self.data.L_labels
        M_tris = self.data.M_tris
        s_gamma = self.data.s_gamma
        d_image = self.data.d_image

        PAbOp = self.linblocks['PAb']
        S_u_k = self.linblocks['S']
        GradOp = self.linblocks['Grad']
        PBLinOp = self.linblocks['PB']
        AdMult = self.linblocks['Adext']
        Id_w2 = self.linblocks['Id_w2']

        if self.alph < np.inf:
            etahat = HuberPerspective(M_tris*N_image, s_gamma*d_image,
                                      lbd=self.lbd, alph=self.alph)
        else:
            etahat = QuadEpiSupp(M_tris*N_image, s_gamma*d_image, a=self.lbd)

        Id_u = IdentityOp(x['u']['size'])
        Id_w12 = IdentityOp(x['w12']['size'])

        if self.data.constraints is not None:
            constrmask, constru = self.data.constraints
            constru_lifted = self.data.mfd.embed_barycentric(constru)[1]
            Gu = ConstrainFct(constrmask, constru_lifted)
        else:
            Gu = PositivityFct(x['u']['size'])

        self.pdhg_G = SplitSum([
            Gu,                         # \delta_{u >= 0} or constraints
            ZeroFct(x['w12']['size']),  # 0
            ZeroFct(x['w']['size']),    # 0
        ])

        self.pdhg_F = SplitSum([
            IndicatorFct(y['p']['size']),        # \delta_{p = 0}
            IndicatorFct(y['q']['size']),        # \delta_{q = 0}
            self.epifct,                         # \max_{v \in epi(rho*)} <v12,v>
            IndicatorFct(y['v3']['size'], c1=1), # \delta_{v3^i = 1}
            etahat,                              # 0.5*lbd*\sum_ji |g1[j,i]|^2/|g2[j,i]|
        ])

        self.pdhg_linop = BlockOp([
            [GradOp,       0,  PBLinOp], # p = Du - P'B'w
            [  Id_u,   PAbOp,        0], # q = u - P'Ab'w12
            [     0,  Id_w12,        0], # v12 = w12
            [ S_u_k,       0,        0], # v3^i = sum_k u[i,k]
            [     0,   Id_w2,   AdMult], # g12 = (Ad'w, w2)
        ])
