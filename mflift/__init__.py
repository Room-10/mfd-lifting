
import logging
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from repyducible.experiment import Experiment as BaseExperiment

from mflift.tools.plot import plot_curves

class Experiment(BaseExperiment):
    extra_source_files = ['demo.py','README.md']

    def init_params(self, *args):
        BaseExperiment.init_params(self, *args)
        if self.pargs.solver == "pdhg":
            self.params['solver'].update({
                'granularity': int(1e4),
                'term_maxiter': int(5e5),
                'steps': 'adaptive',
            })

    def plot(self, record=False):
        if self.pargs.plot != "show":
            return

        logging.info("Plotting results interactively...")
        res_x =  self.model.x.vars(self.result['data'][0], True)
        u_proj = self.model.proj(res_x['u'])
        subgrid = self.data.S.reshape(-1,self.data.S.shape[-1])
        if self.model.name == "rof":
            subgrid = None

        if self.data.d_image == 1:
            crv = self.data.curve(self.data.rhoGrid)
            plot_curves([crv, u_proj], self.data.mfd, subgrid=subgrid)
