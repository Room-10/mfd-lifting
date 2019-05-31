
import logging
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from repyducible.experiment import Experiment as BaseExperiment

from mflift.tools.plot import plot_curves, plot_terrain_maps, plot_hue_images

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
        if self.pargs.plot == "no":
            return

        subgrid = self.data.S.reshape(-1,self.data.S.shape[-1])
        if True or self.model.name == "rof":
            subgrid = None

        outputs = []
        if record:
            logging.info("Recording plots...")
            outputs = [(self.result, "plot-result.pdf")]
            if self.pargs.snapshots:
                outputs += [(s, "snapshot-%02d.png" % i)
                            for i,s in enumerate(self.snapshots)]
        elif self.pargs.plot != "hide":
            logging.info("Plotting results interactively...")
            outputs = [(self.result, None)]

        for res,f in outputs:
            f = f if f is None else os.path.join(self.output_dir, f)
            res_x =  self.model.x.vars(res['data'][0], True)
            u_proj = self.model.proj(res_x['u'])
            if self.data.d_image == 1:
                crv = self.data.curve(self.data.rhoGrid)
                plot_curves([crv, u_proj], self.data.mfd,
                    subgrid=subgrid, filename=f)
            elif self.data.d_image == 2 and self.data.name[:4] == "bull":
                I = self.data.I.reshape(self.data.imagedims + (3,))
                Iu = u_proj.reshape(self.data.imagedims + (3,))
                plot_terrain_maps([I,Iu], self.data.extra, filename=f)
            elif self.data.d_image == 2 and self.data.name == "hue":
                I = self.data.I.reshape(self.data.imagedims)
                Iu = u_proj.reshape(self.data.imagedims)
                plot_hue_images([I,Iu], filename=f)

        if not record:
            self.plot(record=True)
