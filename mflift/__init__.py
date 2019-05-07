
import logging
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from repyducible.experiment import Experiment as BaseExperiment

from mflift.tools.plot import plot_polys

class Experiment(BaseExperiment):
    extra_source_files = ['demo.py','README.md']

    def plot(self, record=False):
        if self.pargs.plot == "no":
            return

        if record or self.pargs.plot == "hide":
            logging.info("Recording plots...")
            outputs = [(self.result, "plot-result.pdf")]
            if self.pargs.snapshots:
                outputs += [(s, "snapshot-%02d.png" % i)
                            for i,s in enumerate(self.snapshots)]

            fig = plt.gcf()

            for res,f in outputs:
                res_x =  self.model.x.vars(res['data'][0], True)

                u_proj = self.model.proj(res_x['u'])
                plot_polys(self.data.mfd.verts, self.data.mfd.simplices)
                plt.plot(u_proj[:,0], u_proj[:,1])
                c = self.data.curve(self.data.rhoGrid)
                plt.plot(c[:,0], c[:,1])

                canvas = FigureCanvasAgg(fig)
                canvas.print_figure(os.path.join(self.output_dir, f))
                plt.close(fig)
        else:
            logging.info("Plotting results interactively...")
            res_x =  self.model.x.vars(self.result['data'][0], True)

            u_proj = self.model.proj(res_x['u'])
            plot_polys(self.data.mfd.verts, self.data.mfd.simplices)
            plt.plot(u_proj[:,0], u_proj[:,1])
            c = self.data.curve(self.data.rhoGrid)
            plt.plot(c[:,0], c[:,1])

            plt.show()
            self.plot(record=True)
