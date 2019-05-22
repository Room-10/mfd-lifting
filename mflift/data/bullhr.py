
import numpy as np
from scipy.io import loadmat

import mflift.data.bull
from mflift.manifolds.sphere import Sphere
from mflift.tools.image import cell_centered_grid

class Data(mflift.data.bull.Data):
    name = "bullhr"
    filename = "data/bullhr-nn.mat"
