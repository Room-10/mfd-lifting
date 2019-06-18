
Lifting of manifold-valued image processing problems with convex regularizers
=============================================================================

Solve manifold-valued image processing problems by convex relaxation.
Only convex first-order regularizing terms are supported.

If you use code from this repository within scientific work, please cite:

    [1] T. Vogt, E. Strekalovskiy, D. Cremers and J. Lellmann.
    Lifting methods for manifold-valued variational problems.
    In: P. Grohs, M. Holler and A. Weinmann (Eds.). Variational methods for
    nonlinear geometric data and applications. Springer Handbook, 2019.

Setup
-----

The recommended (and tested) setup is based on Ubuntu 18.04 with CUDA 8.0 or
newer. In that case, the following lines will do:

    $ git submodule update --init
    $ sudo apt install -y python3 python3-venv python3.6-dev llvm-dev g++
    $ python3 -m venv env
    $ source ./env/bin/activate
    (env) $ pip install --upgrade pip
    (env) $ pip install wheel cython numpy vtk
    (env) $ pip install -e .[cuda]

It is possible to install without `[cuda]` if no GPU is available, but CUDA
is recommended and enabled by default.
For execution on the CPU use the solver parameter `use_gpu=False`.

Basic Usage
-----------

Run the script `demo.py` without arguments to show available datasets and models:

    (env) $ python -m mflift

After that, specify the dataset and model you want to run:

    (env) $ python -m mflift moebius tv

More options are documented when using

    (env) $ python -m mflift moebius tv --help

All logs, plots, parameters, results etc. are automatically stored in a
subdirectory `results` in the current location.
Once computed, results can be plotted again by invoking

    (env) $ python -m mflift ./results/<name-of-result>

Reproducing the experiments in [1]
----------------------------------

The following options have been used to produce the results in the
chapter *Lifting methods for manifold-valued variational problems* (citation [1] above).

Riemannian center of mass on the circle (Fig. 1):

    (env) $ python -m mflift rcom tv

ROF denoising of a curve in R^2 (Fig. 3, left to right):

    (env) $ python -m mflift flat_1d tv --data-params "labels=(10,2)"
    (env) $ python -m mflift flat_1d tv --data-params "labels=(10,4)"
    (env) $ python -m mflift flat_1d tv --data-params "labels=(2,20)"
    (env) $ python -m mflift flat_1d rof

Total variation denoising of a curve on the sphere (Fig. 4, left to right):

    (env) $ python -m mflift sphere_1d tv --data-params "dimsubls=2,dimres=10"
    (env) $ python -m mflift sphere_1d tv --data-params "dimsubls=5,dimres=40"
    (env) $ python -m mflift sphere_1d tv --data-params "dimsubls=25,dimres=75"

Tikhonov denoising of a curve on the Klein bottle (Fig. 10):

    (env) $ python -m mflift klein_1d quadratic --model-params "lbd=50.0"

Tikhonov inpainting of a 2-d signal of rotations SO(3) (Fig. 11):

    (env) $ python -m mflift cam_2d_inpaint quadratic --data-params "mode='complete'"

Denoising of surface normals in DEM (Fig. 12, top to bottom):

    (env) $ python -m mflift bull tv --model-params "lbd=0.4"
    (env) $ python -m mflift bull huber --model-params "lbd=0.75,alph=0.1"
    (env) $ python -m mflift bull quadratic --model-params "lbd=3.0"

Denoising of high resolution cyclic InSAR measurements (Fig. 13):

    (env) $ python -m mflift insar tv --model-params "lbd=0.6"
    (env) $ python -m mflift insar huber --model-params "lbd=0.75,alph=0.1"
    (env) $ python -m mflift insar quadratic --model-params "lbd=1.0"

Troubleshooting
---------------

# Install PyQt5 for interactive surface plots (klein, moebius, sphere) on Ubuntu:

    $ apt install python3-pyqt5 python3-pyqt5.qtsvg
    $ ln -s /usr/lib/python3/dist-packages/PyQt5 ./env/lib/python3.6/site-packages/PyQt5

# Install VTK 7.x with Python 3.x bindings on Ubuntu 16.04:

    $ sudo add-apt-repository -y ppa:elvstone/vtk7
    $ sudo apt update
    $ sudo apt install -y vtk7
    $ echo "/opt/VTK-7.0.0/lib/python3.5/site-packages" > env/lib/python3.5/site-packages/vtk7.pth

Third party data
----------------

Included in this repository under `./data/`, you find third-party data sets
from the following sources:

    bull-nn.mat, bullhr-nn.mat, cm_mountain.mat
    Taken from https://lellmann.net/work/software/mfopt
    Originally due to D. Gesch et al. "The national map -- Elevation". In: US
    geological survey fact sheet 3053 (4). 2009.

    hcc.mat
    Taken from https://lellmann.net/work/software/mfopt

    piano_assembly.stl
    CC-BY-NC-SA kazysd: https://www.thingiverse.com/thing:148696

    Triceratops_plane_cut.stl
    CC-BY-NC-SA BillyOceansBlues: https://www.thingiverse.com/thing:3313805

    vesuflat.gif
    F. Rocca et al. "An overview of SAR interferometry". In: Proceedings of the
    3rd ERS Symposium on Space at the Service of our Environment, Florence (1997).
    http://earth.esa.int/workshops/ers97/program-details/speeches/rocca-et-al
