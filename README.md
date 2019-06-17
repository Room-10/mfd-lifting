
Lifting of manifold-valued image processing problems with convex regularizers
=============================================================================

Solve manifold-valued image processing problems by convex relaxation.
Only convex first-order regularizing terms are supported.

Setup
-----

The recommended (and tested) setup is based on Ubuntu 18.04 with CUDA 8.0 or
newer. In that case, the following lines will do:

    sudo apt install -y python3 python3-venv python3.5-dev llvm-dev g++
    python3 -m venv env
    source ./env/bin/activate
    pip install --upgrade pip
    pip install wheel cython numpy
    git submodule update --init
    pip install -e .[cuda]

It is possible to install without `[cuda]` if no GPU is available, but CUDA
is recommended and enabled by default.
For execution on the CPU use the solver parameter `use_gpu=False`.

Run
---

Run the script `demo.py` without arguments to show available datasets and models:

    python demo.py

After that, specify the dataset and model you want to run:

    python demo.py moebius tv

More options are documented when using

    python demo.py moebius tv --help

All logs, plots, parameters, results etc. are automatically stored in
subdirectories of the `results` directory.

Troubleshooting
---------------

# Install VTK 7.x with Python 3.x bindings on Ubuntu 16.04:

    sudo add-apt-repository -y ppa:elvstone/vtk7
    sudo apt update
    sudo apt install -y vtk7
    echo "/opt/VTK-7.0.0/lib/python3.5/site-packages" > env/lib/python3.5/site-packages/vtk7.pth
