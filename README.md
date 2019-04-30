
Lifting of manifold-valued image processing problems with convex regularizers
=============================================================================

Solve manifold-valued image processing problems by convex relaxation.
Only convex first-order regularizing terms are supported.

Setup
-----

The recommended (and tested) setup is based on Ubuntu 16.04 with CUDA 8.0 or
newer. In that case, the following lines will do:

    sudo apt install -y python3 python3-venv python3.5-dev llvm-dev g++
    python3 -m venv env
    source ./env/bin/activate
    pip install --upgrade pip
    pip install wheel cython numpy
    pip install -r requirements.0.txt
    pip install -r requirements.1.txt
    git submodule update --init
    python setup.py build_ext --inplace

It is possible to run the code without CUDA using the solver parameter
"use_gpu=False", but CUDA is recommended and enabled by default.

Run
---

Run the script `demo.py` without arguments to show available datasets and models:

    python demo.py

After that, specify the dataset and model you want to test:

    python demo.py moebius tv

More options are documented when using

    python demo.py moebius tv --help

All logs, plots, parameters, results etc. are automatically stored in
subdirectories of the `results` directory.
