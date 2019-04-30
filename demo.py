
import os, matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('agg')

if __name__ == "__main__":
    import sys
    from repyducible.demo import pkg_demo
    exp = pkg_demo("mflift", sys.argv[1:])
