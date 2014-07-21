from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

cymodule = 'wave2D_u0_par_loop_cy'
setup(
name=cymodule,
    ext_modules=[Extension(cymodule, [cymodule + '.pyx'],include_dirs=[np.get_include()],extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])],
cmdclass={'build_ext': build_ext},
)
