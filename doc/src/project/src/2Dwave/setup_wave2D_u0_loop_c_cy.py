from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

sources = ['wave2D_u0_loop_c.c', 'wave2D_u0_loop_c_cy.pyx']
module = 'wave2D_u0_loop_c_cy'
setup(
  name=module,
  ext_modules=[Extension(module, sources,
                         libraries=['gomp'], # C libs to link with
                         extra_compile_args=['-fopenmp'],
                         include_dirs=[np.get_include()]
                         )],
  cmdclass={'build_ext': build_ext},
)
