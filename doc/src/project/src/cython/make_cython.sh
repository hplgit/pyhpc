#!/bin/sh

# Cython (easier with pyximport)
module=wave2D_u0_loop_cy
rm -f $module.so
python setup.py build_ext --inplace   # compile
python -c "import $module"                      # test
if [ $? -eq 0 ]; then                           # success?
  echo "Cython module $module successfully built"
else
  echo "Building Cython module $module failed"
  exit 1
fi
