#!/bin/sh
# Compile extension modules for the loop


# Cython (easier with pyximport)
module=wave2D_u0_loop_cy
rm -f $module.so
python setup_${module}.py build_ext --inplace   # compile
python -c "import $module"                      # test
if [ $? -eq 0 ]; then                           # success?
  echo "Cython module $module successfully built"
else
  echo "Building Cython module $module failed"
  exit 1
fi

# Cython (easier with pyximport)
module=wave2D_u0_par_loop_cy
rm -f $module.so
CC='gcc' python setup_${module}.py build_ext --inplace   # compile
python -c "import $module"                      # test
if [ $? -eq 0 ]; then                           # success?
  echo "Cython module $module successfully built"
else
  echo "Building Cython module $module failed"
  exit 1
fi

# Cython interface to C code
module=wave2D_u0_loop_c_cy
rm -f $module.so
CC='gcc' python setup_${module}.py build_ext --inplace   # compile
python -c "import $module"                      # test
if [ $? -eq 0 ]; then                           # success?
  echo "Cython module $module successfully built"
else
  echo "Building Cython module $module failed"
  exit 1
fi

