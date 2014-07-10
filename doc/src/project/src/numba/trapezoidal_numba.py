import numpy as np
from numbapro import *
import time

@jit('float64(float64[::1], float64, float64, int64)', nopython=True)
def trapezoidal(f_vec, a, b, n):
    f_vec[0] /= 2.0
    f_vec[-1] /= 2.0
    h = (b-a)/float(n)
    I = 0
    for i in range(n+1):
        I += f_vec[i]
    I = h*I
    return I

import nose.tools as nt

def test_trapezoidal(a=1.2, b=2.4, n=3):
    """Test that linear functions are exactly integrated."""
    f = lambda x: 4*x - 2.5
    import sympy as sm
    x = sm.Symbol('x')
    I_exact = sm.integrate(4*x - 2.5, (x, a, b))
    x = np.linspace(a, b, n+1)
    f_vec = f(x)
    start = time.time()
    I_num = trapezoidal(f_vec, a, b, n)
    end = time.time()
    print "Time: %g s" %(end-start)
    nt.assert_almost_equal(I_num, I_exact, places=10)

if __name__ == '__main__':
    test_trapezoidal()
