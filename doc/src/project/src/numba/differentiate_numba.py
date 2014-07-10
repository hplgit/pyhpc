import numpy as np
from numbapro import *
import nose.tools as nt
import time

@jit('float64[::1](float64[::1], float64[::1], float64, float64, float64, int64)', nopython=True)
def differentiate(f_vec, df, dx, a, b, n):
    """
    Compute the discrete derivative of a Python function
    f on [a,b] using n intervals. Internal points apply
    a centered difference, while end points apply a one-sided
    difference. Vectorized version.
    """
    # End points
    df[0] = (f_vec[1] - f_vec[0])/dx
    df[-1] = (f_vec[-1] - f_vec[-2])/dx
    # Internal mesh points
    #df[1:-1] = (f_vec[2:] - f_vec[:-2])/(2*dx)
    for i in range(1, n):
        df[i] = (f_vec[i+1] - f_vec[i-1])/(2*dx)
    return df

def test_differentiate(a=1.2, b=2.4, n=3):
    f = lambda x: 4*x - 2.5
    x = np.linspace(a, b, n+1)
    f_vec = f(x)
    df = np.zeros_like(f_vec)
    dx = x[1] - x[0]
    start = time.time()
    df_vec = differentiate(f_vec, df, dx, a, b, n)
    end = time.time()
    print "Time: %g" %(end-start)
    df_exact = np.zeros_like(df_vec) + 4
    df_vec_diff = np.abs(df_vec - df_exact).max()
    nt.assert_almost_equal(df_vec_diff, 0, places=10)

if __name__ == '__main__':
    test_differentiate()
