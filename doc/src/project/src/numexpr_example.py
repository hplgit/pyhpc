import numpy as np
import numexpr as ne
import time
timer_function = time.clock

def sum(M, N):
    x = np.zeros(N)
    expr = ' + '.join(['%d*x' % i for i in range(1, M+1)])
    t0 = timer_function()
    eval(expr)
    np_time = timer_function() - t0

    t0 = timer_function()
    ne.evaluate(expr)
    ne_time = timer_function() - t0

    return ne_time, np_time

if __name__ == '__main__':
    for M in 2, 5, 10, 30:
        for N in 1000, 100000, 1000000:
            ne_time, np_time = sum(M, N)
            print "N=%8d, M=%d numexpr/numpy: %.4f" % (N, M, ne_time/np_time)
