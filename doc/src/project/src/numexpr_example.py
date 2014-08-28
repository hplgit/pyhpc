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
    for N in [10000,100000,1000000, 10000000]:
        ne_time, np_time = sum(N)
        print "N: ", N
        print "Numexpr: ", ne_time
        print "Numpy:   ", np_time
        print "Speed-up:", (np_time/float(ne_time))
        print ""

