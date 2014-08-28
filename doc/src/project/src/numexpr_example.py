import numpy as np
import numexpr as ne
import timeit

def sum(N):
    x = np.zeros(N)
    expr = 'x + 2*x + 3*x + 4*x + 5*x + 6*x + 7*x + 8*x + 9*x'
    t0 = timeit.default_timer()
    eval(expr)
    t1 = timeit.default_timer()
    np_time = t1 - t0

    t0 = timeit.default_timer()
    ne.evaluate(expr)
    t1 = timeit.default_timer()
    ne_time = t1 - t0
    
    return ne_time, np_time

if __name__ == '__main__':
    for N in [10000,100000,1000000, 10000000]:
        ne_time, np_time = sum(N)
        print "N: ", N
        print "Numexpr: ", ne_time
        print "Numpy:   ", np_time
        print "Speed-up:", (np_time/float(ne_time))
        print ""
