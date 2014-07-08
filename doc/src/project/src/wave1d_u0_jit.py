from numpy import *
from numbapro import *
import time

def solver(I, V, f, c, L, Nx, C, T):
    x = linspace(0, L, Nx+1) # Mesh points in space
    dx = x[1] - x[0]
    dt = C*dx/c
    Nt = int(round(T/dt))
    t = linspace(0, Nt*dt, Nt+1) # Mesh points in time
    if f is None or f == 0:
        f = (lambda x, t: 0)
    if V is None or V == 0:
        V = (lambda x: 0)

    V_vec = V(x)
    I_vec = I(x)
    f_vec = f(x,t)

    u = zeros(Nx+1) # Solution array at new time level
    u_1 = zeros(Nx+1) # Solution at 1 time level back
    u_2 = zeros(Nx+1) # Solution at 2 time levels back
    C2 = C**2 # Help variable in the scheme

    #Special formula for first time step
    u = advance(u, u_1, u_2, f_vec, I_vec, V_vec, Nx, Nt, C2, dt, True)
    u = advance(u, u_1, u_2, f_vec, I_vec, V_vec, Nx, Nt, C2, dt, False)

    return u, x, t


@jit('float64[::1](float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], int64, int64, float64, float64, int64)', nopython=True)
def advance(u, u_1, u_2, f, I, V, Nx, Nt, C2, dt, step1):
    if step1:
        #Load initial cond into u_1
        for i in range(0,Nx+1):
            u_1[i] = I[i]

        n = 0
        for i in range(1, Nx):
            u[i] = u_1[i] + dt*V[i] + \
                0.5*C2*(u_1[i-1] - 2*u_1[i] + u_1[i+1]) + \
                0.5*dt**2*f[n]
            u[0] = 0; u[Nx] = 0

        #Switch variables before next step
        for i in range(Nx+1):
            u_2[i] = u_1[i]
            u_1[i] = u[i]
    else:
        for n in range(1, Nt):
            # Update all inner points at time t[n+1]
            for i in range(1, Nx):
                u[i] = - u_2[i] + 2*u_1[i] + \
                    C2*(u_1[i-1] - 2*u_1[i] + u_1[i+1]) + \
                    dt**2*f[n]

            #Insert boundary conditions
            u[0] = 0; u[Nx] = 0

            # Switch variables before next step
            for i in range(Nx+1):
                u_2[i] = u_1[i]
                u_1[i] = u[i]
    return u

import nose.tools as nt

def test_quadratic(Nx=100):
    """Check that u(x,t)=x(L-x)(1+t/2) is exactly reproduced."""
    def exact_solution(x, t):
        return x*(L-x)*(1 + 0.5*t)

    def I(x):
        return exact_solution(x, 0)

    def V(x):
        return 0.5*exact_solution(x, 0)

    def f(x, t):
        return 2*(1 + 0.5*t)*c**2

    L = 2.5
    c = 1.5
    C = 0.75
    T = 18
    
    start = time.time()
    u, x, t = solver(I, V, f, c, L, Nx, C, T)
    end = time.time()
    print "Time: %g s" %(end-start)
    u_e = exact_solution(x, t[-1])
    diff = abs(u - u_e).max()
    nt.assert_almost_equal(diff, 0, places=10)

if __name__ == '__main__':
    test_quadratic()
