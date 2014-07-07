import numpy as np
from numbapro import *
import time 

def solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T, user_action=None):
    x = np.linspace(0, Lx, Nx+1) #mesh points in x dir
    y = np.linspace(0, Ly, Ny+1) #mesh points in y dir
    xv = x[:,np.newaxis] # for vectorized function evaluations
    yv = y[np.newaxis,:]
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    stability_limit = (1/float(c))*(1/np.sqrt(1/dx**2 + 1/dy**2))
    if dt <= 0: # max time step?
        safety_factor = -dt # use negative dt as safety factor
        dt = safety_factor*stability_limit
    elif dt > stability_limit:
        print 'error: dt=%g exceeds the stability limit %g' % \
              (dt, stability_limit)
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, Nt*dt, Nt+1) #mesh points in time
    Cx2 = (c*dt/dx)**2; Cy2 = (c*dt/dy)**2

    # Allow f and V to be None or 0
    if f is None or f == 0:
        f = (lambda x, y, t: 0)
        # or simpler: x*y*0
    if V is None or V == 0:
        V = (lambda x, y: 0)

    u = np.zeros((Nx+1, Ny+1)) #solution array
    u_1 = np.zeros((Nx+1, Ny+1)) #solution at t-dt
    u_2 = np.zeros((Nx+1, Ny+1)) #solution at t-2*dt
    f_vec = f(xv, yv, t[0]) #precompute, size as u
    V_vec = V(xv, yv)
    I_vec = I(xv, yv)
    It = range(0, t.shape[0])
    
    t0 = time.time()
    #Special formula for first time-step
    u_1 = advance(u, u_1, u_2, f_vec, V_vec, I_vec, Cx2, Cy2, Nx, Ny, 0, dt, True, False)
    if user_action is not None:
        user_action(u_1, x, xv, y, yv, t, 0)

    u = advance(u, u_1, u_2, f_vec, V_vec, I_vec, Cx2, Cy2, Nx, Ny, 0, dt, False, True)
    if user_action is not None:
        user_action(u, x, xv, y, yv, t, 1)

    for n in It[1:-1]:
        f_vec[:,:] = f(xv,yv,t[n])
        u = advance(u, u_1, u_2, f_vec, V_vec, I_vec, Cx2, Cy2, Nx, Ny, n, dt, False, False)
        if user_action is not None:
            if user_action(u, x, xv, y, yv, t, n+1):
                break
    t1 = time.time()
    return dt, (t1-t0)

@jit('float64[:,::1](float64[:,::1],float64[:,::1],float64[:,::1],float64[:,::1],float64[:,::1],float64[:,::1], float64, float64, int64, int64, int64, float64, int64, int64)', nopython=True)
def advance(u, u_1, u_2, f, V, I, Cx2, Cy2, Nx, Ny, n, dt, step0, step1):
    Ix = range(0, u.shape[0])
    Iy = range(0, u.shape[1])
    dt2 = dt**2
    if step0:
        #Load initial cond into u_1
        for i in Ix:
            for j in Iy:
                u_1[i,j] = I[i,j]
        return u_1

    if step1:
        Cx2 = 0.5*Cx2; Cy2 = 0.5*Cy2; dt2 = 0.5*dt2
        D1 = 1; D2 = 0
    else:
        D1 = 2; D2 = 1
        
    for i in range(1, u.shape[0]-1):
        for j in range(1, u.shape[1]-1):
            u_xx = u_1[i-1,j] - 2*u_1[i,j] + u_1[i+1,j]
            u_yy = u_1[i,j-1] - 2*u_1[i,j] + u_1[i,j+1]
            u[i,j] = D1*u_1[i,j] - D2*u_2[i,j] + \
                Cx2*u_xx + Cy2*u_yy + dt2*f[i,j]
            if step1:
                u[i,j] += dt*V[i,j]

    #swap variables before next step      
    for i in range(Nx):
        for j in range(Ny):
            u_2[i,j] = u_1[i,j]
            u_1[i,j] = u[i,j]
            
    #Boundary cond u=0        
    for i in Ix: 
        u[i,0] = 0
    for i in Ix: 
        u[i,Ny] = 0
    for j in Iy: 
        u[0,j] = 0
    for j in Iy: 
        u[Nx,j] = 0

    return u

import nose.tools as nt

def test_quadratic(Nx=4, Ny=5):
    def assert_no_error(u, x, xv, y, yv, t, n):
        u_e = exact_solution(xv, yv, t[n])
        diff = abs(u - u_e).max()
        nt.assert_almost_equal(diff, 0, places=10)

    def exact_solution(x, y, t):
        return x*(Lx - x)*y*(Ly - y)*(1 + 0.5*t)

    def I(x, y):
        return exact_solution(x, y, 0)

    def V(x, y):
        return 0.5*exact_solution(x, y, 0)

    def f(x, y, t):
        return 2*c**2*(1 + 0.5*t)*(y*(Ly - y) + x*(Lx - x))

    Lx = 3; Ly = 3
    c = 1.5
    dt = -1 # use longest possible steps
    T = 18

    dt, cpu_time = solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T, user_action=assert_no_error)
    print "Tid: %g s" %cpu_time

if __name__ == '__main__':
    test_quadratic()
