import theano.tensor as T
import theano
import numpy as np

def advance(u, u_1, u_2, f_a, Cx2, Cy2, dt2, V=None, step1=False):
    u_in, u_1_in, u_2_in = T.fmatrices('u_in','u_1_in','u_2_in')
    f_a_in, V_in = T.fmatrices('f_in','V_in')
    step1_in = T.lscalar('step1_in')
    Cx2_in, Cy2_in, dt2_in = T.fscalars('Cx2_in','Cy2_in','dt2_in')

    u_out = T.fmatrix('u_out')

    if V is None:
        V = np.zeros_like(f_a)
        
    step_f = theano.function([u_in, u_1_in, u_2_in, f_a_in, Cx2_in, Cy2_in, dt2_in, V_in, step1_in], u_out, step, on_unused_input='ignore')
    u_out = step_f(u, u_1, u_2, f_a, Cx2, Cy2, dt2, V, step1)

    return u_out

def step(u, u_1, u_2, f_a, Cx2, Cy2, dt2,
                       V, step1):
    if step1:
        dt = sqrt(dt2)  # save
        Cx2 = 0.5*Cx2;  Cy2 = 0.5*Cy2; dt2 = 0.5*dt2  # redefine
        D1 = 1;  D2 = 0
    else:
        D1 = 2;  D2 = 1
    u_xx = u_1[:-2,1:-1] - 2*u_1[1:-1,1:-1] + u_1[2:,1:-1]
    u_yy = u_1[1:-1,:-2] - 2*u_1[1:-1,1:-1] + u_1[1:-1,2:]
    u[1:-1,1:-1] = D1*u_1[1:-1,1:-1] - D2*u_2[1:-1,1:-1] + \
                   Cx2*u_xx + Cy2*u_yy + dt2*f_a[1:-1,1:-1]
    if step1:
        u[1:-1,1:-1] += dt*V[1:-1, 1:-1]
    # Boundary condition u=0
    j = 0
    u[:,j] = 0
    j = u.shape[1]-1
    u[:,j] = 0
    i = 0
    u[i,:] = 0
    i = u.shape[0]-1
    u[i,:] = 0
    return u
