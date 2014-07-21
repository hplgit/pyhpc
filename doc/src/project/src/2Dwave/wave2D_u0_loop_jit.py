from numbapro import *

@jit('float64[:,::1](float64[:,::1],float64[:,::1],float64[:,::1],float64[:,::1],float64[:,::1], float64, float64, int64, int64, int64, float64, int64)', nopython=True)
def advance(u, u_1, u_2, f, V, Cx2, Cy2, Nx, Ny, n, dt, step1):
    Ix = range(0, u.shape[0])
    Iy = range(0, u.shape[1])
    dt2 = dt**2

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
