import numpy as np
cimport numpy as np
from cython.parallel cimport *
cimport cython
cimport openmp
ctypedef np.float64_t DT    # data type

@cython.boundscheck(False)  # turn off array bounds check
@cython.wraparound(False)   # turn off negative indices (u[-1,-1])
@cython.cdivision(True)
cpdef advance(
    np.ndarray[DT, ndim=2, mode='c'] u,
    np.ndarray[DT, ndim=2, mode='c'] u_1,
    np.ndarray[DT, ndim=2, mode='c'] u_2,
    np.ndarray[DT, ndim=2, mode='c'] f,
    double Cx2, double Cy2, double dt2):

    cdef:
        int Ix_start = 0
        int Iy_start = 0
        int Ix_end = u.shape[0]-1
        int Iy_end = u.shape[1]-1
        int num_thr
        int xsize
        int ysize
        int i, j
        double u_xx, u_yy

    xsize = (Ix_end - Ix_start)/2
    ysize = (Iy_end - Iy_start)/2

    for i in prange(Ix_start+1, Ix_end, nogil=True, schedule='static', chunksize=xsize, num_threads=2):
        for j in prange(Iy_start+1, Iy_end, schedule='static', chunksize=ysize, num_threads=2):
            u_xx = u_1[i-1,j] - 2*u_1[i,j] + u_1[i+1,j]
            u_yy = u_1[i,j-1] - 2*u_1[i,j] + u_1[i,j+1]
            u[i,j] = 2*u_1[i,j] - u_2[i,j] + Cx2*u_xx + Cy2*u_yy + dt2*f[i,j]
    
            
    # Boundary condition u=0
    j = Iy_start
    for i in range(Ix_start, Ix_end+1): u[i,j] = 0
    j = Iy_end
    for i in range(Ix_start, Ix_end+1): u[i,j] = 0
    i = Ix_start
    for j in range(Iy_start, Iy_end+1): u[i,j] = 0
    i = Ix_end
    for j in range(Iy_start, Iy_end+1): u[i,j] = 0
    return u
