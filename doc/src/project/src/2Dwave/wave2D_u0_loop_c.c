#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

#define idx(i,j) (i)*(Ny+1) + j

void advance(double* u, double* u_1, double* u_2, double* f,
	     double Cx2, double Cy2, double dt2, int Nx, int Ny)
{
    int i, j;
    double u_xx, u_yy;
    int num_threads = 2;
    int xsize = Nx/num_threads;
    int ysize = Ny/num_threads;

    omp_set_num_threads(num_threads);
    
  /* Scheme at interior points */
    #pragma omp parallel for schedule(static, xsize)
    for (i=1; i<=Nx-1; i++) {
        #pragma omp parallel for schedule(static, ysize)
	for (j=1; j<=Ny-1; j++) {
	    u_xx = u_1[idx(i-1,j)] - 2*u_1[idx(i,j)] + u_1[idx(i+1,j)];
	    u_yy = u_1[idx(i,j-1)] - 2*u_1[idx(i,j)] + u_1[idx(i,j+1)];
	    u[idx(i,j)] = 2*u_1[idx(i,j)] - u_2[idx(i,j)] +
		Cx2*u_xx + Cy2*u_yy + dt2*f[idx(i,j)];
	}
    }
    /* Boundary conditions */
    j = 0;  for (i=0; i<=Nx; i++) u[idx(i,j)] = 0;
    j = Ny; for (i=0; i<=Nx; i++) u[idx(i,j)] = 0;
    i = 0;  for (j=0; j<=Ny; j++) u[idx(i,j)] = 0;
    i = Nx; for (j=0; j<=Ny; j++) u[idx(i,j)] = 0;
}
