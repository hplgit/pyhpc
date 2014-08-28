import numexpr as ne

def advance(u, u_1, u_2, f_a, Cx2, Cy2, dt2, V=None, step1=False):
    if step1:
        #ne.set_vml_num_threads(2)
        dt = sqrt(dt2)
        Cx2 = 0.5*Cx2;  Cy2 = 0.5*Cy2; dt2 = 0.5*dt2
        D1 = 1;  D2 = 0
    else:
        D1 = 2;  D2 = 1

    u_xx = ne.evaluate("u_10 - 2*u_11 + u_12", local_dict={"u_10":u_1[:-2,1:-1],
                                                           "u_11":u_1[1:-1,1:-1],
                                                           "u_12":u_1[2:,1:-1]})
    u_yy = ne.evaluate("u_10 - 2*u_11 + u_12", local_dict={"u_10":u_1[1:-1,:-2],
                                                           "u_11":u_1[1:-1,1:-1],
                                                           "u_12":u_1[1:-1,2:]})
    u[1:-1,1:-1] = ne.evaluate("D1*u_11 - D2*u_21 + Cx2*u_xx + Cy2*u_yy + dt2*f_a", 
                               local_dict={"u_11":u_1[1:-1,1:-1], "u_21":u_2[1:-1,1:-1], 
                                           "D1":D1, "D2":D2,"u_xx": u_xx, "u_yy":u_yy,
                                           "Cx2":Cx2, "Cy2":Cy2, "dt2":dt2, "f_a":f_a[1:-1,1:-1]})

    if step1:
        u[1:-1,1:-1] += ne.evaluate("dt*V", local_dict={"V":V[1:-1,1:-1], "dt":dt})
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
