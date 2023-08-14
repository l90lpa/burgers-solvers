import numpy as np

def ppmlin_step_advection(z,v,dt,dx,N):

    # subroutine for Lin scheme in 1D

    flux = np.empty((N,), dtype=np.float64)
    deltaz = np.empty((N,),dtype=np.float64)
    deltazmax = np.empty((N,),dtype=np.float64)
    deltazmin = np.empty((N,),dtype=np.float64)
    deltazmono = np.empty((N,),dtype=np.float64)
    zplus = np.empty((N,),dtype=np.float64)
    zminus = np.empty((N,),dtype=np.float64)
    dl = np.empty((N,),dtype=np.float64)
    dr = np.empty((N,),dtype=np.float64)
    zminusnew = np.empty((N,),dtype=np.float64)
    zplusnew = np.empty((N,),dtype=np.float64)
    z6 = np.empty((N,),dtype=np.float64)

    # flux to left 

    for i in range(N):
        ai=(i+1) % N
        bi=(i-1) % N
            
        deltaz[i]=0.25*(z[ai]-z[bi])

        deltazmax[i] = max(z[bi],z[i],z[ai])-z[i]
        deltazmin[i] = z[i]-min(z[bi],z[i],z[ai])

    
    for i in range(N):
        ai=(i+1) % N
        bi=(i-1) % N
            
        minpart=min(abs(deltaz[i]),deltazmin[i],deltazmax[i]);

        if (deltaz[i] >= 0.0):
            deltazmono[i]=abs(minpart)
        else:
            deltazmono[i]=-abs(minpart)
                
    
    # Approx of z at the cell edge

    for i in range(N):
        ai=(i+1) % N
        bi=(i-1) % N
            
        zminus[i]=0.5*(z[bi]+z[i])+(1.0/3.0)*(deltazmono[bi]-deltazmono[i])
    
    
    # continuous at cell edges

    for i in range(N):
        ai=(i+1) % N
        bi=(i-1) % N
            
        zplus[i]=zminus[ai]
    
    
    # limit cell edges

    for i in range(N):
        ai=(i+1) % N
        bi=(i-1) % N
            
        dl[i]=min(abs(2.0*deltazmono[i]),abs(zminus[i]-z[i]))
        dr[i]=min(abs(2.0*deltazmono[i]),abs(zplus[i]-z[i]))

        if (2.0*deltazmono[i] >= 0.0):
            dl[i] = abs(dl[i])
            dr[i] = abs(dr[i])
        else:
            dl[i] = -abs(dl[i])
            dr[i] = -abs(dr[i])
            
        zminusnew[i]=z[i] - dl[i]
        zplusnew[i]=z[i] + dr[i]
    
    
    for i in range(N):
        ai=(i+1) % N
        bi=(i-1) % N
            
        zminus[i]=zminusnew[i]
        zplus[i]=zplusnew[i]  
    
        #z6[i] = 6.0*(z[i] - 0.5*(zminus[i]+zplus[i]))
        z6[i] = 3.0*(dl[i]-dr[i])
    
    
    # calculate flux based on velocity direction

    for i in range(N):
        ai=(i+1) % N
        bi=(i-1) % N
            
        if (v >= 0.0):
            flux[i] = zplus[bi] - 0.5*(v*dt/dx)*( -zminus[bi]+zplus[bi] - z6[bi]*(1.0-(2.0/3.0)*(v*dt/dx)) )
        else:
            flux[i] = zminus[i] + 0.5*(abs(v)*dt/dx)*( zplus[i]-zminus[i] + z6[i]*(1.0-(2.0/3.0)*(abs(v)*dt/dx)) )

    z_new = np.empty((N,), dtype=np.float64)
    z_new[:N-1] = -(1.0/dx) * (flux[1:] - flux[:N-1])
    z_new[N-1] = -(1.0/dx) * (flux[0] - flux[N-1])
    return z_new


def ppmunlimited_step_advection(z,v,dt,dx,N):

    # subroutine for PPM (without limiters) in 1D
	
    flux = np.empty((N,), dtype=np.float64)
    zL = np.empty((N,), dtype=np.float64)
    zR = np.empty((N,), dtype=np.float64)
    z6 = np.empty((N,), dtype=np.float64)
    zD = np.empty((N,), dtype=np.float64)
    zp = np.empty((N,), dtype=np.float64)

    for i in range(N):
        ci=(i+2) % N
        ai=(i+1) % N
        bi=(i-1) % N

        # Using the cell average values `z`, a 'reconstruction'/approximation of z over the cells is formed, and this reconstruction
        # is used to a value for z at the cell interfaces.
	
        zp[i]=(-z[ci]+7.0*z[ai]+7.0*z[i]-z[bi])/12.0


    for i in range(N):
        ai=(i+1) % N
        bi=(i-1) % N

        # Construct a parabola on each cell using the cell interface values `zp`, and the cell average values `z`

        zL[i] = zp[bi]
        zR[i] = zp[i]
        zD[i] = zR[i] - zL[i]
        z6[i] = 6.0*(z[i]-0.5*(zL[i]+zR[i]))

    for i in range(N):
        ai=(i+1) % N
        bi=(i-1) % N

        # Compute the flux through the cell interfaces

        if (v >= 0.0):
            flux[i] = zR[bi] - 0.5*(abs(v)*dt/dx)*( zD[bi] - z6[bi]*(1.0-(2.0/3.0)*(abs(v)*dt/dx)) )
        else:
            flux[i] = zL[i] + 0.5*(abs(v)*dt/dx)*( zD[i] + z6[i]*(1.0-(2.0/3.0)*(abs(v)*dt/dx)) )
    

    z_new = np.empty((N,), dtype=np.float64)
    z_new[:N-1] = -(1.0/dx) * (flux[1:] - flux[:N-1])
    z_new[N-1] = -(1.0/dx) * (flux[0] - flux[N-1])
    return z_new
