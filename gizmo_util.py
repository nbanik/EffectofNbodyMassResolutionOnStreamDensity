import h5py
import pickle
import numpy as np
from scipy.stats import gaussian_kde
from galpy.util import _rotate_to_arbitrary_vector


'''

read_snapshot,
extract_data_of_ptype,
make_IC,
make_gizmo_IC,
Gizmo_to_pynbody
compute_surface_density
find_centre_of_density
find_centre
align_stream

'''

    
def read_snapshot(fname='test.hdf5',read_axes='Coordinates',ptype=1):
    data = h5py.File(fname, 'r')
    group = data['PartType{}'.format(int(ptype))]
    
    try:
        dd=group[read_axes][()]
        try:
            if np.size(dd,1) ==3:
                return (dd[:,0],dd[:,1],dd[:,2])
    
        except IndexError:
            return (dd)
    
    except KeyError:
        print ("Allowed axes names %s"%group.keys())
        
        
def extract_data_of_ptype(folder="",ptype=1,nfiles=100,shift_file_ind=0,shift_ID=0,pot_acc=True):
    #currently only supports: coordinates, velocities, mass, ID, acceleration,potential
    ids=[];m=[];x=[];y=[];z=[];vx=[];vy=[];vz=[];pot=[];ax=[];ay=[];az=[];tt=[];

    time_series = [0.01*(i+shift_file_ind) for i in range(nfiles-shift_file_ind)]

    #print (time_series)

    for nn in range(shift_file_ind,nfiles):
                
        tt.append(time_series[nn-shift_file_ind])
        _x,_y,_z=read_snapshot(fname=folder + 'snapshot_{:03d}.hdf5'.format(nn),ptype=ptype)
        _vx,_vy,_vz=read_snapshot(fname=folder + 'snapshot_{:03d}.hdf5'.format(nn),ptype=ptype,read_axes="Velocities")
        _ids=read_snapshot(fname=folder + 'snapshot_{:03d}.hdf5'.format(nn),ptype=ptype,read_axes="ParticleIDs")
        _ms=read_snapshot(fname=folder + 'snapshot_{:03d}.hdf5'.format(nn),ptype=ptype,read_axes="Masses")
        
        _x = _x[_ids >=shift_ID]
        _y = _y[_ids >=shift_ID]
        _z = _z[_ids >=shift_ID]
        _vx = _vx[_ids >=shift_ID]
        _vy = _vy[_ids >=shift_ID]
        _vz = _vz[_ids >=shift_ID]
        _ms = _ms[_ids >=shift_ID]
                      
        if pot_acc:
            _pot=read_snapshot(fname=folder + 'snapshot_{:03d}.hdf5'.format(nn),ptype=ptype,read_axes="Potential")
            _pot = _pot[_ids >=shift_ID]
            _ax,_ay,_az=read_snapshot(fname=folder + 'snapshot_{:03d}.hdf5'.format(nn),ptype=ptype,read_axes="Acceleration")
            _ax = _ax[_ids >=shift_ID]
            _ay = _ay[_ids >=shift_ID]
            _az = _az[_ids >=shift_ID]
        
        _ids=_ids[_ids >=shift_ID]
        
                    
        id_sort = _ids.argsort()

        _ids=_ids[id_sort]
        _x=_x[id_sort]
        _y=_y[id_sort]
        _z=_z[id_sort]
        _vx=_vx[id_sort]
        _vy=_vy[id_sort]
        _vz=_vz[id_sort]
        _ms=_ms[id_sort]
        
        ids.append(_ids)
        x.append(_x)
        y.append(_y)
        z.append(_z)
        vx.append(_vx)
        vy.append(_vy)
        vz.append(_vz)
        m.append(_ms) 
        
        if pot_acc:
            _pot = _pot[id_sort]
            _ax = _ax[id_sort]
            _ay = _ay[id_sort]
            _az = _az[id_sort]
            pot.append(_pot)
            ax.append(_ax)
            ay.append(_ay)
            az.append(_az)
    
    if pot_acc:
        return (tt,ids,x,y,z,vx,vy,vz,m,pot,ax,ay,az)
            
    return (tt,ids,x,y,z,vx,vy,vz,m)

        

def make_IC(xv_d,yv_d,zv_d,vx_d,vy_d,vz_d,mv_d,fn='test',ptype=1):
    '''
    This is an example subroutine provided to demonstrate how to make HDF5-format
    ICs for GIZMO. The specific example here is arbitrary, but can be generalized
    to whatever IC you need
    
    Modified by Nil: This now takes (x,y,z,vx,vy,vz,m) of a prticular particle type
    and then generates a Gizmo compatible IC. This is useful incase we want to extract a certain subset 
    of particles from another snapshot/IC and use that as an IC for another run.
    
    As of now, it only handles one particle type at a time.
    
     
    
    '''

    fname=fn; # output filename 

    # now we get ready to actually write this out
    #  first - open the hdf5 ics file, with the desired filename
    file = h5py.File(fname,'w') 

    # set particle number of each type into the 'npart' vector
    #  NOTE: this MUST MATCH the actual particle numbers assigned to each type, i.e.
    #   npart = np.array([number_of_PartType0_particles,number_of_PartType1_particles,number_of_PartType2_particles,
    #                     number_of_PartType3_particles,number_of_PartType4_particles,number_of_PartType5_particles])
    #   or else the code simply cannot read the IC file correctly!
    Ngrains=len(xv_d)
    npart = np.array([0,0,0,0,0,0]) # we have gas and particles we will set for type 3 here, zero for all others
    npart[ptype]+=Ngrains
    # now we make the Header - the formatting here is peculiar, for historical (GADGET-compatibility) reasons
    h = file.create_group("Header");
    # here we set all the basic numbers that go into the header
    # (most of these will be written over anyways if it's an IC file; the only thing we actually *need* to be 'correct' is "npart")
    h.attrs['NumPart_ThisFile'] = npart; # npart set as above - this in general should be the same as NumPart_Total, it only differs 
                                         #  if we make a multi-part IC file. with this simple script, we aren't equipped to do that.
    h.attrs['NumPart_Total'] = npart; # npart set as above
    h.attrs['NumPart_Total_HighWord'] = 0*npart; # this will be set automatically in-code (for GIZMO, at least)
    h.attrs['MassTable'] = np.zeros(6); # these can be set if all particles will have constant masses for the entire run. however since 
                                        # we set masses explicitly by-particle this should be zero. that is more flexible anyways, as it 
                                        # allows for physics which can change particle masses 
    ## all of the parameters below will be overwritten by whatever is set in the run-time parameterfile if
    ##   this file is read in as an IC file, so their values are irrelevant. they are only important if you treat this as a snapshot
    ##   for restarting. Which you shouldn't - it requires many more fields be set. But we still need to set some values for the code to read
    h.attrs['Time'] = 0.0;  # initial time
    h.attrs['Redshift'] = 0.0; # initial redshift
    h.attrs['BoxSize'] = 1.0; # box size
    h.attrs['NumFilesPerSnapshot'] = 1; # number of files for multi-part snapshots
    h.attrs['Omega0'] = 1.0; # z=0 Omega_matter
    h.attrs['OmegaLambda'] = 0.0; # z=0 Omega_Lambda
    h.attrs['HubbleParam'] = 1.0; # z=0 hubble parameter (small 'h'=H/100 km/s/Mpc)
    h.attrs['Flag_Sfr'] = 0; # flag indicating whether star formation is on or off
    h.attrs['Flag_Cooling'] = 0; # flag indicating whether cooling is on or off
    h.attrs['Flag_StellarAge'] = 0; # flag indicating whether stellar ages are to be saved
    h.attrs['Flag_Metals'] = 0; # flag indicating whether metallicity are to be saved
    h.attrs['Flag_Feedback'] = 0; # flag indicating whether some parts of springel-hernquist model are active
    h.attrs['Flag_DoublePrecision'] = 0; # flag indicating whether ICs are in single/double precision
    h.attrs['Flag_IC_Info'] = 0; # flag indicating extra options for ICs
    ## ok, that ends the block of 'useless' parameters
    
    # Now, the actual data!
    #   These blocks should all be written in the order of their particle type (0,1,2,3,4,5)
    #   If there are no particles of a given type, nothing is needed (no block at all)
    #   PartType0 is 'special' as gas. All other PartTypes take the same, more limited set of information in their ICs
    
    # now assign the collisionless particles to PartType1. note that this block looks exactly like 
    #   what we had above for the gas. EXCEPT there are no "InternalEnergy" or "MagneticField" fields (for 
    #   obvious reasons). 
    
    # use index as id
    id_d=np.arange(0,Ngrains,1)
    
    
    p = file.create_group("PartType{}".format(ptype))
    q=np.zeros((Ngrains,3)); q[:,0]=xv_d; q[:,1]=yv_d; q[:,2]=zv_d;
    p.create_dataset("Coordinates",data=q)
    q=np.zeros((Ngrains,3)); q[:,0]=vx_d; q[:,1]=vy_d; q[:,2]=vz_d;
    p.create_dataset("Velocities",data=q)
    p.create_dataset("ParticleIDs",data=id_d)
    p.create_dataset("Masses",data=mv_d)

    # no PartType4 for this IC
    # no PartType5 for this IC

    # close the HDF5 file, which saves these outputs
    file.close()
    # all done!
    
def make_gizmo_IC(fname_list,offset_list=[],fout_name="test_out.hdf5"):
    
    '''
    This takes in a list of ICs generated by GalIC or something and combined them into one 
    IC in a format compatible with Gizmo
    
    offset_list: a list with each element an array or list of phase space offset for each file
    For no offset like for the main halo, simply put [0. 0. 0. 0. 0. 0.] 
    phase space coordinate must be in the form [x(kpc),y(kpc),z(kpc),vx(km/s),vy(km/s),vz(km/s)]
    order of offset elements must follow that in the fname list
    
        
    '''
    #set offsets to 0 in case no offset provided
    if len(offset_list) == 0:
        _ofs=np.zeros(6)
        offset_list = [_ofs]*len(fname_list)
    
    
    npart=np.array([0,0,0,0,0,0])
    pos=[None]*6
    vel=[None]*6
    mass=[None]*6
    ids=[None]*6
        
    for ii in range(6):
        x=[]
        y=[]
        z=[]
        vx=[]
        vy=[]
        vz=[]
        m=[]
        
        print (ii)
        for ff in range(len(fname_list)):
            data = h5py.File(fname_list[ff], 'r')
            if "PartType{}".format(ii) in data.keys():
                _x,_y,_z = read_snapshot(fname=fname_list[ff],read_axes='Coordinates',ptype=ii)
                _vx,_vy,_vz = read_snapshot(fname=fname_list[ff],read_axes='Velocities',ptype=ii)
                _m = read_snapshot(fname=fname_list[ff],read_axes='Masses',ptype=ii)
                              
                #add offsets
                _x+=offset_list[ff][0];_y+=offset_list[ff][1];_z+=offset_list[ff][2];
                _vx+=offset_list[ff][3];_vy+=offset_list[ff][4];_vz+=offset_list[ff][5];
                                 
                x.append(_x);y.append(_y);z.append(_z);
                vx.append(_vx);vy.append(_vy);vz.append(_vz);
                m.append(_m)
                                
                                     
            else:
                print ("No PartType %i found in %s"%(ii,fname_list[ff]))
                pass
        
        x=np.array([st for sublist in x for st in sublist]);y=np.array([st for sublist in y for st in sublist]);
        z=np.array([st for sublist in z for st in sublist]);vx=np.array([st for sublist in vx for st in sublist]);
        vy=np.array([st for sublist in vy for st in sublist]);vz=np.array([st for sublist in vz for st in sublist]);
        m=np.array([st for sublist in m for st in sublist]);
        
        print (len(x))
        ids[ii]=np.arange(0,len(m),1)
        npart[ii]+=len(x)
        pos[ii]=np.c_[x,y,z]
        vel[ii]=np.c_[vx,vy,vz]
        mass[ii]=m
        
    
    print (npart)
    # now we get ready to actually write this out
    #  first - open the hdf5 ics file, with the desired filename
    
    file = h5py.File(fout_name,'w') 

    h = file.create_group("Header");
    # here we set all the basic numbers that go into the header
    # (most of these will be written over anyways if it's an IC file; the only thing we actually *need* to be 'correct' is "npart")
    h.attrs['NumPart_ThisFile'] = npart; # npart set as above - this in general should be the same as NumPart_Total, it only differs 
                                         #  if we make a multi-part IC file. with this simple script, we aren't equipped to do that.
    h.attrs['NumPart_Total'] = npart; # npart set as above
    h.attrs['NumPart_Total_HighWord'] = 0*npart; # this will be set automatically in-code (for GIZMO, at least)
    h.attrs['MassTable'] = np.zeros(6); # these can be set if all particles will have constant masses for the entire run. however since 
                                        # we set masses explicitly by-particle this should be zero. that is more flexible anyways, as it 
                                        # allows for physics which can change particle masses 
    ## all of the parameters below will be overwritten by whatever is set in the run-time parameterfile if
    ##   this file is read in as an IC file, so their values are irrelevant. they are only important if you treat this as a snapshot
    ##   for restarting. Which you shouldn't - it requires many more fields be set. But we still need to set some values for the code to read
    h.attrs['Time'] = 0.0;  # initial time
    h.attrs['Redshift'] = 0.0; # initial redshift
    h.attrs['BoxSize'] = 1.0; # box size
    h.attrs['NumFilesPerSnapshot'] = 1; # number of files for multi-part snapshots
    h.attrs['Omega0'] = 1.0; # z=0 Omega_matter
    h.attrs['OmegaLambda'] = 0.0; # z=0 Omega_Lambda
    h.attrs['HubbleParam'] = 1.0; # z=0 hubble parameter (small 'h'=H/100 km/s/Mpc)
    h.attrs['Flag_Sfr'] = 0; # flag indicating whether star formation is on or off
    h.attrs['Flag_Cooling'] = 0; # flag indicating whether cooling is on or off
    h.attrs['Flag_StellarAge'] = 0; # flag indicating whether stellar ages are to be saved
    h.attrs['Flag_Metals'] = 0; # flag indicating whether metallicity are to be saved
    h.attrs['Flag_Feedback'] = 0; # flag indicating whether some parts of springel-hernquist model are active
    h.attrs['Flag_DoublePrecision'] = 0; # flag indicating whether ICs are in single/double precision
    h.attrs['Flag_IC_Info'] = 0; # flag indicating extra options for ICs
    ## ok, that ends the block of 'useless' parameters
    
    # Now, the actual data!
    #   These blocks should all be written in the order of their particle type (0,1,2,3,4,5)
    #   If there are no particles of a given type, nothing is needed (no block at all)
    #   PartType0 is 'special' as gas. All other PartTypes take the same, more limited set of information in their ICs
    
    # now assign the collisionless particles to PartType1. note that this block looks exactly like 
    #   what we had above for the gas. EXCEPT there are no "InternalEnergy" or "MagneticField" fields (for 
    #   obvious reasons). 
    
    for jj in range(6):
        if npart[jj] != 0 :
                        
                p = file.create_group("PartType{}".format(jj))
                p.create_dataset("Coordinates",data=pos[jj])
                p.create_dataset("Velocities",data=vel[jj])
                p.create_dataset("ParticleIDs",data=ids[jj])
                p.create_dataset("Masses",data=mass[jj])
            
                   
    file.close()
       
        
    return None

def Gizmo_to_pynbody(fn="test.hdf5",ptypes=[1,2,3,4,5],eps_list=[0.1,1.0,1.0,1.0,1.0],galpy_units=False):
    '''
    We are going to merge all ptypes to dm
    '''
    x=[]
    y=[]
    z=[]
    vx=[]
    vy=[]
    vz=[]
    m=[] 
    eps=[]
    for pt in ptypes:
        _x,_y,_z = read_snapshot(fname=fn,read_axes='Coordinates',ptype=pt)
        _vx,_vy,_vz = read_snapshot(fname=fn,read_axes='Velocities',ptype=pt)
        _m=read_snapshot(fname=fn,read_axes='Masses',ptype=pt)
        
        _x = _x.tolist()
        _y = _y.tolist()
        _z = _z.tolist()
        _vx = _vx.tolist()
        _vy = _vy.tolist()
        _vz = _vz.tolist()
        _m = _m.tolist()
        
        x.append(_x)
        y.append(_y)
        z.append(_z)
        vx.append(_vx)
        vy.append(_vy)
        vz.append(_vz)
        m.append(_m)
        
        eps.append(np.ones(len(_x),dtype=np.float64).tolist())
        
    x = np.array([item for sublist in x for item in sublist])
    y = np.array([item for sublist in y for item in sublist])
    z = np.array([item for sublist in z for item in sublist])
    vx = np.array([item for sublist in vx for item in sublist])
    vy = np.array([item for sublist in vy for item in sublist])
    vz = np.array([item for sublist in vz for item in sublist])
    m = np.array([item for sublist in m for item in sublist])
    eps = np.array([item for sublist in eps for item in sublist])

    pos1= np.c_[(x,y,z)]
    vel1= np.c_[(vx,vy,vz)]
        
    f = pynbody.snapshot.new(dm=len(m))
    if galpy_units:
        vo=220.;ro=8.; 
        f['pos'] =pynbody.array.SimArray(pos1/ro)#,"kpc")
        f['vel'] =pynbody.array.SimArray(vel1/vo)#,'km s**-1')
        f['mass'] =pynbody.array.SimArray(m*1e10/bovy_conversion.mass_in_msol(vo,ro))#,'Msol')
        f['eps'] = pynbody.array.SimArray(eps/ro)#,'kpc')
        
    else:
        f['pos'] =pynbody.array.SimArray(pos1,"kpc")
        f['vel'] =pynbody.array.SimArray(vel1,'km s**-1')
        f['mass'] =pynbody.array.SimArray(m*1e10,'Msol')
        f['eps'] = pynbody.array.SimArray(eps,'kpc')
        f.physical_units()
          
    return (f)




def compute_surface_density(x,y,bins=[30,30],bw="scott",method="Gaussian"):
    
    if method == "hist": #may result in ridges in the density
        data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
        dens = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

        #To be sure to plot all data
        dens[np.where(np.isnan(dens))] = 0.0
        idx = dens.argsort()
        x, y, dens = x[idx], y[idx], dens[idx]
        
    elif method == "Gaussian": #slower
          
        xy = np.vstack((x,y))
        dens = gaussian_kde(xy,bw_method=bw)(xy)
        X = np.c_[x,y]
        idx = dens.argsort()
        x, y, dens = X[:,0][idx], X[:,1][idx], dens[idx]
    
    #print (dens.argmax())
    x_core = x[dens.argmax()]
    y_core = y[dens.argmax()]
    return (x_core,y_core,dens,x,y)

def Plot_Surface_Density(ax,x,y,label=""):
    from scipy.stats import gaussian_kde
    # Calculate the point density
    xy = np.vstack((x,y))
    z = gaussian_kde(xy)(xy)
    X = np.c_[x,y]
    idx = z.argsort()
    
    x, y, z = X[:,0][idx], X[:,1][idx], z[idx]
    
    ax.imshow(x,y, c=z, s=10, edgecolor='',alpha=0.5)
    ax.vlines(x[z.argmax()],-100,100,color='k',lw=1,ls="--")
    ax.hlines(y[z.argmax()],-100,100,color='k',lw=1,ls="--")
    ax.scatter(x[z.argmax()],y[z.argmax()],s=10,marker='o',c='b',label=label)
    ax.legend(loc="upper right")
    
    return None

def find_centre(
    x,y,z,vx,vy,vz,m,
    xstart=0.0,
    ystart=0.0,
    zstart=0.0,
    vxstart=0.0,
    vystart=0.0,
    vzstart=0.0,
    indx=None,
    nsigma=1.0,
    nsphere=100,
    density=False,
    rmin=0.1,
    nmax=500,
    ro=8.0,
    vo=220.0,
):
    """Find the cluster's centre

    - The default assumes the cluster's centre is the centre of density, calculated via the find_centre_of_density function.
    - For density=False, the routine first works to identify a sphere of nsphere stars around the centre in which to perform a centre of mass calculation (similar to NBODY6). Stars beyond nsigma standard deviations are removed from the calculation until only nsphere stars remain. This step prevents long tidal tails from affecting the calculation

    Parameters
    ----------
    cluster : class
        StarCluster
    xstart,ystart,zstart : float
        starting position for centre
    vxstart,vystart,vzstart :
        starting velocity for centre
    indx : bool
        subset of stars to use when finding center
    nsigma : int
        number of standard deviations to within which to keep stars
    nsphere : int
        number of stars in centre sphere (default:100)
    density : bool
        use Yohai Meiron's centre of density calculator instead (default: True)
    rmin : float
        minimum radius to start looking for stars
    nmax : int
        maximum number of iterations to find centre
    ro,vo - For converting to and from galpy units (Default: 8., 220.)

    Returns
    -------
    xc,yc,zc,vxc,vyc,vzc - coordinates of centre of mass

    History
    -------
    2019 - Written - Webb (UofT)
    """
    if indx is None:
        indx = np.ones(len(x), bool)
    elif np.sum(indx) == 0.0:
        print("NO SUBSET OF STARS GIVEN")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ids = np.arange(0,len(x),1)
    if density:
        xc,yc,zc,vxc,vyc,vzc=find_centre_of_density(
            cluster=cluster,
            xstart=xstart,
            ystart=ystart,
            zstart=zstart,
            vxstart=vxstart,
            vystart=vystart,
            vzstart=vzstart,
            indx=indx,
            rmin=rmin,
            nmax=nmax,
        )
    else:

        x = x[indx] - xstart
        y = y[indx] - ystart
        z = z[indx] - zstart
        r = np.sqrt(x ** 2.0 + y ** 2.0 + z ** 2.0)
        i_d = ids[indx]

        while len(r) > nsphere:
            sigma = nsigma * np.std(r)
            indx = r < sigma

            if len(r[indx]) > nsphere:
                i_d = i_d[indx]
                x = x[indx] - np.mean(x[indx])
                y = y[indx] - np.mean(y[indx])
                z = z[indx] - np.mean(z[indx])
                r = np.sqrt(x * x + y * y + z * z)
            else:
                break

        # Find centre of mass and velocity of inner stars:
        indx = np.in1d(ids, i_d)

        xc = np.sum(m[indx] * x[indx]) / np.sum(m[indx])
        yc = np.sum(m[indx] * y[indx]) / np.sum(m[indx])
        zc = np.sum(m[indx] * z[indx]) / np.sum(m[indx])

        vxc = np.sum(m[indx] * vx[indx]) / np.sum(m[indx])
        vyc = np.sum(m[indx] * vy[indx]) / np.sum(m[indx])
        vzc = np.sum(m[indx] * vz[indx]) / np.sum(m[indx])

    return xc, yc, zc, vxc, vyc, vzc


def find_centre_of_density(
    x,y,z,vx,vy,vz,m,
    xstart=0.0,
    ystart=0.0,
    zstart=0.0,
    vxstart=0.0,
    vystart=0.0,
    vzstart=0.0,
    indx=None,
    rmin=0.1,
    nmax=100,
):
    """Find cluster's centre of density 

    - The motivation behind this piece of code comes from phigrape (Harfst, S., Gualandris, A., Merritt, D., et al. 2007, NewA, 12, 357) courtesy of Yohai Meiron
    - The routine first finds the centre of density of the whole system, and then works to identify a sphere stars around the centre in which to perform the final centre of density calculation. Stars with radii outside 80% of the maximum radius are removed from the calculation until the final subset of stars are enclosed within a radius rmin. The maximum size of the final subset is nmax. This step prevents long tidal tails from affecting the calculation

    Parameters
    ----------
    cluster : class
        StarCluster
    xstart,ystart,zstart : float
        starting position for centre (default: 0,0,0)
    vxstart,vystart,vzstart : float
        starting velocity for centre (default: 0,0,0)
    indx: bool
        subset of stars to perform centre of density calculation on (default: None)
    rmin : float
        minimum radius of sphere around which to estimate density centre (default: 0.1 cluster.units)
    nmax : float
        maximum number of iterations (default:100)

    Returns
    -------
    xc,yc,zc,vxc,vyc,vzc : float
        coordinates of centre of mass

    HISTORY
    -------
    2019 - Written - Webb (UofT) with Yohai Meiron (UofT)
    """
    if indx is None:
        #indx = np.ones(cluster.ntot, bool)
        indx = np.ones(len(x), bool)

    #m = cluster.m[indx]
    x = x - xstart
    y = y - ystart
    z = z - zstart
    vx = vx - vxstart
    vy = vy - vystart
    vz = vz - vzstart

    r = np.sqrt(x ** 2.0 + y ** 2.0 + z ** 2.0)
    rlim = np.amax(r)

    xdc, ydc, zdc = xstart, ystart, zstart
    vxdc, vydc, vzdc = vxstart, vystart, vzstart

    n = 0

    while (rlim > rmin) and (n < nmax):
        r2 = x ** 2.0 + y ** 2.0 + z ** 2.0
        indx = r2 < rlim ** 2
        nc = np.sum(indx)
        mc = np.sum(m[indx])

        if mc == 0:
            xc, yc, zc = 0.0, 0.0, 0.0
            vxc, vyc, vzc = 0.0, 0.0, 0.0
        else:

            xc = np.sum(m[indx] * x[indx]) / mc
            yc = np.sum(m[indx] * y[indx]) / mc
            zc = np.sum(m[indx] * z[indx]) / mc

            vxc = np.sum(m[indx] * vx[indx]) / mc
            vyc = np.sum(m[indx] * vy[indx]) / mc
            vzc = np.sum(m[indx] * vz[indx]) / mc

        if (mc > 0) and (nc > 100):
            x -= xc
            y -= yc
            z -= zc
            xdc += xc
            ydc += yc
            zdc += zc

            vx -= vxc
            vy -= vyc
            vz -= vzc
            vxdc += vxc
            vydc += vyc
            vzdc += vzc

        else:
            break
        rlim *= 0.8
        n += 1

    return xdc, ydc, zdc,vxdc, vydc, vzdc


def align_stream(fn,rcore= 0.2,output_galcoords=False):
    
    
    x,y,z = read_snapshot(fn,ptype=4)
    vx,vy,vz = read_snapshot(fn,ptype=4,read_axes="Velocities")
    
    #transform Galactocentric coordinates (box frame) to Galactic coordinates
    v_sun = coord.CartesianDifferential([11.1, 244, 7.25]*u.km/u.s)

    gc2=coord.Galactocentric(x=x*u.kpc,y=y*u.kpc,z=z*u.kpc,v_x=vx*(u.km/u.s),\
                         v_y=vy*(u.km/u.s),v_z=vz*(u.km/u.s),galcen_distance=8.*u.kpc,
                                galcen_v_sun=v_sun,
                                z_sun=0*u.pc)

    gal_c2=gc2.transform_to(coord.ICRS)
    gal_c2.representation_type = 'cartesian'
    
    _x = gal_c2.x.value ; _y = gal_c2.y.value ; _z = gal_c2.z.value ;
    _vx = gal_c2.v_x.value ; _vy = gal_c2.v_y.value ; _vz = gal_c2.v_z.value;
    
    
    #find core center
    xc,yc,zc,_,_,_ =  find_centre_of_density(_x,_y,_z,_vx,_vy,_vz,0.46*np.ones(len(_z)))
    
    #select particles within some r around the center
    _r = np.sqrt((_x-xc)**2 + (_y-yc)**2 + (_z-zc)**2)
    
    _x = _x[_r < rcore] ; _y = _y[_r < rcore] ; _z = _z[_r < rcore] ; 
    _vx = _vx[_r < rcore]; _vy = _vy[_r < rcore] ; _vz = _vz[_r < rcore];
    
    #compute angluar momentum
    L = []

    for ii in range(len(_x)):
        r = [_x[ii],_y[ii],_z[ii]]
        v = [_vx[ii],_vy[ii],_vz[ii]]
        #print (r,v)
        L.append(np.cross(r,v))

    L = np.array(L)
    
    mean_L = np.mean(L,axis=0)
    
    #normalize
    mean_l = np.mean(L,axis=0)/np.linalg.norm(np.mean(L,axis=0))
    #print (mean_l)

    rot_mat = _rotate_to_arbitrary_vector(np.atleast_2d(mean_l),np.array([0,0,1]),inv=False,_dontcutsmall=False)
    #print (rot_mat)

    print (np.dot(rot_mat,mean_l))
    r_rot = []
    v_rot = []

    for ii in range(len(x)):
        r = [gal_c2.x.value[ii],gal_c2.y.value[ii],gal_c2.z.value[ii]]
        v = [gal_c2.v_x.value[ii],gal_c2.v_y.value[ii],gal_c2.v_z.value[ii]]
        r_rot.append(np.dot(rot_mat,r)[0])
        v_rot.append(np.dot(rot_mat,v)[0])

    r_rot = np.array(r_rot)
    v_rot = np.array(v_rot)
    
    if output_galcoords :
        return (gal_c2,mean_L,r_rot[:,0],r_rot[:,1],r_rot[:,2],v_rot[:,0],v_rot[:,1],v_rot[:,2])
    else :
        return (r_rot[:,0],r_rot[:,1],r_rot[:,2],v_rot[:,0],v_rot[:,1],v_rot[:,2])

                
  
