import numpy as np
import pickle
from astropy.io import fits
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from galpy.util import conversion, coords, save_pickles, bovy_plot
from galpy.potential import MWPotential2014, turn_physical_off, vcirc
import astropy.units as u
from galpy.orbit import Orbit
from scipy import integrate, interpolate
from scipy.integrate import quad
from optparse import OptionParser
import numpy
import mockgd1_util
import random

ro=8.
vo=220.
def get_options():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    
        
####### SET THE STREAM #####################################                         
                         
   # parser.add_option("--prog",dest='prog',default=0.,
   #                   type='float',
   #                   help="prog loc in phi1: 0.,-40.0")
                      
    parser.add_option("--sigv",dest='sigv',default=0.46,
                      type='float',
                      help="velocity dispersion")

    parser.add_option("--sim_ind",dest='sim_ind',default=0,
                      type='int',
                      help="simulation index")
                            
    parser.add_option("--leading_arm",action="store_false",dest='leading_arm',default=True,
                      help="leading or trailing arm")
             

                  
#############################################################################################################################################
################# SUBHALO STUFF ############################################################################

# Parameters of this simulation
    
    parser.add_option("-X",dest='Xrs',default=5.,
                      type='float',
                      help="Number of times rs to consider for the impact parameter")
    
    parser.add_option("-M",dest='mass',default='6.5',
                      help="Mass or mass range to consider; given as log10(mass)")

    parser.add_option("--timescdm",dest='timescdm',default='1.',type='float',
                      help="timescdm")
                                            
    parser.add_option("--ravg",dest='ravg',default=17.,type='float',
                      help="r_avg of the stream in kpc")                  
                     
    parser.add_option("--cutoff",dest='cutoff',default=None,type='float',
                      help="Log10 mass cut-off in power-spectrum")
                      
    parser.add_option("--massexp",dest='massexp',default=-2.,type='float',
                      help="Exponent of the mass spectrum (doesn't work with cutoff)")
                      
    parser.add_option("--ratemin",dest='ratemin',default=-1.,type='float',
                      help="minimum timescdm")
                      
    parser.add_option("--ratemax",dest='ratemax',default=1.,type='float',
                      help="maximum timescdm")
                      
    parser.add_option("--rsfac",dest='rsfac',default=1.,type='float',
                      help="Use a r_s(M) relation that is a factor of rsfac different from the fiducial one")
                      
    parser.add_option("--plummer",action="store_true", 
                      dest="plummer",default=False,
                      help="If set, use a Plummer DM profile rather than Hernquist")
                                               
    parser.add_option("--sigma",dest='sigma',default=120.,type='float',
                      help="Velocity dispersion of the population of DM subhalos")
    
    return parser

parser= get_options()
options,args= parser.parse_args()



###########SUBHALO STUFF#############
def parse_times(times,age):
    if 'sampling' in times:
        nsam= int(times.split('sampling')[0])
        return [float(ti)/conversion.time_in_Gyr(vo,ro)
                for ti in numpy.arange(1,nsam+1)/(nsam+1.)*age]
    return [float(ti)/conversion.time_in_Gyr(vo,ro)
            for ti in times.split(',')]
def parse_mass(mass):   
    return [float(m) for m in mass.split(',')]

# Functions to sample

def nsubhalo(m):
    return 0.3*(10.**6.5/m)

def rs(m,plummer=options.plummer,rsfac=1.,rs_fit=True,rs_Nbody_kpc=0.2):
    #rs_sample: if True uses the rs(M) fit to determine rs, if false then returns rs_Nbody_kpc/2.8/ro
    #NOTE: the division by 2.8 is because the softening length used in N-Body sims (Gizmo) is generally 2.8 times the Plummer softening
    if rs_fit:
        
        if plummer:
        #print ('Plummer')
            return 1.62*rsfac/ro*(m/10.**8.)**0.5
        else:
            return 1.05*rsfac/ro*(m/10.**8.)**0.5

    else :
        return rs_Nbody_kpc/2.8/ro

h=0.6774


def alpha(m_wdm):
    return (0.048/h)*(m_wdm)**(-1.11) #in Mpc , m_wdm in keV

def lambda_hm(m_wdm):
    nu=1.12
    return 2*numpy.pi*alpha(m_wdm)/(2**(nu/5.) - 1.)**(1/(2*nu))

def M_hm(m_wdm):
    Om_m=0.3089
    rho_c=1.27*10**11 #Msun/Mpc^3 
    rho_bar=Om_m*rho_c
    return (4*numpy.pi/3)*rho_bar*(lambda_hm(m_wdm)/2.)**3

def Einasto(r):
    al=0.678 #alpha_shape
    rm2=199 #kpc, see Erkal et al 1606.04946 for scaling to M^1/3
    return numpy.exp((-2./al)*((r/rm2)**al -1.))*4*numpy.pi*(r**2)

def dndM_cdm(M,c0kpc=2.02*10**(-13),mf_slope=-1.9):
    #c0kpc=2.02*10**(-13) #Msun^-1 kpc^-3 from Denis' paper
    m0=2.52*10**7 #Msun from Denis' paper
    return c0kpc*((M/m0)**mf_slope)

def fac(M,m_wdm):
    beta=-0.99
    gamma=2.7
    return (1.+gamma*(M_hm(m_wdm)/M))**beta
    
def dndM_wdm(M,m_wdm,c0kpc=2.02*10**(-13),mf_slope=-1.9):
    return fac(M,m_wdm)*dndM_cdm(M,c0kpc=2.02*10**(-13),mf_slope=-1.9)

def nsub_cdm(M1,M2,r=20.,c0kpc=2.02*10**(-13),mf_slope=-1.9):
    #number density of subhalos in kpc^-3
    m1=10**(M1)
    m2=10**(M2)
    return integrate.quad(dndM_cdm,m1,m2,args=(c0kpc,mf_slope))[0]*integrate.quad(Einasto,0.,r)[0]*(8.**3.)/(4*numpy.pi*(r**3)/3) #in Galpy units


def simulate_cdm_subhalos(sdf_pepper,log10Msub_min=5.,log10Msub_max=9.,timescdm=1.,
                          mf_slope=-1.9,c0kpc=2.02*10**(-13),r=17.,Xrs=5.,sigma=120./220.,
                          rs_fit=True,rs_Nbody_kpc=0.2):
    
    '''
    Sample amp and slope such that dN/dM = amp*M^slope and simulate subhalo impacts
    
    '''
    Mbin_edge=np.arange(log10Msub_min,log10Msub_max+1,1)
    Nbins=len(Mbin_edge)-1
    #compute number of subhalos in each mass bin
    nden_bin=np.empty(Nbins)
    rate_bin=np.empty(Nbins)
    for ll in range(Nbins):
        nden_bin[ll]=nsub_cdm(Mbin_edge[ll],Mbin_edge[ll+1],r=r,c0kpc=c0kpc,mf_slope=mf_slope) 
        Mmid=10**(0.5*(Mbin_edge[ll]+Mbin_edge[ll+1]))
        rate_bin[ll]=sdf_pepper.subhalo_encounters(sigma=sigma,nsubhalo=nden_bin[ll],bmax=Xrs*rs(Mmid,plummer=True,rs_fit=rs_fit,rs_Nbody_kpc=rs_Nbody_kpc))

    rate = timescdm*np.sum(rate_bin)
          
    Nimpact= numpy.random.poisson(rate)

    if Nbins ==1 : #if only one bin then all subhalos are assigned the same mean mass
        sample_GM= lambda: 10.**(0.5*(Mbin_edge[0]+Mbin_edge[1]))/conversion.mass_in_msol(vo,ro)
        timpact_sub= numpy.array(sdf_pepper._uniq_timpact)[numpy.random.choice(len(sdf_pepper._uniq_timpact),size=Nimpact,
                                p=sdf_pepper._ptimpact)]
        # Sample angles from the part of the stream that existed then
        impact_angle_sub= numpy.array([sdf_pepper._icdf_stream_len[ti](numpy.random.uniform()) for ti in timpact_sub])
        GM_sub= numpy.array([sample_GM() for a in impact_angle_sub])
        rs_sub= numpy.array([rs(gm*conversion.mass_in_msol(vo,ro),rs_fit=rs_fit,rs_Nbody_kpc=rs_Nbody_kpc) for gm in GM_sub])

    else:
              
        norm= 1./quad(lambda M : M**(mf_slope +0.5),10**(Mbin_edge[0]),10**(Mbin_edge[Nbins]))[0]

        def cdf(M):
            return quad(lambda M : norm*(M)**(mf_slope +0.5),10**Mbin_edge[0],M)[0]

        MM=numpy.linspace(Mbin_edge[0],Mbin_edge[Nbins],10000)

        cdfl=[cdf(i) for i in 10**MM]
        icdf= interpolate.InterpolatedUnivariateSpline(cdfl,10**MM,k=1)
        sample_GM=lambda: icdf(numpy.random.uniform())/conversion.mass_in_msol(vo,ro)
        timpact_sub= numpy.array(sdf_pepper._uniq_timpact)[numpy.random.choice(len(sdf_pepper._uniq_timpact),size=Nimpact,
                                p=sdf_pepper._ptimpact)]
        # Sample angles from the part of the stream that existed then
        impact_angle_sub= numpy.array([sdf_pepper._icdf_stream_len[ti](numpy.random.uniform()) for ti in timpact_sub])
        GM_sub= numpy.array([sample_GM() for a in impact_angle_sub])
        rs_sub= numpy.array([rs(gm*conversion.mass_in_msol(vo,ro),rs_fit=rs_fit,rs_Nbody_kpc=rs_Nbody_kpc) for gm in GM_sub])
   
    
    # impact b
    impactb_sub= (2.*numpy.random.uniform(size=len(impact_angle_sub))-1.)*Xrs*rs_sub
    # velocity
    
    subhalovel_sub= numpy.empty((len(impact_angle_sub),3))
    for ii in range(len(timpact_sub)):
        subhalovel_sub[ii]=sdf_pepper._draw_impact_velocities(timpact_sub[ii],sigma,impact_angle_sub[ii],n=1)[0]
    # Flip angle sign if necessary
    if not sdf_pepper._gap_leading: impact_angle_sub*= -1.
         
    return impact_angle_sub,impactb_sub,subhalovel_sub,timpact_sub,GM_sub,rs_sub


sigv=options.sigv

if options.leading_arm :
            sdf='mockgd1_pepper_leading_Plummer_sigv0.46_td4.0_86sampling_InterpSphPotAnalyticDisk.pkl'
            arm='leading'
else :
            sdf='mockgd1_pepper_trailing_Plummer_sigv0.46_td4.0_86sampling_InterpSphPotAnalyticDisk.pkl'
            arm='trailing'

with open(sdf,'rb') as savefile:
        sdf_smooth= pickle.load(savefile,encoding='latin1')
        sdf_pepper= pickle.load(savefile,encoding='latin1')
                                                                              

for ii in range(10):

        impact_angles,impactbs,subhalovels,timpacts,GMs,rss =simulate_cdm_subhalos(sdf_pepper,r=options.ravg,Xrs=options.Xrs,
                                                                                   timescdm=options.timescdm,rs_fit=False,rs_Nbody_kpc=0.2,
                                                                                   log10Msub_min=4.5,log10Msub_max=5.5)    
        #print (GMs)
        #print (rss)
        print ('%i subhalo impact'%len(GMs))   
   
        if len(GMs) == 0 : #no hits
               print ("no hits")
               apar_out=np.arange(0.01,2.0,0.01)
               #sdf_smooth=gd1_util.setup_gd1model(leading=options.leading_arm,age=tstream,new_orb_lb=new_orb_lb,isob=isob,sigv=sigv)
               dens_unp= [sdf_smooth._density_par(a) for a in apar_out]
               omega_unp= [sdf_smooth.meanOmega(a,oned=True) for a in apar_out]
      
               fo=open('dens_Omega/1e5Msun_rs_Nbody0p2kpc_{}xcdm/{}/mockGD1_sigv{}_{}_densOmega_CDMsubhalo1e5Msun_Plummer_{}.dat'
                       .format(options.timescdm,arm,sigv,arm,random.randint(0,1000000)),'w')
               fo.write('#apar   dens_unp   dens  omega_unp   omega' + '\n')
        
               for  jj in range(len(apar_out)):
                   fo.write(str(apar_out[jj]) + '   ' + str(dens_unp[jj]) + '   ' + str(dens_unp[jj]) + '   ' + str(omega_unp[jj]) + '   ' + str(omega_unp[jj]) + '\n' )
                
               fo.close() 
        
    
        else :
               sdf_pepper.set_impacts(impactb=impactbs,subhalovel=subhalovels,impact_angle=impact_angles,timpact=timpacts,rs=rss,GM=GMs)
               apar_out=np.arange(0.01,2.0,0.01)
               dens_unp= [sdf_smooth._density_par(a) for a in apar_out]
               omega_unp= [sdf_smooth.meanOmega(a,oned=True) for a in apar_out]
               densOmega= np.array([sdf_pepper._densityAndOmega_par_approx(a) for a in apar_out]).T
        
               fo=open('dens_Omega/1e5Msun_rs_Nbody0p2kpc_{}xcdm/{}/mockGD1_sigv{}_{}_densOmega_CDMsubhalo1e5Msun_Plummer_{}.dat'
                       .format(options.timescdm,arm,sigv,arm,random.randint(0,1000000)),'w')
                                                                                                                
               fo.write('#apar   dens_unp   dens  omega_unp   omega' + '\n')
        
               for  jj in range(len(apar_out)):
                   fo.write(str(apar_out[jj])+ '   '+str(dens_unp[jj])+'   '+str(densOmega[0][jj]) + '   ' + str(omega_unp[jj]) + '   ' + str(densOmega[1][jj]) + '\n' )
                
               fo.close()
