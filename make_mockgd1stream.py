import os, os.path
import pickle
import numpy
import matplotlib
matplotlib.use('agg')
from scipy import interpolate
from matplotlib import cm, pyplot
from streampepperdf_new import streampepperdf
from scipy import integrate, interpolate
from optparse import OptionParser
from galpy.util import conversion
from galpy.actionAngle.actionAngleIsochroneApprox import dePeriod
import mockgd1_util
from galpy.potential import NFWPotential, MiyamotoNagaiPotential,HernquistPotential
import astropy.units as u
from galpy.util import conversion, save_pickles, coords, plot
import numpy as np
from galpy import potential
from scipy.interpolate import UnivariateSpline



def get_options():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    
                          
       
    #set timpact chunks                        
    parser.add_option("-t","--timpacts",dest='timpacts',default=None,
                      help="Impact times in Gyr to consider; should be a comma separated list")
    
    
    parser.add_option("--td",dest='td',default=4.,
                      type='float',
                      help="tdisrupt in Gyr")

    parser.add_option("--sigv",dest='sigv',default=0.46,
                      type='float',
                      help="velocity dispersion in km/s")

     parser.add_option("--isob",dest='isob',default=1.1,
                      type='float',
                      help="isochrone parameter")
                         
    parser.add_option("--leading_arm",action="store_false",dest='leading_arm',default=True,
                      help="leading or trailing arm")
                      
    parser.add_option("--plummer",action="store_false", 
                      dest="plummer",default=True,
                      help="If set, use a Plummer DM profile rather than Hernquist")
                   
    
    return parser

    
parser= get_options()
options,args= parser.parse_args()

########setup timpact chunks

ro=8.
vo=220.

print ("td=%.2f"%options.td)

def parse_times(times,age,ro=ro,vo=vo):
    if 'sampling' in times:
        nsam= int(times.split('sampling')[0])
        return [float(ti)/bovy_conversion.time_in_Gyr(vo,ro)
                for ti in numpy.arange(1,nsam+1)/(nsam+1.)*age]
    return [float(ti)/bovy_conversion.time_in_Gyr(vo,ro)
            for ti in times.split(',')]
            
timpacts= parse_times(options.timpacts,options.td,ro=ro,vo=vo)



if  options.leading_arm:
    lead_name='leading'
    lead=True
    
else :
    lead_name='trailing'
    lead=False
    
if options.plummer:
    hernquist=False
    sub_type = 'Plummer'
else :
    hernquist=True
    sub_type = 'Hernquist'


rgrid = np.array([1.0, 1.620253164556962, 2.240506329113924, 2.8607594936708862, 3.481012658227848,
         4.10126582278481, 4.7215189873417724, 5.341772151898734, 5.962025316455696,
         6.582278481012658, 7.20253164556962, 7.822784810126582, 8.443037974683545, 
         9.063291139240507, 9.683544303797468, 10.30379746835443, 10.924050632911392,
         11.544303797468354, 12.164556962025316, 12.784810126582277, 13.40506329113924,
         14.025316455696203, 14.645569620253164, 15.265822784810126, 15.886075949367088, 
         16.50632911392405, 17.126582278481013, 17.746835443037973, 18.367088607594937,
         18.987341772151897, 19.60759493670886, 20.22784810126582, 20.848101265822784,
         21.468354430379748, 22.088607594936708, 22.70886075949367, 23.32911392405063, 
         23.949367088607595, 24.569620253164555, 25.189873417721518, 25.81012658227848,
         26.430379746835442, 27.050632911392405, 27.670886075949365, 28.29113924050633,
         28.91139240506329, 29.531645569620252, 30.151898734177212, 30.772151898734176, 
         31.39240506329114, 32.0126582278481, 32.63291139240506, 33.25316455696203,
         33.87341772151898, 34.49367088607595, 35.11392405063291, 35.734177215189874,
         36.35443037974684, 36.974683544303794, 37.59493670886076, 38.21518987341772,
         38.835443037974684, 39.45569620253164, 40.075949367088604, 40.69620253164557,
         41.31645569620253, 41.936708860759495, 42.55696202531645, 43.177215189873415, 
         43.79746835443038, 44.41772151898734, 45.0379746835443, 45.65822784810126, 
         46.278481012658226, 46.89873417721519, 47.51898734177215, 48.13924050632911,
         48.75949367088607, 49.379746835443036, 50.0])

m_over_r2 = np.array([0.10738246142864227, 0.09845212732943764, 0.09336166768108434,
                      0.08909043318906584, 0.08541126902241353, 0.0821231540472492, 
                      0.07906372802021001, 0.07638001879341012, 0.07379467293207213,
                      0.07132184725453164, 0.0690528020866983, 0.06690969331050176,
                      0.0648670107637155, 0.0629247910403598, 0.06115523695726764,
                      0.05941708983712577, 0.057754700664653345, 0.05617594001966648,
                      0.05469924672857998, 0.05324612854579047, 0.051866263211996895,
                      0.050556075702075064, 0.04927898718290904, 0.04807714547558049,
                      0.04689822182988586, 0.04576537206434164, 0.044686880469953415,
                      0.043647862101065714, 0.0426583893189878, 0.041690096172332776,
                      0.04076868635081027, 0.039880074197205274, 0.0390214284048719,
                      0.03817897557750763, 0.03737766864972137, 0.03659427413601633,
                      0.03584532037860031, 0.03511260267079368, 0.03440948259375023,
                      0.03372475691490205, 0.0330657116452067, 0.03242999011161868, 
                      0.03180966477294754, 0.031213024451557377, 0.030633822361588837,
                      0.030068904748923927, 0.029520835943614137, 0.028986059103884285,
                      0.028471141214763417, 0.02796936206496345, 0.027480119457985147,
                      0.027003188825860284, 0.026538136719286596, 0.02608808711070303,
                      0.02565083608146509, 0.025225487238234128, 0.024809394517110173,
                      0.024402980472162064, 0.02401216077282636, 0.023629880776559024,
                      0.02325861072268433, 0.0228965611632571, 0.022542897807058954,
                      0.022195563146832187, 0.02185608348120908, 0.021526711218428585,
                      0.021205520148143853, 0.020895149702207842, 0.02058711389797605,
                      0.020287175412254357, 0.01999679784611772, 0.01971166196613089,
                      0.019433253292072836, 0.019161382756810105, 0.0188959346847113,
                      0.01863715186982597, 0.018381947205696338, 0.018134122517156973,
                      0.017891681778732335, 0.017654316711425783]) #10^10Msun/kpc^2


interp_m_over_r2 = UnivariateSpline(rgrid,m_over_r2) #10^10Msun/kpc^2

def rad_acc(r):
    acc = 1e10*interp_m_over_r2(r)*(conversion._G*0.001) #(km/s)^2/kpc
    acc*=-1.*(8./(220.*220.)) #in galpy units
    return (acc)

Phi0 = (conversion._G*0.001)*(1e10)*interp_m_over_r2(rgrid[0])*rgrid[0] #(km/s)^2
Phi0/=(220.*220.)

ip_halo= potential.interpSphericalPotential(rforce=rad_acc,rgrid=rgrid/8.,Phi0=Phi0)

disk_pot = MiyamotoNagaiPotential(amp=5.78*10**10*u.Msun,a=3.5*u.kpc,b=0.5*u.kpc)
disk_pot.turn_physical_off()

tot_pot = ip_halo + disk_pot
    
sdf_smooth=mockgd1_util.setup_gd1model(age=options.td,isob=options.isob,leading=lead,sigv=options.sigv)    


#if not os.path.exists(folder):
#    os.makedirs(folder,exist_ok=True)
        

pepperfilename= 'mockgd1_pepper_{}_{}_sigv{}_td{}_{}sampling_InterpSphPotAnalyticDisk.pkl'.format(lead_name,sub_type,options.sigv,options.td,len(timpacts))

sdf_pepper=mockgd1_util.setup_gd1model(timpact=timpacts,age=options.td,hernquist=hernquist,isob=options.isob,leading=lead,sigv=options.sigv)    

save_pickles(pepperfilename,sdf_smooth,sdf_pepper)
