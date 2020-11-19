from galpy.df import streamdf, streamgapdf
from streampepperdf_new import streampepperdf
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.util import conversion #for unit conversions


R0, V0= 8., 220.
def setup_mockgd1model(leading=True,pot=MWPotential2014,
                   timpact=None,Zsun=0.025,
                   hernquist=True,isob=0.8,
                   age=9.,sigv=0.46,
                   singleImpact=False,
                   length_factor=1.,
                   **kwargs):
    
    aAI= actionAngleIsochroneApprox(pot=pot,b=isob)
    obs= Orbit.from_name("GD1")
        
    if timpact is None:
        sdf= streamdf(sigv/220.,progenitor=obs,pot=pot,aA=aAI,leading=leading,
                      nTrackChunks=11,vsun=[-11.1,244.,7.25],Zsun=Zsun,
                      tdisrupt=age/conversion.time_in_Gyr(V0,R0),
                      vo=V0,ro=R0)
    elif singleImpact:
        sdf= streamgapdf(sigv/220.,progenitor=obs,pot=pot,aA=aAI,
                         leading=leading,
                         nTrackChunks=11,vsun=[-11.1,244.,7.25],Zsun=Zsun,
                         tdisrupt=age/conversion.time_in_Gyr(V0,R0),
                         vo=V0,ro=R0,
                         timpact=timpact,
                         spline_order=3,
                         hernquist=hernquist,**kwargs)
    else:
        sdf= streampepperdf(sigv/220.,progenitor=obs,pot=pot,aA=aAI,
                            leading=leading,
                            nTrackChunks=101,vsun=[-11.1,244.,7.25],Zsun=Zsun,
                            tdisrupt=age/conversion.time_in_Gyr(V0,R0),
                            vo=V0,ro=R0,
                            timpact=timpact,
                            spline_order=1,
                            hernquist=hernquist,
                            length_factor=length_factor)
    sdf.turn_physical_off()  #original
    #obs.turn_physical_off()
    return sdf
