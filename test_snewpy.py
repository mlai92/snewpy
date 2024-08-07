import astropy.units as u
from snewpy.models.ccsn import Nakazato_2013, Bollig_2016
from snewpy.flavor_transformation import AdiabaticMSW
from snewpy.neutrino import MassHierarchy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from snewpy.flux import Container
from snewpy.neutrino import Flavor

import pylab as plt
from contextlib import contextmanager


def plot_quantity(x:u.Quantity, y:u.Quantity, xlabel=None, ylabel=None, **kwargs):
    """Plot the X vs Y array, with given axis labels, adding units"""

    #just in case we are passed bare np.arrays iwthout units
    x = u.Quantity(x)
    y = u.Quantity(y)
    
    if(len(x)==len(y)):
        plt.plot(x.value,y.value,**kwargs)
    else:
        plt.stairs(edges=x.value,values=y.value,**kwargs)
    if xlabel is not None:
        if(not x.unit.is_unity()):
            xlabel+=', '+x.unit._repr_latex_()
        plt.xlabel(xlabel)
    if ylabel is not None:
        if(not y.unit.is_unity()):
            ylabel+=', '+y.unit._repr_latex_()
        plt.ylabel(ylabel)


from snewpy.flux import Container
def sum_rates(rates:list):
    res = sum([rate.array for rate in rates])
    rate = rates[0]#take first as an instance
    return Container(res,rate.flavor, rate.time, rate.energy, integrable_axes=rate._integrable_axes)

#Utility function to draw the flux
from snewpy.flux import Axes

def project(flux, axis, integrate=True):
    axis = Axes[axis] #convert to enum
    integrate_axis = Axes.time 
    if axis == integrate_axis:
        integrate_axis = Axes.energy 
    fI = (flux.integrate if integrate else flux.sum)(integrate_axis)
    return fI.axes[axis], fI
    
def plot_projection(flux, axis, step=False, integrate=True):
    x,fI = project(flux,axis, integrate)
    y = fI.array.squeeze().T
    if step:
        #we're dealing with bins, not points
        l = plt.step(x[:-1], y, where='pre', label=[Flavor(flv).to_tex() for flv in flux.flavor])
    else:
        l = plt.plot(x, y, label=[Flavor(flv).to_tex() for flv in flux.flavor])
    
    plt.ylabel(f'{fI.__class__.__name__},  {y.unit}')
    plt.xlabel(f'{Axes[axis].name},  {x.unit}')
    return l



# Initialise two SN models. This automatically downloads the required data files if necessary.
nakazato = Nakazato_2013(progenitor_mass=20*u.solMass, revival_time=100*u.ms, metallicity=0.004, eos='shen')
bollig = Bollig_2016(progenitor_mass=27*u.solMass)

print("Nakazato")
print(nakazato)


times    = nakazato.time #np.linspace(0,2,1500)<<u.second;
energies = np.linspace(0,50,501)<<u.MeV

flux = nakazato.get_flux(t = times, E = energies, distance=10<<u.kpc)


# Adiabatic MSW flavor transformation with normal mass ordering
msw_nmo = AdiabaticMSW(mh=MassHierarchy.NORMAL)

# Assume a SN at the fiducial distance of 10 kpc and normal mass ordering.

#flux = bollig.get_flux(distance=10*u.kpc, t=t_list*u.s, E= E_list*u.erg, flavor_xform=msw_nmo)
flux_t = nakazato.get_flux(t = times, E = energies, distance=10<<u.kpc, flavor_xform=msw_nmo)
#print(msw_nmo)

#plot the neutrino flux 
fig,ax = plt.subplots(2,2, figsize=(12,6))
plt.sca(ax[0,0])
plot_projection(flux, 'energy', integrate=True)
plt.legend()

plt.sca(ax[0,1])
plot_projection(flux, 'time', integrate=True)
plt.legend()
plt.xscale('log')

plt.sca(ax[1,0])
plot_projection(flux_t, 'energy', integrate=True)

plt.legend()

plt.sca(ax[1,1])
plot_projection(flux_t, 'time', integrate=True)
plt.legend()
plt.xscale('log')

plt.show()

from snewpy.rate_calculator import RateCalculator
rc = RateCalculator()

#calculate time differential rate 
rates = rc.run(flux, 'argo')


for ch, rate in rates.items():
    l = plot_projection(rate, 'time', integrate=False)
    l[0].set_label(ch)
plt.yscale('log')
plt.xscale('log')
plt.legend(loc='right')
plt.ylim(0.1)
plt.show()

