import numpy as np

"""UnderdampedLangevinInference is a package developed by Pierre
Ronceray and David Brückner which performs inference on trajectories
of underdamped stochastic processes.  Compatible Python 3.6 / 2.7.

Reference: 
    David Brückner, Pierre Ronceray and Chase Broedersz
    Inferring the dynamics of underdamped stochastic systems

This is a demo on using experimental data of interacting cell pairs from this reference:
    D. B. Brückner,  N. Arlt, A. Fink, P. Ronceray, J. O. Rädler, C. P. Broedersz
    Learning the dynamics of cell-cell interactions in confined cell migration 
    PNAS 118:e2016602118, 2021 
"""

# Import the package:
from UnderdampedLangevinInference import *
import matplotlib.pyplot as plt
 
################################################################
# I. Prepare the data 

#read in Dataset_S01 from https://www.pnas.org/doi/10.1073/pnas.2016602118#supplementary-materials
trajectories = np.loadtxt('pnas.2016602118.sd01.txt')
Ntrajectories = int(trajectories.shape[0]/2)

delta_t_expt = 1/6 #in hours
factor_divide = 50 #in microns

X = []
X_divided = []
T = []
for j_system in range(0,Ntrajectories-1):
    xlist_system = trajectories[j_system*2:j_system*2+2,:]
    
    #plot an example trajectory
    if j_system==0:
        plt.figure()
        plt.plot(xlist_system[0,:])
        plt.plot(xlist_system[1,:])

    xlist_system_divided = xlist_system/factor_divide #normalize data to avoid large numbers
    x_list_nonan = np.array([x for x in xlist_system[0,:] if not np.isnan(x)])
    last_entry = len(x_list_nonan)
    
    #ordering: time, particles, dimension 
    xlist = np.einsum('djt->tjd',np.array([xlist_system[:,:last_entry]]))
    xlist_divided = np.einsum('djt->tjd',np.array([xlist_system_divided[:,:last_entry]]))
    tlist = np.linspace(0,last_entry-1,last_entry)*delta_t_expt

    X.append(xlist)
    X_divided.append(xlist_divided)
    T.append(tlist)


# We use a wrapper class, StochasticTrajectoryData, to format the data
# in a way that the inference methods can use.
data = StochasticTrajectoryData(X,T,trajectory_type='multiple') 
data_divided = StochasticTrajectoryData(X_divided,T,trajectory_type='multiple') 



################################################################
# II. Perform inference.

S = UnderdampedLangevinInference( data_divided )

dim=1
order_single=3
Rmax = 0.4 #maximum range of largest kernel
Npts = 3 #number of kernels
rmin = Rmax/Npts

import ULI_bases

center_x=0.
width_x=2
order_x=3
order_v=3
functions_single = ULI_bases.fourierX_polyV_basis(dim,center_x,width_x,order_x,order_v)

def exponential_kernel(r0):
    return lambda r : np.exp(-r/r0)
kernels = [ exponential_kernel(r0) for r0 in np.linspace(rmin,Rmax,Npts) ]

S.compute_current(basis={ 'type' : 'twocell_alignment_cohesion', 'functions_single' : functions_single, 'kernels_cohesion' : kernels, 'kernels_alignment' : kernels}) 
print('current done.')

S.compute_diffusion(method='noisy') 
print('diffusion done.')

S.compute_force()
S.compute_force_error() 

S.print_report()

# Prepare Matplotlib:
plt.close('all')
fig_size = [12,3]
params = {'axes.labelsize': 10,
          'font.size':   14,
          'legend.fontsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)
plt.clf()

import plotting_scripts_twocells
plt.figure()
H,W = 1,3

N_bins = 50
plt.subplot(H,W,1)
plotting_scripts_twocells.plot_Fxv(S.F_ansatz,N_bins,factor_divide)

plt.subplot(H,W,2)
plotting_scripts_twocells.plot_cohesion(kernels,kernels,S.F_coefficients,N_bins,factor_divide)

plt.subplot(H,W,3)
plotting_scripts_twocells.plot_friction(kernels,S.F_coefficients,N_bins,factor_divide)

plt.tight_layout()
