import numpy as np

"""UnderdampedLangevinInference is a package developed by Pierre
Ronceray and David Brückner which performs inference on trajectories
of underdamped stochastic processes.  Compatible Python 3.6 / 2.7.

Reference: 
    David Brückner, Pierre Ronceray and Chase Broedersz
    Inferring the dynamics of underdamped stochastic systems

This is a demo on the example of the stochastic Van der Pol oscillator.

"""

# Import the package:
from UnderdampedLangevinInference import *
 
################################################################
# I. Prepare the data (here using a simulated model).

# Force field parameters (VdP oscillator)
dim = 1
initial_position = np.array([[0.1]]) 
mu = 2
force = lambda X,V : mu * (1-X**2)*V - X

# Diffusion parameters (constant noise)
diffusion = 1. * np.identity(dim) 


# Simulation parameters
dt = 0.02
oversampling = 10
prerun = 100
Npts = 1000
tau = dt * Npts
tlist = np.linspace(0.,tau,Npts)

# Run the simulation using our OverdampedLangevinProcess class
np.random.seed(1)
X = UnderdampedLangevinProcess(force,diffusion,tlist,initial_position=initial_position,oversampling=oversampling,prerun=prerun)

# Add artificial measurement noise:
noise_amplitude = 0.002
noise = noise_amplitude * np.random.normal(size=X.data.shape)

# The input of the inference method is the "xlist" array, which has
# shape Nsteps x 1 x dim (the middle index is used for multiple
# particles with identical properties; we do not use it in this demo).
# You can replace xlist and tlist with your own data!
xlist = X.data + noise
tlist = X.t

# We use a wrapper class, StochasticTrajectoryData, to format the data
# in a way that the inference methods can use.
data = StochasticTrajectoryData(xlist,tlist) 
data_exact = StochasticTrajectoryData(X.data,X.t,V=X.data_v) 


################################################################
# II. Perform inference.

S = UnderdampedLangevinInference( data )

S.compute_current(basis = { 'type' : 'polynomial', 'order' : 3} ) 
S.compute_diffusion(method='noisy') 
S.compute_force()
S.compute_force_error() 

S.compare_to_exact(data_exact=data_exact, force_exact = force, D_exact = diffusion)
S.print_report()

################################################################
# III. Plot the results and compare to exact fields.

# Prepare Matplotlib:
import matplotlib.pyplot as plt
fig_size = [12,8]
params = {'axes.labelsize': 12,
          'font.size':   14,
          'legend.fontsize': 10,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': True,
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)
plt.clf()
fig = plt.figure(1)
fig.subplots_adjust(left=0.06, bottom=0.07, right=0.96, top=0.94, wspace=0.35, hspace=0.3)
H,W = 2,3

pic_name = "ULI_demo_VdP.png"

# Plot the trajectory (xv phase space):
plt.subplot(H,W,1)
plt.plot(data.X_ito[:,0,:],data.v_hat[:,0,:],color='b')
plt.xlabel(r"$x$",labelpad=0)
plt.ylabel(r"$v$",labelpad=0)
plt.title("Input data")

# Plot the trajectory (x vs t):
plt.subplot(H,W,4)
plt.plot(data.t,data.X_ito[:,0,:],color='b')
plt.ylabel(r"$x(t)$")
plt.xlabel(r"$t$")
plt.title("Input data")

# Plot the force field - blue is inferred, black is the exact one used to generate the data.
plt.subplot(H,W,2)
S.data.plot_phase_space_forces(S.F_ansatz,color='b',alpha=0.4,zorder=0,width = 0.1,scale=0.2)
S.data.plot_phase_space_forces(force,color='k',alpha=1,zorder=-1,width=0.05,scale=0.2)
plt.xlabel(r"$x$",labelpad=0)
plt.ylabel(r"$v$",labelpad=0)
plt.xlim([-2,2])
plt.title("Flow field: black=exact,blue=inferred")

#Performance of force field inference
plt.subplot(H,W,5)
plt.title("Performance of force inference")
ULI_plotting_toolkit.comparison_scatter(S.exact_F,S.ansatz_F,y=0.8,alpha=1)

plt.xlabel(r"exact $F(x,v)$",labelpad=-1)
plt.ylabel(r"inferred $F(x,v)$",labelpad=0)

# Use the inferred force and diffusion fields to simulate a new
# trajectory with the same times list, and plot it.
Y = S.simulate_bootstrapped_trajectory(oversampling=20)
data_bootstrap = StochasticTrajectoryData(Y.data,Y.t)

plt.subplot(H,W,3)
plt.title("Predicted trajectory")
plt.plot(data_bootstrap.X_ito[:,0,:],data_bootstrap.v_hat[:,0,:],color='orange')

plt.xlabel(r"$x$",labelpad=0)
plt.ylabel(r"$v$",labelpad=0)

plt.subplot(H,W,6)
plt.title("Predicted trajectory")
plt.plot(data_bootstrap.t,data_bootstrap.X_ito[:,0,:],color='orange')

plt.ylabel(r"$x(t)$")
plt.xlabel(r"$t$")


plt.tight_layout()
plt.show()
plt.savefig(pic_name)


