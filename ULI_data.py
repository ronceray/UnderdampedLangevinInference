import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.linalg import LinAlgError


class StochasticTrajectoryData(object):
    """This class is a formatter and wrapper class for stochastic
    trajectories. It performs basic operations (discrete derivative,
    mid-point "Stratonovich" positions), and provides plotting
    functions for the trajectory and inferred force and diffusion
    fields (2D with matplotlib, 3D with mayavi2).

    """ 
    def __init__(self,X=None,t=None,V=None,trajectory_type="single"):
        """Two possible types of input:
        
        - "single" (default): A single time-lapse trajectory. X is a
          list of length Nsteps of Nparticles x d numpy arrays, where
          d is the dimensionality of phase space, Nsteps the length of
          the trajectory, and Nparticles the number of particles
          (assumed identical with constant number of particles.). Note
          that dt should be constant along the trajectory.

          If V=dX/dt is provided, it will be used; otherwise it will be
          computed using centered discrete differences.

        - "multiple": concatenate multiple "single" trajectories,
          possibly of different length or number of particles. This is
          useful when multiple short trajectories of the same system
          are available, or for multi-particles tracks with variable
          number of particles (which should be chopped down into small
          bits with constant number of particles). In this case X,t,
          and V are lists of the same length (V optional) of arrays as
          in 'single'. 

        Note that there is a real stuctural difference between the two
        types of data: 'single' has time x particles x dimension numpy
        arrays, while 'multiple' has time-lists of particles x
        dimension numpy arrays. The core inference method is made
        compatible with both, but the plotting routines are designed
        for the 'single' data structure.

        """
        if trajectory_type == 'single':
            Nparticles,self.d = X[0].shape
            self.t = t[1:-2]
            self.dt = t[2:-1] - t[1:-2] 
            self.X_ito = 1. * X[1:-2]
            self.X_smooth = 1./3  * ( X[:-3] + X[1:-2] + X[2:-1] )
            self.X_strat = 0.5  *  ( X[1:-2] + X[2:-1] )
            self.dX_plus  = X[3:] - X[2:-1]
            self.dX  = X[2:-1] - X[1:-2]
            self.dX_minus = X[1:-2] - X[:-3] 
            self.Nparticles = [ X.shape[0] for X in self.X_ito ]
            self.Ntrajectories = 1
            # Total time-per-particle in the trajectory
            self.tauN = np.einsum('t,t->',self.dt,self.Nparticles)
            if V is None:
                alpha = 0.5
                self.v_hat = np.einsum('tim,t->tim',alpha * self.dX - (alpha - 1.) * self.dX_minus, 1. /self.dt)
            else:
                self.v_hat = V[1:-2]
            self.a_hat = np.einsum('tim,t->tim',self.dX - self.dX_minus, 1. /self.dt**2)

        elif trajectory_type == 'multiple':
            # Concatenate multiple trajectories
            self.Ntrajectories = len(X)
            self.d = X[0].shape[2]
            self.t = None    # This argument becomes invalid: no
                             # absolute time (only used for plotting
                             # anyways)
            self.dt,self.X_ito,self.X_smooth,self.X_strat,self.dX_plus,self.dX,self.dX_minus,\
                self.Nparticles,self.v_hat,self.a_hat = [],[],[],[],[],[],[],[],[],[]
            self.tauN = 0.
            for i in range(self.Ntrajectories):
                data_single = StochasticTrajectoryData(X=X[i],t=t[i],V=( None if V is None else V[i]) ,trajectory_type="single")
                self.dt       += list(data_single.dt) 
                self.X_ito    += list(data_single.X_ito) 
                self.X_smooth += list(data_single.X_smooth)
                self.X_strat  += list(data_single.X_strat)
                self.dX_plus  += list(data_single.dX_plus)
                self.dX       += list(data_single.dX)
                self.dX_minus += list(data_single.dX_minus)
                self.Nparticles += list(data_single.Nparticles)
                self.v_hat    += list(data_single.v_hat)
                self.a_hat    += list(data_single.a_hat)
                self.tauN += data_single.tauN
            
                if self.d != data_single.d:
                    raise ValueError("Dataset has non-homogeneous dimension:",self.d,data_single.d)
            self.dt = np.array(self.dt)

    def inner_product_empirical(self,f,g,integration_style='Ito',args=None):
        # Time integration of two functions (with indices) of the
        # position along the trajectory, with averaging over particle
        # indices.

        # Two ways of calling it (can be mixed):
        #  - with an array (t-1) x Nparticles x dim (eg for Xdot)
        
        #  - with a dxN-dimensional callable function f(X) (eg for the
        #    moments); will apply it on all time-points and particles,
        #    with corresponding arguments.
        if integration_style=='Ito':
            X = self.X_ito
            V = self.v_hat
            dt = self.dt
        elif integration_style=='Smooth':
            X = self.X_smooth
            V = self.v_hat
            dt = self.dt
        elif integration_style=='Dnoisy':
            X = self.X_strat + 0.25*(self.dX_plus-self.dX_minus)
            V = [  (4 * self.dX[t] + 1 * (self.dX_plus[t]+self.dX_minus[t])) / (6 * self.dt[t]) for t in range(len(self.dX)) ]
            dt = self.dt
        else:
            raise KeyError("Wrong integration_style keyword.")
        func = lambda t : np.einsum('im,in->mn', f(X[t],V[t]) if callable(f) else f[t] , g(X[t],V[t]) if callable(g) else g[t] )
        return self.trajectory_integral(func,dt)

    def trajectory_integral(self,func,dt=None):
        # A helper function to perform trajectory integrals with
        # constant memory use. The function takes integers as
        # argument.
        if dt is None:
            dt = self.dt
        result = 0. # Generic initialization - works with any
                    # array-like function.
        for t,ddt in enumerate(dt):
            result += ddt * func(t)
        return result / self.tauN

 
    def plot_phase_space_forces(self,field,N = 10,scale=1.,autoscale=False,color='g',radius=None,positions = None,**kwargs):
        """Plot the force field in the (x,vx) space for a 1D process
        """

        x_range = 1.1 * (self.X_ito[:,0,0].max() - self.X_ito[:,0,0].min())
        x_center = 0.5 * (self.X_ito[:,0,0].max() + self.X_ito[:,0,0].min())

        v_range = 1.1 * abs(self.v_hat[:,0,0]).max() 
        v_center = 0.

        positions = []
        gridX,gridV = [],[]
        Fx,Fv = [],[]
        for x in np.linspace(x_center - 0.5*x_range,x_center + 0.5*x_range,N):
            for v in np.linspace(v_center - 0.5*v_range,v_center + 0.5*v_range,N):
                gridX.append(x)
                gridV.append(v)
                Fx.append(v)
                Fv.append(float(field(np.array([[x]]),np.array([[v]]))))

        plt.quiver(gridX,gridV,scale*np.array(Fx),scale*np.array(Fv) ,scale = 1.0,units = 'xy',color = color,minlength=0.,**kwargs)


def axisvector(index,dim):
    """d-dimensional vector pointing in direction index."""
    return np.array([ 1. if i == index else 0. for i in range(dim)])
 


def data_multiparticle_parser(time_index,particle_index,X,dt,max_dx = 1e100):
    """A helper class to initialize StochasticTrajectoryData with typical
    particle tracking data. dt is a scalar (constant between time
    increments); time and particle indices are integer arrays, and X
    is an array with the same length as the index arrays. 

    max_dx is an optional argument to filter out NaN's and tracking
    identity errors.

    Note that this is written conservatively and works well only for
    datasets with very few NaN's / particle loss: each one wastes a
    full 4 time steps for all particles (could be improved).

    """

    Npts = len(time_index)
    tn_to_x = {(time_index[i],particle_index[i]) : X[i] for i in range(Npts) }
    # t - > list of n
    tvals = sorted(set(time_index))
    t_to_nlist = { t : []  for t in tvals }
    for ind,n in enumerate(particle_index):
        t_to_nlist[time_index[ind]].append(n)
        
    t_to_tnlist = [ (t,tuple(sorted(t_to_nlist[t]))) for t in tvals ] 
    
    # break into bits with same n set and no jump larger than dx
    datasets = [  ]
    tsets = []
    t_prev,n_prev = t_to_tnlist[0]
    current_set = [ (t_prev,n_prev) ]
    for (t,nlist) in t_to_tnlist[1:]+[(None,None)]:
        if nlist == n_prev and t == t_prev+1 and all(np.linalg.norm(tn_to_x[(t,n)] - tn_to_x[(t_prev,n)]) < max_dx for n in nlist):
            current_set.append((t,nlist))
        else:
            # Break the series
            if len(current_set) >= 4:
                datasets.append( np.array([[ tn_to_x[(t,n)] for n in nlist ] for (t,nlist) in current_set ]) )
                tsets.append( np.array([ t * dt for (t,nlist) in current_set ]) )
            current_set = []
        (t_prev,n_prev) = (t,nlist)
    return StochasticTrajectoryData( X=datasets, t=tsets, V=None, trajectory_type="multiple")
    
    
