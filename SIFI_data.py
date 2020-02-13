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
    def __init__(self,X,t,V=None):
        """X is a list of length Nsteps of Nparticles x d numpy arrays, where
        d is the dimensionality of phase space, Nsteps the length of
        the trajectory, and Nparticles the number of particles
        (assumed identical).

        If V=dX/dt is provided, it will be used; otherwise it will be
        computed using centered discrete differences.

        """
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
        # Total time-per-particle in the trajectory
        self.tauN = np.einsum('t,t->',self.dt,self.Nparticles)

        self.alpha = 0.5
        if V is None:
            self.v_hat = np.einsum('tim,t->tim',self.alpha * self.dX - (self.alpha - 1.) * self.dX_minus, 1. /self.dt)
        else:
            self.v_hat = V
            
        self.a_hat = np.einsum('tim,t->tim',self.dX - self.dX_minus, 1. /self.dt**2) 


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
            V = np.einsum('tim,t->tim',( 4 * self.dX+ 1 * (self.dX_plus+self.dX_minus)), 1 / (6 * self.dt))
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
 

