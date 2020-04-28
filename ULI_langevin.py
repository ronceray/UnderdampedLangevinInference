
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm


class UnderdampedLangevinProcess(object):
    """A simple class to simulate underdamped Langevin processes.

    dX/dt = V(t)
    dV/dt = F(X,V) + \sqrt{2 D} xi

    The resulting simulated data is a 3D array:
    - 1st index is time index
    - 2nd is an index for multiple independent copies of the process,
      or, in the case of interacting particles systems, a particle
      index. Note that we haven't implemented multi-particles
      multi-copies simulations: independent copies are considered as
      non-interacting particles. 
    - 3rd is spatial/phase space index

    The optional argument "oversampling" allows to simulate
    intermediate points but only record a fraction of them, in cases
    where the dynamics is sensitive to the integration timestep.

    The optional argument "prerun" allows to simulate an initial
    equilibration phase (number of time steps, using the first time
    interval for dt).

    """
    
    def __init__(self, F, D, tlist, initial_position, initial_velocity = 0., oversampling = 1, prerun = 0):
        self.F = F
        self.Nparticles,self.d = initial_position.shape
        self.oversampling = oversampling
        self.prerun = prerun

        if hasattr(D,'shape'): # Constant diffusion tensor provided as an array
            Dc = 1.*D
            if Dc.shape == (self.d,self.d):
                Dc = np.array([ Dc for i in range(self.Nparticles)])
            elif Dc.shape != (self.Nparticles,self.d,self.d):
                raise ValueError("Diffusion matrix shape is wrong:",Dc.shape," while dim is:",self.d)
            self.__D__ = Dc
            self.D = lambda X,V : self.__D__
            self.__sqrt2D__ = np.array([ sqrtm(2 * self.__D__[i,:,:]) for i in range(self.Nparticles) ])
            self.sqrt2D = lambda X,V : self.__sqrt2D__
            self.divD = lambda X,V : np.zeros((self.Nparticles,self.d))
        else: 
            self.D = D
            if self.d >= 2:
                def sqrt2D(X,V):
                    D = self.D(X,V)
                    return np.array([ sqrtm(2 * D[i,:,:]) for i in range(self.Nparticles) ])
            else:
                def sqrt2D(X,V):
                    return (2*self.D(X,V))**0.5
            self.sqrt2D = sqrt2D

        self.t = tlist

        self.Ntimesteps = len(self.t) 
        self.dt = self.t[1:] - self.t[:-1]
        self.simulate(initial_position,oversampling,prerun,initial_velocity)

        
    def dv(self,x,v,dt):
        """ The velocity increment in time dt."""
        return dt * self.F(x,v)   +  np.einsum('imn,in->im',self.sqrt2D(x,v), np.random.normal(size=(self.Nparticles,self.d)) ) * dt**0.5

    
        
    def simulate(self,initial_position,oversampling,prerun,initial_velocity):
        # pre-equilibration:
        dt = self.dt[0] / oversampling
        x = 1. * initial_position
        x_prev =  x - initial_velocity * dt
        v = (x - x_prev )/dt
        for j in range( prerun*oversampling ):
            v = (x - x_prev )/dt
            dv = self.dv(x,v,dt)
            # Verlet integration
            x_prev,x = x, 2*x - x_prev + dv * dt
            
        # Start recording:
        self.data = np.zeros((self.Ntimesteps,self.Nparticles,self.d))
        self.data_v = np.zeros((self.Ntimesteps,self.Nparticles,self.d))
        for i,delta_t in enumerate(self.dt):
            self.data[i,:,:] = x
            self.data_v[i,:,:] = v 
            dt = delta_t / oversampling
            for j in range( oversampling ):
                v = (x - x_prev )/dt
                dv = self.dv(x,v,dt)
                # Verlet integration
                x_prev,x = x, 2*x - x_prev + dv * dt 


        self.data[-1,:,:] = x


class ParticlesUnderdampedLangevinProcess(UnderdampedLangevinProcess):
    """ Simulate overdamped Langevin processes with pair
    interactions between identical particles.
    """
    
    def __init__(self, force_single, force_pair, D, tlist, initial_position, initial_velocity=0., oversampling = 1, prerun = 0):
        self.Fsingle = force_single
        self.Fpair = force_pair
        self.D = D
        self.t = tlist
        Nparticles,dim = initial_position.shape
        def force(X,V):
            return np.array([ self.Fsingle(X[i,:],V[i,:]) for i in range(Nparticles) ]) + \
                   np.array([ np.sum(np.array([ self.Fpair(X[i,:],X[j,:],V[i,:],V[j,:]) for j in range(Nparticles) if i != j ]),axis=0) for i in range(Nparticles) ])
        #np.array([ np.sum(np.array([ self.Fpair(X[i,:],X[j,:]) for j in range(Nparticles) if i != j ]),axis=0) for i in range(Nparticles) ])

        UnderdampedLangevinProcess.__init__(self,force, D, tlist, initial_position, initial_velocity=initial_velocity, oversampling=oversampling,prerun=prerun)





