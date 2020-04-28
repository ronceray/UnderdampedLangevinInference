"""UnderdampedLangevinInference is a package developed by Pierre
Ronceray and David Brueckner which performs inference on trajectories
of stochastic inertial processes.  Compatible Python 3.6 / 2.7.

Reference: 
    David Brueckner, Pierre Ronceray and Chase Broedersz
    Inferring the dynamics of underdamped stochastic systems

"""

from ULI_projectors import TrajectoryProjectors


import numpy as np


class UnderdampedLangevinInference(object): 
    """This class performs the inference of force and diffusion,
    assembles them together, and provides error estimates. It is
    initialized using a StochasticTrajectoryData instance that formats
    the data on which to perform the inference (see the corresponding
    module).  

    The model on which the data is fitted is:

    dx/dt = v
    dv/dt = F(x,v) + sqrt(2D(x,v)) d_xi(t)

    written in the Ito convention. Its main routines are (more details
    in the preamble of each one):
    
    - compute_diffusion: infer D - either as a constant tensor or as a
      linear combination of basis functions.

    - compute_force: infer F(x) as a linear combination of basis
      functions.

    - compute_force_error, compute_diffusion_error: predict the
      normalized mean-squared error (MSE) expected on the force and
      diffusion respectively, due to finite trajectory length and to
      time discretization.

    """
    
    
    def __init__(self,data):
        self.data = data 

    def compute_diffusion(self,basis=None,method='noisy'): 
        self.diffusion_method = method
        # Select the (noisy) local diffusion matrix estimator:
        if self.diffusion_method == 'discrete':
            # For ideal data - slightly faster convergence 
            D_local = self.__D_discrete__
            integration_style = 'Smooth'
        elif self.diffusion_method == 'noisy':
            # For data with measurement noise: corrects the bias
            D_local = self.__D_noisy__
            integration_style = 'Dnoisy'
        else:
            raise KeyError("Wrong diffusion_method argument:",diffusion_method)

        self.D_average = np.einsum('t,tmn->mn',self.data.dt,np.array([ np.einsum('imn->mn', D_local(t) ) for t in range(len(self.data.dt)) ]))/self.data.tauN
        self.D_average_inv = np.linalg.inv(self.D_average)

        self.Lambda = np.einsum('t,tmn->mn',self.data.dt,np.array([ np.einsum('imn->mn', self.__Lambda__(t) ) for t in range(len(self.data.dt)) ]))/self.data.tauN

        
        if basis is not None:
            # Case of a state-dependent diffusion coefficient: fit the
            # local estimator with the basis functions.
            self.diffusion_basis = basis 
            
            # Select a projection basis
            import ULI_bases
            if "functions" in self.diffusion_basis:
                funcs = self.diffusion_basis["functions"]
            else:
                funcs,is_interacting = ULI_bases.basis_selector(self.diffusion_basis,self.data)

            # Prepare the functions:
            self.diffusion_projectors = TrajectoryProjectors(self.data.inner_product_empirical,funcs)
            # Reshape into vectors as inner product allows for only one 
            # non-particle index:
            D_local_reshaped = [ np.array([ flatten_symmetric(Di,self.data.d) for Di in D_local(t) ]) for t in range(len(self.data.dt))  ]
            D_projections_reshaped = np.einsum('ma,ab->mb',self.data.inner_product_empirical(D_local_reshaped, self.diffusion_projectors.b, integration_style = integration_style), self.diffusion_projectors.H )
            # Back to matrix form:
            self.D_projections = np.array([ inflate_symmetric(Di,self.data.d) for Di in D_projections_reshaped.T]).T 
            self.D_ansatz,self.D_coefficients = self.diffusion_projectors.projector_combination(self.D_projections) 
            self.D_inv_ansatz = lambda X,V : np.linalg.inv(self.D_ansatz(X,V)) 

        
    def compute_current(self,basis): 
        self.force_basis = basis
         # Select a projection basis - either by specifying it
        # explicitly with the 'functions' keyword of the 'basis'
        # argument, or by parsing it among the pre-defined bases
        # defined in ULI_bases.
        import ULI_bases
        if "functions" in self.force_basis:
            funcs = self.force_basis["functions"]
        else:
            funcs,is_interacting = ULI_bases.basis_selector(self.force_basis,self.data)
            
        # Prepare the functions:
        self.force_projectors = TrajectoryProjectors(self.data.inner_product_empirical,funcs)   

        # Compute the projection coefficients. Indices m,n... denote
        # spatial indices; indices i,j.. denote particle indices;
        # indices a,b,... are used for the tensorial structure of the
        # projections.

        # The time-irreversible part of the force is obtained as a
        # Stratonovich average.
        self.phi_projections = np.einsum('ma,ab->mb',self.data.inner_product_empirical( self.data.a_hat, self.force_projectors.b, integration_style = 'Smooth' ), self.force_projectors.H )
        self.phi_ansatz,self.phi_coefficients = self.force_projectors.projector_combination(self.phi_projections)

    def compute_force(self): 
        # The time-reversible part is obtained using the gradient of
        # the fitting function.
        if hasattr(self,'D_ansatz'):
            self.w_projections = - np.einsum('ma,ab->mb',self.data.trajectory_integral(lambda t : np.einsum('imn,inia->ma',self.D_ansatz(self.data.X_smooth[t],self.data.v_hat[t]),self.force_projectors.grad_v_b(self.data.X_smooth[t],self.data.v_hat[t]))), self.force_projectors.H )

        elif hasattr(self,'D_average'):
            print("Computing the force assuming constant diffusion.")
            self.g_projections = - np.einsum('ma,ab->mb',self.data.trajectory_integral(lambda t : np.einsum('imia->ma',self.force_projectors.grad_v_b(self.data.X_smooth[t],self.data.v_hat[t]))), self.force_projectors.H )
            self.w_projections = np.einsum('mn,na->ma',self.D_average,self.g_projections)
        else:
            raise ValueError("Need to provide or compute diffusion in order to compute the force.")

        self.w_ansatz,self.w_coefficients = self.force_projectors.projector_combination(self.w_projections)

        # Assemble both parts into a force ansatz.
        self.F_projections = self.w_projections+self.phi_projections
        self.F_ansatz,self.F_coefficients = self.force_projectors.projector_combination(self.F_projections)
        

    def compute_force_error(self,maxpoints=100):
        indices = np.array([ i for i in range(0,len(self.data.X_ito),len(self.data.X_ito)//maxpoints + 1) ])
        tauN_sample = sum( self.data.dt[t] * self.data.Nparticles[t] for t in indices )

        if hasattr(self,'D_ansatz'):
            print("Computing error assuming state-dependent diffusion using subsampling:",maxpoints/len(self.data.X_ito))
            Dinv_Ito_dt   = [ self.D_inv_ansatz(self.data.X_ito[i],self.data.v_hat[i]) * self.data.dt[i] for i in indices ] 
            ansatz_F_Ito = [ self.F_ansatz(self.data.X_ito[i],self.data.v_hat[i]) for i in indices ]
            self.force_information = 0.25 * sum([ np.einsum('imn,im,in->',Dinv_Ito_dt[t],ansatz_F_Ito[t],ansatz_F_Ito[t]) for t in range(len(indices))]) * ( self.data.tauN / tauN_sample )

        else:
            print("Computing error assuming constant diffusion.")
            if not hasattr(self,'D_average'):
                self.compute_diffusion(basis=None)
            self.force_information = 0.25 * np.einsum('ma,na,mn->',self.F_projections ,self.F_projections, self.D_average_inv) * self.data.tauN 

        self.error_force_information = ( 2 * self.force_information + np.prod(self.F_projections.shape)**2 / 4 ) ** 0.5
        
        # Squared typical error due to trajectory length
        self.force_projection_error = 0.5 * np.prod(self.F_projections.shape) / self.force_information
        print("Expected mean squared error:",self.force_projection_error)

        # Compute the partial information vs the number of fitting
        # functions
        self.partial_information = 0.25 * np.einsum('ma,na,mn->a',self.F_projections ,self.F_projections, self.D_average_inv)*self.data.tauN
        self.cumulative_information = [ self.partial_information[0]]
        for i in self.partial_information[1:]:
            self.cumulative_information.append(i+self.cumulative_information[-1])
        self.cumulative_information = np.array(self.cumulative_information)
        self.cumulative_error = np.array([ ( 2*I + (self.data.d*(n+1))**2/4 )**0.5 for n,I in enumerate( self.cumulative_information ) ])
        self.cumulative_bias = np.array([ self.data.d * n/4 for n,I in enumerate(self.cumulative_information)])
 
    def print_report(self):
        """ Tell us a bit about yourself.
        """
        print("             ")
        print("  --- StochasticForceInference report --- ")
        print("Average diffusion tensor:\n",self.D_average)
        if hasattr(self,'Lambda'):
            print("Measurement noise covariance:\n",self.Lambda)
        if hasattr(self,'force_projections_self_consistent_error'):
            print("Force information: inferred/bootstrapped error",self.force_information,self.error_force_information)
            print("Force: squared typical error on projections:",self.force_projections_error)

 
    # "Provide" external values for some parameters will bypass their
    # inference.
    def provide_diffusion(self,D,D_is_constant=True):
        if D_is_constant:
            self.D_average = D
            self.D_average_inv = np.linalg.inv(D)
        else:
            self.D_ansatz = D
            self.D_inv_ansatz = lambda X,V : np.linalg.inv(self.D_ansatz(X,V))
            self.D_average = np.einsum('t,tmn->mn',self.data.dt,np.array([ np.einsum('imn->mn', D(self.data.X_ito[t],self.data.v_hat[t]) ) for t in range(len(self.data.dt)) ]))/self.data.tauN
            self.D_average_inv = np.linalg.inv(self.D_average)

    def compare_to_exact(self,data_exact=None,force_exact=None,D_exact=None,maxpoints = 1000):
        if data_exact is None:
            # In the case of noisy input data, we want to compare to
            # the force inferred on the real trajectory, not the noisy
            # one (which would use values of the force field that
            # weren't explored by the trajectory, and thus cannot be
            # predicted).
            self.data_exact = self.data
        else:
            self.data_exact = data_exact
            
        indices = np.array([ i for i in range(0,min(len(self.data_exact.dt),len(self.data_exact.dt)),len(self.data.dt)//maxpoints + 1) ])

        if hasattr(self,'F_ansatz') and force_exact is not None:
            self.exact_F = [ force_exact(self.data_exact.X_ito[i],self.data_exact.v_hat[i]) for i in indices ]
            self.ansatz_F   = [ self.F_ansatz(self.data_exact.X_ito[i],self.data_exact.v_hat[i]) for i in indices ]

            # Compute the MSE along the trajectory. Data is scaled by
            # the average diffusion for dimensional correctness.
            self.MSE_F = sum([np.einsum('im,in,mn->',self.exact_F[i]-self.ansatz_F[i],self.exact_F[i]-self.ansatz_F[i],self.D_average_inv) for i,t in enumerate(indices) ]) / sum([np.einsum('im,in,mn->',self.ansatz_F[i],self.ansatz_F[i],self.D_average_inv) for i,t in enumerate(indices)])
            print("Normalized mean squared error on F along trajectory:",self.MSE_F)
            
        if hasattr(self,'D_ansatz') and D_exact is not None and not hasattr(D_exact,'shape'):
            self.exact_D =    [ D_exact(self.data_exact.X_ito[i],self.data_exact.v_hat[i]) for i in indices ]
            self.ansatz_D   = [ self.D_ansatz(self.data_exact.X_ito[i],self.data_exact.v_hat[i]) for i in indices ]
            self.MSE_D = sum([np.einsum('imn,iop,no,pm->',self.exact_D[i]-self.ansatz_D[i],self.exact_D[i]-self.ansatz_D[i],self.D_average_inv,self.D_average_inv) for i,t in enumerate(indices)]) / sum([np.einsum('imn,iop,no,pm->',self.ansatz_D[i],self.ansatz_D[i],self.D_average_inv,self.D_average_inv) for i,t in enumerate(indices)])
            print("Normalized mean squared error on D along trajectory:",self.MSE_D)
        elif  hasattr(self,'D_average') and D_exact is not None and hasattr(D_exact,'shape'):
            self.MSE_Dav =( np.einsum('mn,op,no,pm->',D_exact-self.D_average,D_exact-self.D_average,self.D_average_inv,self.D_average_inv) /
                         np.einsum('mn,op,no,pm->',self.D_average,self.D_average,self.D_average_inv,self.D_average_inv) )
            print("Normalized mean squared error on average D:",self.MSE_Dav)

    def simulate_bootstrapped_trajectory(self,oversampling=1,tlist=None):
        """Simulate an overdamped Langevin trajectory with the inferred
        ansatz force field and similar time series and initial
        conditions as the input data.
        """
        from ULI_langevin import UnderdampedLangevinProcess
        if tlist is None:
            tlist = self.data.t
        if hasattr(self,'D_ansatz'): 
            return UnderdampedLangevinProcess(self.F_ansatz, self.D_ansatz, tlist, initial_position = 1. * self.data.X_ito[0],initial_velocity = 1. * self.data.v_hat[0],oversampling=oversampling)
        else:
            print("Simulating bootstrapped trajectory assuming constant diffusion.")
            return UnderdampedLangevinProcess(self.F_ansatz, self.D_average, tlist, initial_position = 1. * self.data.X_ito[0],oversampling=oversampling)


    def select_truncated_functions(self,min_partial_info=1):
        indices_functions_to_keep = np.array([ i for i,dI in enumerate(self.partial_information) if dI > min_partial_info ])
        return lambda X,V : self.force_projectors.b(X,V)[:,indices_functions_to_keep]
    

        
    # Local diffusion estimators. All these are local-in-time noisy
    # estimates of the diffusion tensor (noise is O(1)). Choose it
    # adapted to the problem at hand.
    def __D_discrete__(self,t):
        # Simple 3-points estimator (does not correct for measurement noise)
        return np.einsum('im,in->imn',self.data.a_hat[t],
                                      self.data.a_hat[t]) * 0.75 * self.data.dt[t] 

    def __D_noisy__(self,t):
        # A 4-points estimator that corrects for biases induced by measurement noise
        D = 3 * (                   np.einsum('im,in->imn',self.data.dX[t],self.data.dX[t])             * (-1 ) + # a
                                    np.einsum('im,in->imn',self.data.dX_minus[t],self.data.dX_minus[t]) * ( 1 ) + # b
                                    np.einsum('im,in->imn',self.data.dX_plus[t],self.data.dX_plus[t])   * ( 1 ) + # c
                                    np.einsum('im,in->imn',self.data.dX_plus[t],self.data.dX_minus[t])  * (-3 ) + # d
                                    np.einsum('im,in->imn',self.data.dX[t],self.data.dX_plus[t])        * ( 1 ) +  # e 
                                    np.einsum('im,in->imn',self.data.dX[t],self.data.dX_minus[t])       * ( 1 )   # f
              ) / ( 11 * self.data.dt[t]**3 )  
        return 0.5*(D + np.einsum('inm->imn',D))


    def __Lambda__(self,t):
        # An estimator for the amplitude of the measurement noise
        L =     (                   np.einsum('im,in->imn',self.data.dX[t],self.data.dX[t])             * ( 10) + # a
                                    np.einsum('im,in->imn',self.data.dX_minus[t],self.data.dX_minus[t]) * ( 1 ) + # b
                                    np.einsum('im,in->imn',self.data.dX_plus[t],self.data.dX_plus[t])   * ( 1 ) + # c
                                    np.einsum('im,in->imn',self.data.dX_plus[t],self.data.dX_minus[t])  * ( 8 ) + # d
                                    np.einsum('im,in->imn',self.data.dX[t],self.data.dX_plus[t])        * (-10) +  # e 
                                    np.einsum('im,in->imn',self.data.dX[t],self.data.dX_minus[t])       * (-10)   # f
              ) / 44
        return 0.5*(L + np.einsum('inm->imn',L))

        
def flatten_symmetric(M,dim):
    # A helper function to go from dxd array to d(d+1)/2 vector with
    # the upper triangular values.
    return np.array([ M[i,j] for i in range(dim) for j in range(i+1)])

def inflate_symmetric(V,dim):
    # The revert operation
    M = np.zeros((dim,dim))
    k = 0
    for i in range(dim):
        for j in range(i+1):
            M[i,j] = V[k]
            M[j,i] = V[k]
            k += 1
    return M 
