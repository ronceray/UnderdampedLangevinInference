
""" A library of projection bases for Underdamped Langevin Inference. """



import numpy as np



def basis_selector(basis,data):
    is_interacting = False
    if basis['type'] == 'polynomial':
        funcs = polynomial_basis(data.d,basis['order'])
    elif basis['type'] == 'Fourier':
        funcs = Fourier_basis(data.d,basis['width_x'],basis['width_v'],basis['center'],basis['order'])
    elif basis['type'] == 'fourierX_polyV':
        funcs = fourierX_polyV_basis(data.d,basis['center_x'],basis['width_x'],basis['order_x'],basis['order_v'])
    else: # Interacting particles basis
        is_interacting = True
        if basis['type'] == 'particles_pair_interaction': 
            funcs = particles_pair_interaction(data.d,basis['kernels'])
        if basis['type'] == 'particles_pair_alignment_interaction': 
            funcs = particles_pair_alignment_interaction(basis['kernels'])
        elif basis['type'] == 'self_propelled_particles': 
            funcs = self_propelled_particles_basis(basis['order'],basis['kernels'])
        elif basis['type'] == 'aligning_self_propelled_particles': 
            funcs = aligning_self_propelled_particles_basis(basis['order'],basis['kernels'])
        elif basis['type'] == 'aligning_flock': 
            funcs = aligning_flock_basis(basis['order'],basis['kernels'],data.d,basis['translation_invariant'])
        else:
            raise KeyError("Unknown basis type.") 
    return funcs,is_interacting


def polynomial_basis(dim,order):
    # A simple polynomial basis, X -> X_mu X_nu ... up to polynomial
    # degree 'order'.
    
    # We first generate the coefficients, ie the indices mu,nu.. of
    # the polynomials, in a non-redundant way. We start with the
    # constant polynomial (empty list of coefficients) and iteratively
    # add new indices.
    coeffs = [ np.array([[]],dtype=int) ]
    for n in range(order):
            # Generate the next coefficients:
            new_coeffs = []
            for c in coeffs[-1]:
                # We generate loosely ordered lists of coefficients
                # (c1 >= c2 >= c3 ...)  (avoids redundancies):
                for i in range( (c[-1]+1) if c.shape[0]>0 else 2*dim ):
                    new_coeffs.append(list(c)+[i])
            coeffs.append(np.array(new_coeffs,dtype=int)) 
    # Group all coefficients together
    coeffs = [ c for degree in coeffs for c in degree ]
    def Polynomial(X,V):
        return np.array([[ np.prod(np.array(list(X[i,:])+list(V[i,:]))[c]) for c in coeffs ] for i in range(X.shape[0])])
    return Polynomial

def velocity_polynomial_basis(dim,order):
    # A polynomial in the velocity only.
    coeffs = [ np.array([[]],dtype=int) ]
    for n in range(order):
            # Generate the next coefficients:
            new_coeffs = []
            for c in coeffs[-1]:
                # We generate loosely ordered lists of coefficients
                # (c1 >= c2 >= c3 ...)  (avoids redundancies):
                for i in range( (c[-1]+1) if c.shape[0]>0 else dim ):
                    new_coeffs.append(list(c)+[i])
            coeffs.append(np.array(new_coeffs,dtype=int)) 
    # Group all coefficients together
    coeffs = [ c for degree in coeffs for c in degree ]
    return lambda X,V : np.array([[ np.prod(x[c]) for c in coeffs ] for x in V]) 

def polynomial_basis_labels(dim,order):
    # A helper function: get the human-readable labels of the
    # functions in the polynomial basis.
    coeffs = [ np.array([[]],dtype=int) ]
    for n in range(order):
            # Generate the next coefficients:
            new_coeffs = []
            for c in coeffs[-1]:
                # We generate loosely ordered lists of coefficients
                # (c1 >= c2 >= c3 ...)  (avoids redundancies):
                for i in range( (c[-1]+1) if c.shape[0]>0 else 2*dim ):
                    new_coeffs.append(list(c)+[i])
            coeffs.append(np.array(new_coeffs,dtype=int)) 
    # Group all coefficients together
    coeffs = [ c for degree in coeffs for c in degree ]
    coeffs_lowdim = np.array([ [ list(c).count(i) for i in range(2*dim) ] for c in coeffs ])
    labels = []
    for c in coeffs_lowdim:
        label = ""
        for i,n in enumerate(c[:dim]):
            if n > 0:
                label += "x"+str(i)
                if n > 1:
                    label += "^"+str(n)
                label +="."
        for i,n in enumerate(c[dim:]):
            if n > 0:
                label += "v"+str(i)
                if n > 1:
                    label += "^"+str(n)
                label +="."
        if len(label)==0:
            label = "1"
        labels.append(label)
    return labels


def Fourier_basis(dim,width_x,width_v,center,order):
    coeffs = [ np.array([[]],dtype=int) ]
    for n in range(order):
            # Generate the next coefficients:
            new_coeffs = []
            for c in coeffs[-1]:
                # We generate loosely ordered lists of coefficients
                # (c1 >= c2 >= c3 ...)  (avoids redundancies):
                for i in range( (c[-1]+1) if c.shape[0]>0 else 2*dim ):
                    new_coeffs.append(list(c)+[i])
            coeffs.append(np.array(new_coeffs,dtype=int))
        
    coeffs = [ c for degree in coeffs[1:] for c in degree ]
    coeffs_lowdim = np.array([ [ list(c).count(i) for i in range(2*dim) ] for c in coeffs ])
    
    def Fourier(X,V):
        Xc = 2 * np.pi* (X - center) / width_x
        Vs = 2 * np.pi* V / width_v
        vals = np.ones((len(Xc),2*len(coeffs_lowdim)+1))
        for j,x in enumerate(Xc):
            xv = np.array(list(x)+list(V[j])) 
            for i,c in enumerate(coeffs_lowdim):
                vals[j,2*i+1] = np.cos( xv.dot(c))
                vals[j,2*i+2] = np.sin( xv.dot(c))
        return vals 
    return Fourier 

def fourierX_polyV_basis(dim,center_x,width_x,order_x,order_v):
    
    def polyV_basis(dim,order):
        coeffs = [ np.array([[]],dtype=int) ]
        for n in range(order):
                # Generate the next coefficients:
                new_coeffs = []
                for c in coeffs[-1]: 
                    # We generate loosely ordered lists of coefficients
                    # (c1 >= c2 >= c3 ...)  (avoids redundancies):
                    for i in range( (c[-1]+1) if c.shape[0]>0 else dim ):
                        new_coeffs.append(list(c)+[i])
                coeffs.append(np.array(new_coeffs,dtype=int)) 
        # Group all coefficients together
        coeffs = [ c for degree in coeffs for c in degree ] 
        return lambda V : np.array([[ np.prod(v[c]) for c in coeffs ] for v in V]) 

    def fourierX_basis(dim,order,center,width):
        coeffs = [ np.array([[]],dtype=int) ]
        for n in range(order):
                # Generate the next coefficients:
                new_coeffs = []
                for c in coeffs[-1]:
                    # We generate loosely ordered lists of coefficients
                    # (c1 >= c2 >= c3 ...)  (avoids redundancies):
                    for i in range( (c[-1]+1) if c.shape[0]>0 else dim ):
                        new_coeffs.append(list(c)+[i])
                coeffs.append(np.array(new_coeffs,dtype=int))
            
        coeffs = [ c for degree in coeffs[1:] for c in degree ]
    
        coeffs_lowdim = np.array([ [ list(c).count(i) for i in range(dim) ] for c in coeffs ])
        def Fourier(X):
            Xc = 2 * np.pi* (X - center) / width
            vals = np.ones((len(Xc),2*len(coeffs_lowdim)+1))
            for j,x in enumerate(Xc):
                for i,c in enumerate(coeffs_lowdim):
                    vals[j,2*i+1] = np.cos( x.dot(c))
                    vals[j,2*i+2] = np.sin( x.dot(c))
            return vals
        return Fourier 
    
    fourierX = fourierX_basis(dim,order_x,center_x,width_x)
    polyV = polyV_basis(dim,order_v)
    
    def fourierX_polyV(X,V):
        fX = fourierX(X)
        pV = polyV(V)
        return np.reshape(np.einsum('ia,ib->iab',fX,pV),(fX.shape[0],fX.shape[1]*pV.shape[1]))
        
    return fourierX_polyV 

### INTERACTING PARTICLES ###
def particles_pair_interaction(dim,kernels):
    # Radially symmetric vector-like pair interactions as a sum of
    # kernels.  Two-particle functions are chosen to be of the form
    # f(R_ij) * (Xj - Xi)/Rij for a given set of functions f
    # (kernels).
    def pair_function_spherical(X):
        # X is a Nparticles x dim - shaped array.
        Nparticles = X.shape[0]
        Xij = np.array([[ Xj - Xi for j,Xj in enumerate(X) ] for i,Xi in enumerate(X) ])
        Rij = np.linalg.norm(Xij,axis=2)
        f_Rij = np.nan_to_num(np.array([ f(Rij)/Rij for f in kernels ]))
        # Combine the kernel index f and the spatial index m into a
        # single function index a:
        return np.einsum('fij,ijm->ifm',f_Rij,Xij).reshape((Nparticles,dim * len(kernels)))
    return pair_function_spherical

def self_propelled_particles_basis(order_single,kernels):
    # A basis adapted to 2D self-propelled particles without alignment
    self_propulsion =  lambda X,V : np.array([ np.cos(X[:,2]), np.sin(X[:,2]),-V[:,2] ]).T 
    poly = polynomial_basis(2,order_single)
    pair = particles_pair_interaction(2,kernels)
    return lambda X,V :  np.array([ v for v in poly(X[:,:2],V[:,:2]).T ]+[ V[:,2] ]+[ v for v in self_propulsion(X,V).T ]+[ v for v in pair(X[:,:2]).T ]).T


def particles_pair_alignment_interaction(kernels):
    # Radially symmetric vector-like pair interactions as a sum of
    # kernels.  Two-particle functions are chosen to be of the form
    # f(R_ij) * (Xj - Xi)/Rij for a given set of functions f
    # (kernels).
    def pair_function_alignment(X):
        # X is a Nparticles x dim - shaped array.
        Nparticles = X.shape[0]
        Xij = np.array([[ Xj - Xi for j,Xj in enumerate(X) ] for i,Xi in enumerate(X) ])
        Xij[:,:,2] = np.sin(Xij[:,:,2])
        Rij = np.linalg.norm(Xij[:,:,:2],axis=2)
        f_Rij = np.nan_to_num(np.array([ f(Rij)/Rij for f in kernels ]))
        # Combine the kernel index f and the spatial index m into a
        # single function index a:
        return np.einsum('fij,ijm->ifm',f_Rij,Xij).reshape((Nparticles, 3 * len(kernels)))
    return pair_function_alignment


def aligning_self_propelled_particles_basis(order_single,kernels):
    # A basis adapted to 2D self-propelled particles without alignment
    self_propulsion =  lambda X,V : np.array([ np.cos(X[:,2]), np.sin(X[:,2]),-V[:,2] ]).T 
    poly = polynomial_basis(2,order_single)
    pair_align = particles_pair_alignment_interaction(kernels)
    return lambda X,V :  np.array([ v for v in poly(X[:,:2],V[:,:2]).T ]+[ v for v in self_propulsion(X,V).T ]+[ v for v in pair_align(X).T ] ).T



def aligning_flock_basis(order_single,kernels,dim,translation_invariant):
    # A basis adapted to 2D self-propelled particles without alignment
    if translation_invariant:
        poly = velocity_polynomial_basis(dim,order_single)
    else:
        poly = polynomial_basis(dim,order_single)
    def pair_align(X,V):
        # X is a Nparticles x dim - shaped array.
        Nparticles = X.shape[0]
        Xij = np.array([[ Xj - Xi for j,Xj in enumerate(X) ] for i,Xi in enumerate(X) ])
        Vij = np.array([[ Vj - Vi for j,Vj in enumerate(V) ] for i,Vi in enumerate(V) ])
        Rij = np.linalg.norm(Xij,axis=2)
        f_Rij = np.nan_to_num(np.array([ f(Rij) for f in kernels ]))
        # Combine the kernel index f and the spatial index m into a
        # single function index a:
        fX_i = np.einsum('fij,ijm->fim',f_Rij,Xij)
        fV_i = np.einsum('fij,ijm->fim',f_Rij,Vij)
        return np.einsum('fim->ifm', np.array([ x for x in fX_i]+[ v for v in fV_i])).reshape((Nparticles, (2*dim) * len(kernels)))
    
    return lambda X,V :  np.array([ v for v in poly(X,V).T ]+[ v for v in pair_align(X,V).T ] ).T


 
