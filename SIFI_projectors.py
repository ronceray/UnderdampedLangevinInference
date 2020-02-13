import numpy as np
from scipy.linalg import sqrtm, qr, LinAlgError




def flip(H):
    (w,h) = H.shape
    return np.array([[ H[w-1-i,h-1-j] for i in range(h)] for j in range(w)])


class TrajectoryProjectors(object):
    """This class deals with the orthonormalization of the input functions
    b with respect to the inner product.

    The b function is a single lambda function with array
    input/output, with structure

    (Nparticles,dim) -> (Nparticles,Nfunctions)

    where Nfunctions is the number of functions in the basis.
    """
    def __init__(self,inner_product,functions):
        # The inner product should not contract spatial indices - only
        # perform time integration.
        self.inner = inner_product
        self.b = functions
        self.B = self.inner(self.b,self.b)
        # Compute the normalization matrix H = B^{-1/2}:
        self.H = np.real(flip(qr(sqrtm(flip(np.linalg.pinv(self.B))),mode='r')[0]))

    def projector_combination(self,c_coefficients):
        """Returns a lambda function sum_alpha c_coefficients[..,alpha] *
        c_alpha(x). Avoids re-computing the dot product with the
        normalization matrix H."""
        b_coefficients = np.einsum('...a,ba->...b', c_coefficients, self.H)   
        function = lambda x,v : np.einsum('ia,...a->i...',self.b(x,v), b_coefficients) 
        return function, 1.*b_coefficients

    def grad_x_b(self,x,v,epsilon=1e-6):
        """The numerical gradient of the input function (to do: add an option
        for analytical formula for gradient). Output has shape:
        Nparticles x dim x Nparticles x Nfunctions.

        """
        Nparticles,d = x.shape
        dx = [[ np.array([[ 0 if (i,m)!= (ind,mu) else epsilon for m in range(d)] for i in range(Nparticles) ] )\
                for mu in range(d) ] for ind in range(Nparticles) ]  
        return np.array([[ (self.b(x+dx[ind][mu],v) - self.b(x-dx[ind][mu],v))/(2*epsilon) for mu in range(d)] for ind in range(Nparticles) ])

    def grad_v_b(self,x,v,epsilon=1e-6):
        """The numerical gradient of the input function (to do: add an option
        for analytical formula for gradient). Output has shape:
        Nparticles x dim x Nparticles x Nfunctions.

        """
        Nparticles,d = x.shape
        dv = [[ np.array([[ 0 if (i,m)!= (ind,mu) else epsilon for m in range(d)] for i in range(Nparticles) ] )\
                for mu in range(d) ] for ind in range(Nparticles) ]  
        return np.array([[ (self.b(x,v+dv[ind][mu]) - self.b(x,v-dv[ind][mu]))/(2*epsilon) for mu in range(d)] for ind in range(Nparticles) ])
