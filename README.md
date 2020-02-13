# StochasticInertialForceInference	


StochasticInertialForceInference is a package aimed at inferring the 
deterministic and stochastic contributions of stochastic inertial processes (underdamped Langevin dynamics). 

**Reference**: 
    David Brückner, Pierre Ronceray and Chase Broedersz, 
    "*Inferring the non-linear dynamics of stochastic inertial systems*", 2020.

**Authors**: Pierre Ronceray and David Brückner. 

**Contact**: pierre.ronceray@outlook.com or d.brueckner@campus.lmu.de

Web page: www.pierre-ronceray.net.

See also **StochasticForceInference**, the overdamped (Brownian dynamics) equivalent of this package: https://github.com/ronceray/StochasticForceInference

-----------------------------------------------------------------------

Developed in Python 3.6. Dependencies:

- NumPy, SciPy

- Optional: Matplotlib (2D plotting, recommended); Mayavi2 (3D
  plotting)

-----------------------------------------------------------------------

**Contents**:

**StochasticInertialForceInference.py**: a front-end includer of all classes
   useful to the user.

**SIFI_data.py**: contains a data wrapper class, StochasticTrajectoryData,
   which formats trajectories for force and diffusion inference. Also
   contains a number of plotting routines. See this file for the different ways to
   initialize it using data.

**SIFI_inference.py**: implements the core force and diffusion
   inference class, StochasticInertialForceInference, that reconstructs the
   these fields and computes the inference error on the force field.  
   Takes as input a StochasticTrajectoryData instance, and inference parameters.

**SIFI_langevin.py**: contains the class UnderdampedLangevinProcess, which
   implements a simple Ito integration of the second order Langevin equation, useful
   for testing the method with known models. It takes as input a force
   field and a diffusion tensor field. Also used by
   StochasticInertialForceInference to predict new trajectories with the
   inferred force and diffusion field.

**SIFI_projectors.py**: implements an internal class, TrajectoryProjectors,
   used by StochasticInertialForceInference. Given a
   set of fitting functions, it orthonormalizes it as a premise to the
   inference.

**SIFI_bases.py**: provides an internal dictionary of (more or less)
   standard fitting bases, such as polynomials. This dictionary is
   called by StochasticInertialForceInference at
   initialization, unless a custom base is provided by the user.

**SIFI_plotting_toolkit.py**: a few plotting functions for the convenience
   of the author.

**SIFI_demo_VdP.py**: a fully commented example of force and diffusion
   inference on the example of 1D non-linear oscillator. **Start here!**	       
   
-----------------------------------------------------------------------


Enjoy, and please send feedback to pierre.ronceray@outlook.com or d.brueckner@campus.lmu.de !

       	   	       				    
						
-----------------------------------------------------------------------
