# UnderdampedLangevinInference	

---
This package is now DEPRECATED as it has been merged with StochasticForceInference. 
---
UnderdampedLangevinInference is a package aimed at inferring the 
deterministic and stochastic contributions of underdamped stochastic processes (underdamped Langevin dynamics). 

**Reference**: 
    David Brückner, Pierre Ronceray and Chase Broedersz, 
    "*Inferring the dynamics of underdamped stochastic systems*", Phys. Rev. Lett. 125, 058103 (2020).
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.058103

**Authors**: Pierre Ronceray and David Brückner. 

**Contact**: pierre.ronceray@outlook.com or david.brueckner@ist.ac.at

**Website**: www.pierre-ronceray.net and www.davidbrueckner.de

See also **StochasticForceInference**, the overdamped (Brownian dynamics) equivalent of this package: https://github.com/ronceray/StochasticForceInference

-----------------------------------------------------------------------

Developed in Python 3.6. Dependencies:

- NumPy, SciPy

- Optional: Matplotlib (2D plotting, recommended); Mayavi2 (3D
  plotting)

-----------------------------------------------------------------------

###Contents:

**UnderdampedLangevinInference**: a front-end includer of all classes
   useful to the user.

**ULI_data.py**: contains a data wrapper class, StochasticTrajectoryData,
   which formats trajectories for force and diffusion inference. Also
   contains a number of plotting routines. See this file for the different ways to
   initialize it using data.

**ULI_inference.py**: implements the core force and diffusion
   inference class, UnderdampedLangevinInference, that reconstructs the
   these fields and computes the inference error on the force field.  
   Takes as input a StochasticTrajectoryData instance, and inference parameters.

**ULI_langevin.py**: contains the class UnderdampedLangevinProcess, which
   implements a simple Ito integration of the second order Langevin equation, useful
   for testing the method with known models. It takes as input a force
   field and a diffusion tensor field. Also used by
   UnderdampedLangevinInference to predict new trajectories with the
   inferred force and diffusion field.

**ULI_projectors.py**: implements an internal class, TrajectoryProjectors,
   used by UnderdampedLangevinInference. Given a
   set of fitting functions, it orthonormalizes it as a premise to the
   inference.

**ULI_bases.py**: provides an internal dictionary of (more or less)
   standard fitting bases, such as polynomials. This dictionary is
   called by UnderdampedLangevinInference at
   initialization, unless a custom base is provided by the user.

**ULI_plotting_toolkit.py**: a few plotting functions for the convenience
   of the author.

**ULI_demo_VdP.py**: a fully commented example of force and diffusion
   inference on the example of 1D non-linear oscillator. **Start here!**
   
**ULI_demo_VdP.py**: a fully commented example of force, interaction and diffusion
   inference on the example of experimental data of two interacting migrating cells.
   For this, download Dataset_S01.txt from https://www.pnas.org/doi/10.1073/pnas.2016602118, add it into your directory and hit run! **Check this if you want to get started on experimental data!**
   
-----------------------------------------------------------------------


Enjoy, and please send feedback to pierre.ronceray@outlook.com or d.brueckner@campus.lmu.de !

       	   	       				    
						
-----------------------------------------------------------------------

