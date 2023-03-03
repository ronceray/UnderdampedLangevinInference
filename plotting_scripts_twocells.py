import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_Fxv(F_ansatz,N_bins,factor_divide):
    vmax_plot = 80 #100 um/h
    xmax_plot = 47 #50 um

    Fmax = 300 #400 um/h^2

    N=N_bins
    x_range=2
    v_range=4
    x_center = 0.
    v_center = 0.

    streamfactor = 5.5

    gridX = np.linspace(x_center - 0.5*x_range,x_center + 0.5*x_range,N)
    gridV = np.linspace(v_center - 0.5*v_range,v_center + 0.5*v_range,N)

    Fx = np.array([v for v in gridV for x in gridX])
    Fv = np.array([F_ansatz(np.array([[x]]),np.array([[v]])) for v in gridV for x in gridX])

    X, Y = np.meshgrid(gridX*factor_divide, gridV*factor_divide)
    Fv_matrix = np.array(Fv*factor_divide).reshape(N,N)
    Fx_matrix = np.array(Fx*factor_divide).reshape(N,N)
    
    speed = np.sqrt(Fx_matrix**2 + Fv_matrix**2)
    lw = streamfactor*np.sqrt(speed) / np.sqrt(speed.max())
    
    plt.pcolor(X, Y, Fv_matrix,cmap='jet',vmin=-Fmax,vmax=Fmax,shading='auto')
    plt.colorbar(ticks=[-Fmax, 0, Fmax])
    
    plt.streamplot(X, Y,Fx_matrix,Fv_matrix,color='white',density=[0.7, 0.7],linewidth=lw)

    plt.xticks([-30,0,30])
    plt.yticks([-80,0,80])
    plt.xlim([-xmax_plot,xmax_plot])
    plt.ylim([-vmax_plot,vmax_plot])
    plt.xlabel(r"$x \ (\mu \mathrm{m})$",fontsize=12)
    plt.ylabel(r"$v \ (\mu \mathrm{m h}^{-1})$",fontsize=12)
       
        
def plot_cohesion(kernels_coh,kernels_al,F_coefficients,N_bins,factor_divide):
    rmax = 100 #100 um  22/50
    cohmax = 100
    
    dim = 1
    color = 'g'

    Nfuncs_coh = len(kernels_coh)
    Nfuncs_al = len(kernels_al)
    rvals = np.linspace(0,2,500)

    Fmunu_radial = F_coefficients[:dim,-dim*(Nfuncs_coh+Nfuncs_al):-dim*Nfuncs_al].reshape(dim,Nfuncs_coh,dim)
    
    # Select the fitting coefficients on the radial kernels, and take the
    # isotropic part:
    Fradial_params = np.einsum('mkm->k',Fmunu_radial)/dim
    Fradial = lambda r : sum( kernels_coh[i](r) * Fradial_params[i]   for i in range(Nfuncs_coh) )
    
    plt.plot([0,rmax],[0,0],lw=0.8,color='k')
    plt.plot(rvals*factor_divide,rvals*factor_divide*Fradial(rvals),lw=4,color=color)

    plt.ylim(-cohmax,cohmax)
    plt.xlim(0,rmax)
    
    plt.xlabel(r"$|\Delta x| \ (\mu \mathrm{m})$")
    plt.ylabel(r"$f(|\Delta x|)*|\Delta x| \ (\mu \mathrm{m \ h}^{-2})$")
    
def plot_friction(kernels_al,F_coefficients,N_bins,factor_divide):
    rmax = 100 #100 um  22/50
    gammamax = 3
    
    dim = 1
    color = 'g'
    Nfuncs_al = len(kernels_al)
    rvals = np.linspace(0,2,500)

    Fmunu_align = F_coefficients[:dim,-dim*Nfuncs_al:].reshape(dim,Nfuncs_al,dim)
    
    # Select the fitting coefficients on the radial kernels, and take the
    # isotropic part:
    Falign_params = np.einsum('mkm->k',Fmunu_align)/dim
    Falign = lambda r : sum( kernels_al[i](r) * Falign_params[i]  for i in range(Nfuncs_al) )
    
    start_r = 0
    plt.plot([0,rmax],[0,0],lw=0.8,color='k')
    plt.plot(rvals[start_r:]*factor_divide,Falign(rvals[start_r:]),lw=4,color=color)
    
    plt.ylim(-gammamax,gammamax)
    plt.yticks([-gammamax,0,gammamax])
    plt.xlim(0,rmax)

    plt.xlabel(r"$|\Delta x| \ (\mu \mathrm{m})$")
    plt.ylabel(r"$\gamma (|\Delta x|) \ (\mathrm{h}^{-1})$")
    

def plot_inference_fig(F_ansatz,kernels_coh,kernels_al,F_coefficients,ID,factor_divide,directory_print,file_type):

    # Prepare Matplotlib:
    plt.close('all')
    fig_size = [9,3]
    params = {'axes.labelsize': 10,
              'font.size':   14,
              'legend.fontsize': 12,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              #'text.usetex': True,
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params)
    plt.clf()

    file_suffix = '_' + ID + file_type


    fig1 = plt.figure(1)
    #margins = 0.1
    #fig1.subplots_adjust(left=margins, bottom=margins, right=1-margins, top=1-margins, wspace=0.4, hspace=0.4)
    H,W = 1,3

    N_bins = 50
    plt.subplot(H,W,1)
    plot_Fxv(F_ansatz,N_bins,factor_divide)
    
    plt.subplot(H,W,2)
    plot_cohesion(kernels_coh,kernels_al,F_coefficients,N_bins,factor_divide)
    
    plt.subplot(H,W,3)
    plot_friction(kernels_al,F_coefficients,N_bins,factor_divide)
    
    plt.tight_layout()

