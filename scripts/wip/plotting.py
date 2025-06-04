"""
File for plotting relevant plots and figures

This file contains functions that:
- plots spline basis functions using the spline basis defined in mrx.SplineBases
- plots B-field associted with analytical solution of GSE from Freidberg (Eq. 6.103)
"""
import numpy as np
import matplotlib.pyplot as plt
from mrx.SplineBases import SplineBasis

def plot_spline_basis(n:int, p:int, spline_type='clamped'): #Default spline type is clamped
    """
    Plot the spline basis functions for given parameters.
    
    Args:
        n (int): Number of basis functions
        p (int): Degree of spline
        spline_type (str): Type of spline ('clamped', 'periodic', or 'constant')
    """
    # Create spline basis
    basis = SplineBasis(n, p,spline_type)
    
    # Create points for evaluation
    x = np.linspace(0, 1, 400)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Define custom (cute) colors
    custom_colors = ['#b080c2', '#ba1e96', '#65a0ba', '#62c467', 
                    '#FFEEAD', '#ad1609', '##6b81c9', '#5a147d']
    
    # Plot each basis function with different colors
    for i in range(n):
        y = [basis(x_i, i) for x_i in x]  
        plt.plot(x, y, label=f'B_{i,3}', color=custom_colors[i], linewidth=2.5)
    
    plt.title(f'{spline_type.capitalize()} Spline Basis Functions (n={n}, p={p})', 
              fontsize=26, pad=20)
    plt.xlabel('x', fontsize=25)
    plt.ylabel('Splines', fontsize=25)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=25)
    plt.tight_layout()
    plt.show()



# Analytical solution in Freidberg 6.103. Note the lack of dependence on phi.
def B_r(rho, theta, epsilon, B0, q_star, nu):
    """Compute the r component of the magnetic field.
    
    B_r = -epsilon * B0 * (nu/(2*q_star)) * (rho^2 - 1) * sin(theta)
    """
    return -epsilon * B0 * (nu/(2*q_star)) * (rho**2 - 1) * np.sin(theta)

def B_theta(rho, theta, epsilon, B0, q_star, nu):
    """Compute the theta component of the magnetic field.
    
    B_theta = epsilon * B0 * (1/q_star) * [rho + (nu/2)(3*rho^2 - 1)cos(theta)]
    """
    return epsilon * B0 * (1/q_star) * (rho + (nu/2) * (3*rho**2 - 1) * np.cos(theta))


def B_phi(rho, theta, epsilon, B0, beta_t, nu):
    """Compute the phi component of the magnetic field.
    
    B_phi = B0 * (1 - epsilon*rho*cos(theta) - beta_t*(1-rho^2)*(1 + nu*rho*cos(theta)))
    """
    return B0 * (1 - epsilon*rho*np.cos(theta) - beta_t*(1-rho**2)*(1 + nu*rho*np.cos(theta)))

def plot_Friedberg_GS(n_rho:int, n_theta:int):
    """Plot the magnetic field components in toroidal coordinates for Eq. (6.103) in Freidberg

    Args:
        n_rho (int): Number of grid points in rho direction
        n_theta (int): Number of grid points in theta direction
    """
    # Set font sizes
    small = 14
    medium = 18
    big = 24

    # Set font sizes for different elements for readability 
    plt.rc('font', size=small)          
    plt.rc('axes', titlesize=big)    
    plt.rc('axes', labelsize=small)    
    plt.rc('xtick', labelsize=small)    
    plt.rc('ytick', labelsize=small)    
    plt.rc('legend', fontsize=medium)   
    plt.rc('figure', titlesize=big)  

    # Create grid points
    rho = np.linspace(0, 1, n_rho)
    theta = np.linspace(0, 2*np.pi, n_theta)
    rho_grid, theta_grid = np.meshgrid(rho, theta)
    
    # Set parameters from P. 157 in Freidberg
    epsilon = 0.35  # inverse aspect ratio
    B0 = 1.0      
    q_star = 1.5  
    nu = 0.8    # crucial high tokamak parameter
    beta_t = 0.12 # toroidal beta
    
    # Compute magnetic field components
    B_rad = B_r(rho_grid, theta_grid, epsilon, B0, q_star, nu)
    B_th = B_theta(rho_grid, theta_grid, epsilon, B0, q_star, nu)
    B_ph = B_phi(rho_grid, theta_grid, epsilon, B0, beta_t, nu)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    
    # Plot B_r
    c1 = ax1.contourf(rho_grid, theta_grid, B_rad, levels=20, cmap='viridis')
    ax1.set_title(r'$B_{r}$')
    ax1.set_xlabel(r'$\rho$')
    ax1.set_ylabel(r'$\theta$')
    plt.colorbar(c1, ax=ax1)
    
    # Plot B_theta
    c2 = ax2.contourf(rho_grid, theta_grid, B_th, levels=20, cmap='viridis')
    ax2.set_title(r'$B_{\theta}$')
    #ax2.set_xlabel(r'$\rho$')
    #ax2.set_ylabel(r'$\theta$')
    plt.colorbar(c2, ax=ax2)
    
    # Plot B_phi
    c3 = ax3.contourf(rho_grid, theta_grid, B_ph, levels=20, cmap='viridis')
    ax3.set_title(r'$B_{\phi}$')
    ax3.set_xlabel(r'$\rho$')
    ax3.set_ylabel(r'$\theta$')
    plt.colorbar(c3, ax=ax3)
    
    plt.suptitle('Magnetic Field Components in Toroidal Coordinates', y=1.05)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #plot_spline_basis(n=4, p=3, spline_type='periodic')
    #plot_spline_basis(n=4, p=3, spline_type='clamped') 
    plot_Friedberg_GS(50, 50)