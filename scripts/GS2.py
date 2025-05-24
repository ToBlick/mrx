import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


# 
# Source: http://www.jetp.ras.ru/cgi-bin/dn/e_026_02_0400.pdf

def solovev_solution(R: jnp.ndarray, Z: jnp.ndarray,b: float,c_0: float,a: float, r:float) -> jnp.ndarray:
    """
    Compute a Solov'ev analytical solution to the Grad-Shafranov equation.
    
    The solution has the form:
    ψ(R,Z) =  (br^2+c_0R^2)*Z**2/2 +(a-c_0)*(R^2-r^2)**2/8

    The magnetic axis is at R=r, Z=0
    
    Args:
        R: Radial coordinate 
        Z: Vertical coordinate 
        b: constant
        c_0: constant
        a: constant
        r: Radius of the magnetic axis
    
    Returns:
        np.ndarray: Poloidal flux function ψ(R,Z)
    """
    R_mesh, Z_mesh = jnp.meshgrid(R, Z)
    
    # Compute the solution
    psi = (b*r**2+c_0*R_mesh**2)*Z_mesh**2/2 +(a-c_0)*(R_mesh**2-r**2)**2/8
    
    return psi

# Rewrite as callable function for JAX differentiation
def solovev_solution_function(R: jnp.ndarray, Z: jnp.ndarray,b: float,c_0: float,a: float, r:float) -> jnp.ndarray:
    return (b*r**2+c_0*R**2)*Z**2/2 +(a-c_0)*(R**2-r**2)**2/8



def plot_flux_surfaces(R: jnp.ndarray, Z: jnp.ndarray, psi: jnp.ndarray) -> None:
    """
    Plot the flux surfaces (contours of constant ψ).
    
    Args:
        R: Radial coordinate array
        Z: Vertical coordinate array
        psi: Poloidal flux function array
    """
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # 2D contour plot
    ax1 = fig.add_subplot(121)
    R_mesh, Z_mesh = np.meshgrid(R, Z)
    contours = ax1.contour(R_mesh, Z_mesh, psi, levels=10)
    ax1.clabel(contours, inline=True, fontsize=8)
    ax1.set_xlabel('R')
    ax1.set_ylabel('Z')
    ax1.set_title('Flux Surfaces')
    ax1.grid(True)
    plt.show()





def main():
    """
    Main function 
    """
    # Set up coordinate grid
    R = np.linspace(0.0, 5.0, 100)  # Avoid R=0
    Z = np.linspace(-5.0, 5.0, 100)


    # Constants for the solution
    b = 1.0
    c_0 = 1.0
    a = 1.0
    r = 1.0
    

    # Compute solution
    psi = solovev_solution(R, Z,b,c_0,a,r)
    
    # Plot flux surfaces
    plot_flux_surfaces(R, Z, psi)
  

if __name__ == "__main__":
    main() 