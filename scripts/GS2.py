import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mrx.Utils import grad as Grad


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

def solovev_magnetic_field(R: float, Z: float, b: float, c_0: float, a: float, r: float, B_0: float) -> jnp.ndarray:
    """
    Compute the magnetic field components (B_R, B_phi, B_Z) for the Solovev equilibrium.
    
    The magnetic field components are defined as:
    B_R = -(1/R) * ∂ψ/∂Z
    B_phi = B_0 * r/R
    B_Z = (1/R) * ∂ψ/∂R

    We need to compute the partial derivatives. We have that:
    ψ_R(R,Z) =  (c_0R)*Z**2 +(a-c_0)*2R*(R^2-r^2)/4
    ψ_Z(R,Z) =  (br^2+c_0R^2)*Z 

    Args:
        R: Radial coordinate
        Z: Vertical coordinate
        b: constant
        c_0: constant
        a: constant
        r: Radius of the magnetic axis
        B_0: B_phi at magnetic axis phi = 0 
        
    Returns:
        jnp.ndarray: (B_R, B_phi, B_Z) magnetic field components
    """
    # Mesh R and Z

    R_mesh, Z_mesh = jnp.meshgrid(R, Z)


    # Compute partial derivatives 

    def dpsi_dR(R,Z,c_0,a,r,B_0):
        return (c_0*R)*Z**2 +(a-c_0)*2*R*(R**2-r**2)/4

    def dpsi_dZ(R,Z,b,c_0,a,r,B_0):
        return (b*r**2+c_0*R**2)*Z
    
    dpsi_dR_eval = dpsi_dR(R_mesh,Z_mesh,c_0,a,r,B_0)
    dpsi_dZ_eval = dpsi_dZ(R_mesh,Z_mesh,b,c_0,a,r,B_0)

    # Calculate magnetic field components
    B_R = -(1/R_mesh) * dpsi_dZ_eval
    B_phi = B_0 * r/R_mesh  
    B_Z = (1/R_mesh) * dpsi_dR_eval 
    
    return B_R, B_phi, B_Z

print(solovev_magnetic_field(jnp.linspace(0.1,1,10), jnp.linspace(0,1,10), 1.0, 1.0, 1.0, 1.0,1.0))

def plot_flux_surfaces(R: jnp.ndarray, Z: jnp.ndarray, psi: jnp.ndarray) -> None:
    """
    Plot the flux surfaces.
    
    Args:
        R: Radial coordinate jnp.ndarray
        Z: Vertical coordinate jnp.ndarray
        psi: Poloidal flux function jnp.ndarray
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