# %%
import os

import jax
import jax.numpy as jnp
import sys

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory for figures
os.makedirs("script_outputs", exist_ok=True)

# @partial(jax.jit, static_argnames=["n", "p"])
def get_err(n, p):
    """
    Computes the error, condition number, and sparsity of the solution to the Poisson equation on a toroidal domain.

    Args:
        n: Number of elements in each direction.
        p: Polynomial degree.

    Returns:
        error: Error of the solution.
        cond: Condition number of the system.
        sparsity: Sparsity of the system.
    """
    # Set up finite element spaces
    q = p
    ns = (n, n, n)
    ps = (p, p, p)
    types = ("clamped", "periodic", "periodic")  # Types

    # Domain parameters
    a = 1/3
    R0 = 1.0
    π = jnp.pi

    def _X(r, χ):
        """Toroidal radial coordinate. Formula is:
        
        X(r, χ) = R0 + a * r * cos(2πχ)
        """
        return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * π * χ))

    def _Y(r, χ):
        """Toroidal vertical coordinate. Formula is:
        
        Y(r, χ) = R0 + a * r * cos(2πχ)
        """
        return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * π * χ))

    def _Z(r, χ):
        """Toroidal azimuthal coordinate. Formula is:
        
        Z(r, χ) = a * r * sin(2πχ)
        """
        return jnp.ones(1) * a * r * jnp.sin(2 * π * χ)

    def F(x):
        """Toroidal coordinate mapping function. Formula is:
        
        F(r, χ, z) = (X(r, χ) * cos(2πz), -Y(r, χ) * sin(2πz), Z(r, χ))

        Args:   
            x: Input logical coordinates (r, χ, z)

        Returns:
            F: Coordinate mapping function
        """
        r, χ, z = x
        return jnp.ravel(jnp.array([_X(r, χ) * jnp.cos(2 * π * z),
                                    -_Y(r, χ) * jnp.sin(2 * π * z),
                                    _Z(r, χ)]))
        
    def u(x):
        """Exact solution of the Poisson equation. Formula is:
        
        u(r, χ, z) = 1/4 * (r**2 - r**4) * cos(2πz)

        Args:
            x: Input logical coordinates (r, χ, z)

        Returns:
            u: Exact solution of the Poisson equation
        """
        r, χ, z = x
        return 1/4 * (r**2 - r**4) * jnp.cos(2 * π * z) * jnp.ones(1)
    
    def f(x):
        """Source term of the Poisson equation. Formula is:

        f(r, χ, z) = cos(2πz) * (-1/a**2 * (1 - 4r**2) - 1/(a*R) * (r/2 - r**3) * cos(2πχ) + 1/4 * (r**2 - r**4) / R**2 )

        Args:
            x: Input logical coordinates (r, χ, z)

        Returns:
            f: Source term of the Poisson equation
        """
        r, χ, z = x
        R = R0 + a * r * jnp.cos(2 * jnp.pi * χ)
        return jnp.cos(2 * jnp.pi * z) * (-1/a**2 * (1 - 4*r**2) - 1/(a*R) * (r/2 - r**3) * jnp.cos(2 * jnp.pi * χ) + 1/4 * (r**2 - r**4) / R**2 ) * jnp.ones(1)

    # Create DeRham sequence
    Seq = DeRhamSequence(ns, ps, q, types, F, polar=True, dirichlet=True)

    Seq.evaluate_1d()
    Seq.assemble_M0()
    Seq.assemble_dd0()

    # Solve the system
    u_hat = jnp.linalg.solve(Seq.M0 @ Seq.dd0, Seq.P0(f))
    u_h = DiscreteFunction(u_hat, Seq.Λ0, Seq.E0)

    # do not vmap here because of memory issues
    def diff_at_x(x):
        """Difference between exact and computed solution.

        Args:
            x: Input logical coordinates (r, χ, z)

        Returns:
            diff: Difference between exact and computed solution
        """
        return u(x) - u_h(x)

    def body_fun(carry, x):
        return None, diff_at_x(x)

    # TODO: Explain what is happening below.
    _, df = jax.lax.scan(body_fun, None, Seq.Q.x)
    L2_df = jnp.einsum('ik,ik,i,i->', df, df, Seq.J_j, Seq.Q.w)**0.5
    L2_f = jnp.einsum('ik,ik,i,i->',
                      jax.vmap(u)(Seq.Q.x), jax.vmap(u)(Seq.Q.x),
                      Seq.J_j, Seq.Q.w)**0.5
    error = L2_df / L2_f
    return error, jnp.linalg.cond(Seq.M0 @ Seq.dd0), jnp.sum(jnp.abs(Seq.M0 @ Seq.dd0) > 1e-12) / Seq.dd0.size


def main():
    """Run get_err for a single (n,p) taken from command-line arguments and save results to a text file.

    Usage: python toroid_poisson.py <n> <p>

    Raises:
        ValueError: If n or p are not integers or n <= p
    """
    if len(sys.argv) < 3:
        print("Usage: python toroid_poisson.py <n> <p>")
        sys.exit(1)

    try:
        n = int(sys.argv[1])
        p = int(sys.argv[2])
    except ValueError:
        print("Both n and p must be integers.")
        sys.exit(1)

    # Compute results
    error, cond, sparsity = get_err(n, p)

    # get_err returns (error, cond, sparsity). The user requested the order: error, sparsity, cond.
    error_f = float(error)
    sparsity_f = float(sparsity)
    cond_f = float(cond)

    # Ensure output directory exists
    os.makedirs("script_outputs", exist_ok=True)

    out_name = f"toroid_poisson_{n}_{p}.txt"
    out_path = os.path.join("script_outputs", out_name)
    with open(out_path, "w") as fh:
        fh.write(f"error {error_f:.18e}\n")
        fh.write(f"sparsity {sparsity_f:.18e}\n")
        fh.write(f"cond {cond_f:.18e}\n")

    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()

# %%
