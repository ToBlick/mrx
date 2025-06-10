import jax.numpy as jnp

# This is an implementation of the MINRES algorithm

# M is the preconditioner
# A is the main matrix
# b is the right side
# n is the dimension 
# The algorithm is based on the paper "Obtaining Pseudo-inverse Solutions With MINRES"

# Test case with a symmetric positive definite matrix
A = jnp.array([
    [3.0, 1.0, 0.0],
    [1.0, 3.0, 1.0],
    [0.0, 1.0, 3.0]
])

# Diagonal preconditioner 
M = jnp.array([
    [0.25, 0.0, 0.0],   
    [0.0, 0.25, 0.0],
    [0.0, 0.0, 0.25]
])

# b
b = jnp.array([[1.0], [2.0], [3.0]])

def MINRES(n, A, M, b, max_iter=30, tol=1e-10):
    """
    Implementation of MINRES algorithm based on the paper
    "Obtaining Pseudo-inverse Solutions With MINRES"
    """
    # Initialization
    r_carrot = b  # r_carrot_0
    z_t = b  # z_1
    w_t = jnp.matmul(M, z_t)  # w_1
    β_t = jnp.sqrt(jnp.abs(jnp.matmul(jnp.transpose(z_t), w_t)))  # β_1
    ϕ_t = β_t  # ϕ_0
    r_hat = w_t  # r_hat_0
    c_t = jnp.float32(-1.0)  # c_0
    s_t = jnp.float32(0.0)  # s_0
    δ_t = jnp.float32(0.0)  # δ_1
    z_t = jnp.zeros((n, 1))  # z_1
    x_t = jnp.zeros((n, 1))  # x_0
    d_t = jnp.zeros((n, 1))  # d_0 
    d_previous = jnp.zeros((n, 1))  # d_{-1}
    β_previous = jnp.float32(1.0)
    z_previous = b  # z_0
    ϵ_t = jnp.float32(0.0)  # ϵ_0

    # Iteration
    for i in range(max_iter):
        # Check convergence
        residual = jnp.linalg.norm(jnp.matmul(A, x_t) - b)
        # If convergence is reached, break
        if residual < tol:
            break

        # Main work
        q_t = jnp.matmul(A, w_t) / β_t
        α_t = jnp.float32(jnp.matmul(jnp.transpose(w_t) / β_t, q_t))
        z_next = q_t - (α_t * z_t / β_t) - (β_t * z_previous / β_previous)
        w_next = jnp.matmul(M, z_next)
        β_next = jnp.sqrt(jnp.abs(jnp.matmul(jnp.transpose(z_next), w_next)))

        # Extract scalar values
        B = jnp.array([[α_t[0], β_next[0]]]).reshape((1, 2))
        α_t = B[0, 0]
        β_next = B[0, 1]

        # Matrix computation
        K = jnp.array([[c_t * δ_t + s_t * α_t, s_t * β_next],
                      [s_t * δ_t - c_t * α_t, -c_t * β_next]])

        gamma_2 = jnp.sqrt((K[1, 0])**2 + β_next**2)
        
        # If gamma_2 is not zero
        if gamma_2 > jnp.finfo(jnp.float32).eps:
            # Store next epsilon before updating other parameters
            ϵ_next = K[0, 1]  # s_t * β_next

            # Update parameters
            c_t = K[1, 0] / gamma_2
            s_t = β_next / gamma_2
            τ_t = c_t * ϕ_t
            ϕ_t = s_t * ϕ_t

            # Update direction vector 
            d_next = (w_t / β_t - K[0, 0] * d_t - ϵ_t * d_previous) / gamma_2
            
            # Update solution
            x_t = x_t + τ_t * d_next

            # Update residuals
            r_carrot = (s_t**2) * r_carrot - (ϕ_t * c_t * z_next) / β_next
            r_hat = (s_t**2) * r_hat - (ϕ_t * c_t * w_next) / β_next

            # Update  variables for next iteration
            z_previous = z_t
            z_t = z_next
            w_t = w_next
            β_previous = β_t
            β_t = β_next
            d_previous = d_t
            d_t = d_next
            δ_t = K[0, 0]
            ϵ_t = ϵ_next
        else:
            break

    return x_t

# Solve and verify
x = MINRES(3, A, M, b)

# Compute exact solution for comparison
x_exact = jnp.linalg.solve(A, b)

print("\nMINRES Test Results:")
print("-------------------")
print(f"MINRES solution x: {x.flatten()}")
print(f"Exact solution  : {x_exact.flatten()}")
print(f"Ax: {jnp.matmul(A, x).flatten()}")
print(f"b : {b.flatten()}")
print(f"Absolute Error vs exact: {jnp.linalg.norm(x - x_exact):.2e}")
print(f"Residual norm (|Ax-b|): {jnp.linalg.norm(jnp.matmul(A, x) - b):.2e}")
