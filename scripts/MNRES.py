import jax
import jax.numpy as jnp

# This is an implementation of the MINRES algorithm

# M is the preconditioner
# A is the main matrix
# b is the right side
# n is the dimension 
# The algorithm is based on the paper "Obtaining Pseudo-inverse Solutions With MINRES"


def MINRES(n, A, M,b):
    # Initialization
    r_carrot = b
    z_t = b
    w_t = jnp.matmul(M,z_t)
    β_t = jnp.sqrt(jnp.matmul(jnp.transpose(z_t),w_t))
    ϕ_t = β_t
    r_hat = w_t
    c_t = jnp.float32(-1)
    s_t = jnp.float32(0.0)
    δ_t = jnp.float32(0.0)
    z_t = jnp.zeros((n,1))
    x_t = jnp.zeros((n,1))
    d_t = jnp.zeros((n,1))
    d_previous = jnp.zeros((n,1))
    β_previous =jnp.float32(1)
    z_previous = b
    t = 1
    ϵ_t = 1 # check this
    while jnp.linalg.norm(jnp.matmul(A,x_t)-b)>10**(-2):
        q_t = jnp.matmul(A,w_t)/β_t
        α_t= jnp.float32(jnp.matmul(jnp.transpose(w_t)/β_t,q_t))
        z_t = q_t-(α_t*z_t/β_t)-(β_t*z_previous/β_previous)
        w_t =  jnp.matmul(M,z_t)
        β_t = jnp.sqrt(jnp.matmul(jnp.transpose(z_t),w_t))
        B = jnp.array([[α_t[0],β_t[0]]]).reshape((1, 2))
        α_t = B[0,0]
        β_t  = B[0,1]
        K = jnp.array([[c_t*δ_t + s_t*α_t, s_t*β_t], [s_t*δ_t-c_t*α_t, -c_t*β_t]])
        #K = jnp.array([[1, 0], [0, 1]])

        #K = jnp.matmul(jnp.array([[c_t,s_t],[s_t,-c_t]]),jnp.array([[δ_t,0],[α_t, β_t]]))
        #K = jnp.matmul(jnp.array([[c,s],[s,-c]]),jnp.array([[1,0],[0,1]]))
        #K = jnp.matmul(jnp.array([[c,s],[s,-c]]),jnp.array([[δ,0],[α_t, β_t]]))

        gamma_2 = jnp.sqrt((K[2,1])**2 + β_t**2)

        print(gamma_2)

        if gamma_2 != 0:
            c_t = K[2,1]/gamma_2
            s_t = β_t/gamma_2
            τ_t = c_t*ϕ_t
            ϕ_t = s_t*ϕ_t
            d_t = (w_t/β_t -K[1,1]*d_t-ϵ_t*d_previous)/gamma_2
            x_t = x_t + τ_t*d_t
            print(x_t)
            r_carrot = (s_t**2)*r_carrot-(ϕ_t*c_t*z_t)/β_t

            if β_t!= 0:
                r_hat = (s_t**2)*r_hat-(ϕ_t*c_t*w_t)/β_t
            else:
                r_hat = jnp.zeros((n,1))
                return x_t
            
        else:
            c_t = 0
            τ_t = 0
            s_t = 1
            ϕ_t = ϕ_t
            r_hat = r_hat
            r_carrot = r_carrot
            x_t = x_t
            return x_t

        t = t+1

    
#print(MINRES(n, A, M,b))