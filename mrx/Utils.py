import jax
import jax.numpy as jnp

def jacobian(f):
    return lambda x: jnp.linalg.det(jax.jacfwd(f)(x))

def inv33(mat):
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    det = m1 * (m5 * m9 - m6 * m8) + m4 * (m8 * m3 - m2 * m9) + m7 * (m2 * m6 - m3 * m5)
    return (
        jnp.array([
            [m5 * m9 - m6 * m8, m3 * m8 - m2 * m9, m2 * m6 - m3 * m5],
            [m6 * m7 - m4 * m9, m1 * m9 - m3 * m7, m3 * m4 - m1 * m6],
            [m4 * m8 - m5 * m7, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4],
        ]) / det
    )
    
def div(F):
    def div_F(x):
        DF = jax.jacfwd(F)(x)
        return jnp.trace(DF) * jnp.ones(1)
    return div_F

def curl(F):
    def curl_F(x):
        DF = jax.jacfwd(F)(x)
        return jnp.array([  DF[2, 1] - DF[1, 2], 
                            DF[0, 2] - DF[2, 0], 
                            DF[1, 0] - DF[0, 1] ])
    return curl_F

def grad(F):
    def grad_F(x):
        DF = jax.jacfwd(F)(x)
        return jnp.ravel(DF)
    return grad_F

def l2_product(f, g, Q, F=lambda x: x):
    Jj = jax.vmap(jacobian(F))(Q.x)
    return jnp.einsum("ij,ij,i,i->", jax.vmap(f)(Q.x), jax.vmap(g)(Q.x), Jj, Q.w)