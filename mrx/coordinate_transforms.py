import jax.numpy as jnp

__all__ = [
    # Polar coordinates
    "cart_to_pol",
    "vector_cart_to_pol",
    "pol_to_cart",
    "vector_pol_to_cart",
    # Cylindrical coordinates
    "cart_to_cyl",
    "vector_cart_to_cyl",
    "cyl_to_cart",
    "vector_cyl_to_cart",
    # Tokamak coordinates
    "get_tok_to_cyl",
    "get_vector_tok_to_cyl",
    "get_cyl_to_tok",
    "get_vector_cyl_to_tok",
    "get_tok_to_cart",
    "get_cart_to_tok",
    "get_vector_tok_to_cart",
    "get_vector_cart_to_tok",
]

### Polar coordinates

def cart_to_pol(x):
    r = jnp.sqrt(x[0]**2 + x[1]**2)
    θ = jnp.arctan2(x[1], x[0])
    return jnp.array([r, θ])

def vector_cart_to_pol(v, x):
    r, θ = cart_to_pol(x)
    v1 = v[0]*jnp.cos(θ) + v[1]*jnp.sin(θ)
    v2 = -v[0]*jnp.sin(θ) + v[1]*jnp.cos(θ)
    return jnp.array([v1, v2])

def pol_to_cart(x):
    r, θ = x
    return jnp.array([r*jnp.cos(θ), r*jnp.sin(θ)])

def vector_pol_to_cart(v, x):
    x1, x2 = pol_to_cart(x)
    r, θ = x
    v1 = (jnp.cos(θ) * v[0] - jnp.sin(θ) * v[1])
    v2 = (jnp.sin(θ) * v[0] + jnp.cos(θ) * v[1])
    return jnp.array([v1, v2])

### Cylindrical coordinates

def cart_to_cyl(x):
    r, θ = cart_to_pol(x[:2])
    z = x[2]
    return jnp.array([r, θ, z])

def vector_cart_to_cyl(v, x):
    v1, v2 = vector_cart_to_pol(v[:2], x[:2])
    v3 = v[2]
    return jnp.array([v1, v2, v3])

def cyl_to_cart(x):
    x1, x2 = pol_to_cart(x[:2])
    return jnp.array([x1, x2, x[2]])

def vector_cyl_to_cart(v, x):
    v1, v2 = vector_pol_to_cart(v[:2], x[:2])
    v3 = v[2]
    return jnp.array([v1, v2, v3])

### Tokamak coordinates

def get_tok_to_cyl(R0):
    def tok_to_cyl(x):
        r, θ, z = x
        R = R0 + r*jnp.cos(θ)
        Z = r*jnp.sin(θ)
        phi = -z/R0
        return jnp.array([R, phi, Z])
    return tok_to_cyl

def get_vector_tok_to_cyl(R0):
    def vector_tok_to_cyl(v, x):
        R, phi, Z = get_tok_to_cyl(R0)(x)
        r, θ, z = x
        vr, vθ, vz = v
        vR = vr * jnp.cos(θ) - vθ * jnp.sin(θ)
        vZ = vr * jnp.sin(θ) + vθ * jnp.cos(θ)
        vphi = - vz
        return jnp.array([vR, vphi, vZ])
    return vector_tok_to_cyl

def get_cyl_to_tok(R0):
    def cyl_to_tok(x):
        R, phi, Z = x
        r = jnp.sqrt((R - R0)**2 + Z**2)
        θ = jnp.arctan2(Z, R - R0)
        z = - R0 * phi
        return jnp.array([r, θ, z])
    return cyl_to_tok

def get_vector_cyl_to_tok(R0):
    def vector_cyl_to_tok(v, x):
        r, θ, z = get_cyl_to_tok(R0)(x)
        R, phi, Z = x
        vR, vphi, vZ = v
        vr = jnp.cos(θ) * vR + jnp.sin(θ) * vZ
        vθ = jnp.cos(θ) * vZ - jnp.sin(θ) * vR
        vz = - vphi
        return jnp.array([vr, vθ, vz])
    return vector_cyl_to_tok

def get_tok_to_cart(R0):
    return lambda x: cyl_to_cart(get_tok_to_cyl(R0)(x))

def get_cart_to_tok(R0):
    return lambda x: get_cyl_to_tok(R0)(cart_to_cyl(x))

def get_vector_tok_to_cart(R0):
    def vector_tok_to_cart(v, x):
        v_cyl = get_vector_tok_to_cyl(R0)(v, x)
        x_cyl = get_tok_to_cyl(R0)(x)
        return vector_cyl_to_cart(v_cyl, x_cyl)
    return vector_tok_to_cart

def get_vector_cart_to_tok(R0):
    def vector_cart_to_tok(v, x):
        v_cyl = vector_cart_to_cyl(v, x)
        x_cyl = cart_to_cyl(x)
        return get_vector_cyl_to_tok(R0)(v_cyl, x_cyl)
    return vector_cart_to_tok