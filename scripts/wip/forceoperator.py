# %%
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Flat, Pushforward
from mrx.LazyMatrices import LazyDerivativeMatrix, LazyMassMatrix, LazyProjectionMatrix
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.Projectors import CurlProjection, EFieldProjector, Projector
from mrx.Quadrature import QuadratureRule
from mrx.Utils import inv33, jacobian_determinant

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

n = 6
p = 3

ns = (n, n, 1)
ps = (p, p, 0)
types = ('clamped', 'clamped', 'constant')
bcs = ('dirichlet', 'dirichlet', 'none')
quad_order = 4

B0 = 1
a = 1
R0 = 3
q0 = 1
q1 = 1
beta = 0


def F(r):
    x, y, z = r
    return jnp.array([a * x, 2 * jnp.pi * a * y, 2 * jnp.pi * R0 * z])


def q(x):
    return q0 + (q1 - q0) * x**2 / a**2


def B(r):
    x, y, z = r
    return B0 * jnp.array([0.0, a/q(x)/R0, 1.0])


def p(r):
    x, y, z = r
    return beta * B0**2 / 2 * (1 + (a/q(x)/R0)**2) + (B0 * a / R0)**2 * (1/q0**2 - 1/q(x)**2)
