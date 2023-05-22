from jax import random
from typing import Tuple
from jax import numpy as jnp, vmap


def uvarc(
    beta: float, a: float, b: float, c: float, d: float, e: float, f: float
) -> float:
    cb = jnp.cos(beta)
    sb = jnp.sin(beta)
    r0 = a + b * cb + c * sb
    i0 = d + e * cb + f * sb
    r1 = a - b * cb - c * sb
    i1 = d - e * cb - f * sb
    return -1 + jnp.sqrt(r0**2 + i0**2) + jnp.sqrt(r1**2 + i1**2)


def _cos2_beta_poly_coeffs(
    a: float, b: float, c: float, d: float, e: float
) -> Tuple[float, float, float, float, float, float]:
    # Coefficients of the polynomial in cos^2(beta) in descending power order
    return [
        256 * a**2 * b**2 * d**2
        + 256 * a**2 * b**2 * e**2
        + 256 * a**2 * c**2 * d**2
        + 256 * a**2 * c**2 * e**2
        - 64 * a * b**4 * d
        - 128 * a * b**3 * c * e
        - 512 * a * b**2 * d**3
        - 512 * a * b**2 * d * e**2
        - 128 * a * b * c**3 * e
        - 1024 * a * b * c * d**2 * e
        - 1024 * a * b * c * e**3
        + 64 * a * c**4 * d
        + 512 * a * c**2 * d**3
        + 512 * a * c**2 * d * e**2
        + 4 * b**6
        + 12 * b**4 * c**2
        + 64 * b**4 * d**2
        - 64 * b**4 * e**2
        + 512 * b**3 * c * d * e
        + 12 * b**2 * c**4
        - 384 * b**2 * c**2 * d**2
        + 384 * b**2 * c**2 * e**2
        + 256 * b**2 * d**4
        + 512 * b**2 * d**2 * e**2
        + 256 * b**2 * e**4
        - 512 * b * c**3 * d * e
        + 4 * c**6
        + 64 * c**4 * d**2
        - 64 * c**4 * e**2
        + 256 * c**2 * d**4
        + 512 * c**2 * d**2 * e**2
        + 256 * c**2 * e**4,
        -512 * a**2 * b**2 * d**2
        - 512 * a**2 * b**2 * e**2
        - 256 * a**2 * c**2 * d**2
        - 256 * a**2 * c**2 * e**2
        + 128 * a * b**4 * d
        + 256 * a * b**3 * c * e
        - 64 * a * b**2 * c**2 * d
        + 1024 * a * b**2 * d**3
        + 1024 * a * b**2 * d * e**2
        + 128 * a * b * c**3 * e
        + 1536 * a * b * c * d**2 * e
        + 1536 * a * b * c * e**3
        - 64 * a * c**4 * d
        - 512 * a * c**2 * d**3
        - 512 * a * c**2 * d * e**2
        - 8 * b**6
        - 20 * b**4 * c**2
        - 128 * b**4 * d**2
        + 96 * b**4 * e**2
        - 832 * b**3 * c * d * e
        - 16 * b**2 * c**4
        + 576 * b**2 * c**2 * d**2
        - 576 * b**2 * c**2 * e**2
        - 512 * b**2 * d**4
        - 768 * b**2 * d**2 * e**2
        - 256 * b**2 * e**4
        + 704 * b * c**3 * d * e
        - 512 * b * c * d**3 * e
        - 512 * b * c * d * e**3
        - 4 * c**6
        - 64 * c**4 * d**2
        + 96 * c**4 * e**2
        - 256 * c**2 * d**4
        - 768 * c**2 * d**2 * e**2
        - 512 * c**2 * e**4,
        256 * a**2 * b**2 * d**2
        + 320 * a**2 * b**2 * e**2
        + 64 * a**2 * c**2 * e**2
        - 64 * a * b**4 * d
        - 160 * a * b**3 * c * e
        + 64 * a * b**2 * c**2 * d
        - 512 * a * b**2 * d**3
        - 640 * a * b**2 * d * e**2
        - 32 * a * b * c**3 * e
        - 512 * a * b * c * d**2 * e
        - 768 * a * b * c * e**3
        + 128 * a * c**2 * d * e**2
        + 4 * b**6
        + 12 * b**4 * c**2
        + 64 * b**4 * d**2
        - 32 * b**4 * e**2
        + 352 * b**3 * c * d * e
        + 8 * b**2 * c**4
        - 192 * b**2 * c**2 * d**2
        + 256 * b**2 * c**2 * e**2
        + 256 * b**2 * d**4
        + 320 * b**2 * d**2 * e**2
        + 64 * b**2 * e**4
        - 224 * b * c**3 * d * e
        + 512 * b * c * d**3 * e
        + 512 * b * c * d * e**3
        - 32 * c**4 * e**2
        + 320 * c**2 * d**2 * e**2
        + 320 * c**2 * e**4,
        -64 * a**2 * b**2 * e**2
        + 32 * a * b**3 * c * e
        + 128 * a * b**2 * d * e**2
        + 128 * a * b * c * e**3
        - 4 * b**4 * c**2
        - 32 * b**3 * c * d * e
        - 32 * b**2 * c**2 * e**2
        - 64 * b**2 * d**2 * e**2
        - 128 * b * c * d * e**3
        - 64 * c**2 * e**4,
    ]


def fplus_coeffs(
    abar: float, bbar: float, cbar: float, dbar: float, ebar: float, fbar: float
) -> Tuple[float, float, float, float, float]:
    a = (
        abar**2
        + 0.5 * bbar**2
        + 0.5 * cbar**2
        + dbar**2
        + 0.5 * ebar**2
        + 0.5 * fbar**2
    )
    b = 2 * abar * bbar + 2 * dbar * ebar
    c = 2 * abar * cbar + 2 * dbar * fbar
    d = 0.5 * (bbar**2 - cbar**2) + 0.5 * (ebar**2 - fbar**2)
    e = bbar * cbar + ebar * fbar
    return a, b, c, d, e


def maximise_expected_circular_variance(
    abar: float, bbar: float, cbar: float, dbar: float, ebar: float, fbar: float
) -> Tuple[float, float]:
    """
    Return argmax and max of
    U(b) = -1
            + |(a̅ + b̅ cos(b) + c̅ sin(b)) + i(d̅ + e̅ cos(b) + f̅ sin(b))|
            + |(a̅ - b̅ cos(b) - c̅ sin(b)) + i(d̅ - e̅ cos(b) - f̅ sin(b))|

    Args:
        abar: a̅
        bbar: b̅
        cbar: c̅
        dbar: d̅
        ebar: e̅
        fbar: f̅

    Returns:
        Tuple of argmax in [0, π) and max in [0, ∞) of U(b)

    """
    a, b, c, d, e = fplus_coeffs(abar, bbar, cbar, dbar, ebar, fbar)

    cos2_beta_coeffs = _cos2_beta_poly_coeffs(a, b, c, d, e)

    cos2_beta_roots = jnp.roots(jnp.array(cos2_beta_coeffs), strip_zeros=False)
    cos2_beta_roots = jnp.where(jnp.isreal(cos2_beta_roots), cos2_beta_roots.real, 0.0)
    betas_plus = jnp.arccos(jnp.sqrt(cos2_beta_roots)).real
    betas_min = jnp.arccos(-jnp.sqrt(cos2_beta_roots)).real
    betas = jnp.concatenate([betas_plus, betas_min])
    betas = jnp.where(jnp.isnan(betas), 0.0, betas)
    evals = vmap(lambda beta: uvarc(beta, abar, bbar, cbar, dbar, ebar, fbar))(betas)

    max_ind = evals.argmax()
    return betas[max_ind], evals[max_ind]
