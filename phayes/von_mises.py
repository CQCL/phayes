from typing import Tuple, Union, Callable
from jax import numpy as jnp, vmap
from jax.lax import fori_loop
from jax.scipy import special
from tensorflow_probability.substrates.jax.math import bessel_ive
from .maximise import maximise_expected_circular_variance

kappa_max = 500


def bessel_ratio(kappa: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    return jnp.where(
        kappa < kappa_max, special.i1e(kappa) / special.i0e(kappa), 1 - 0.5 / kappa
    )


def _inverse_bessel_ratio_small(
    r: Union[float, jnp.ndarray]
) -> Union[float, jnp.ndarray]:
    return (
        2 * r
        - r**3
        - 1 / 6 * r**5
        - 1 / 24 * r**7
        + 1 / 360 * r**9
        + 53 / 2160 * r**11
    ) / (1 - r**2)


def _inverse_bessel_ratio_large(
    r: Union[float, jnp.ndarray]
) -> Union[float, jnp.ndarray]:
    return 1 / (2 * (1 - r) - (1 - r) ** 2 - (1 - r) ** 3)


def inverse_bessel_ratio(r: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    # https://dl.acm.org/doi/pdf/10.1145/355945.355949
    ibr = jnp.where(
        r < 0.8, _inverse_bessel_ratio_small(r), _inverse_bessel_ratio_large(r)
    )
    return jnp.where(r < bessel_ratio(kappa_max), ibr, 0.5 / (1 - r))


def _circular_mean_to_von_mises(circular_m1: complex) -> Tuple[float, float]:
    circular_mean = jnp.angle(circular_m1)
    one_minus_circular_var = jnp.abs(circular_m1)
    approximate_kappa = inverse_bessel_ratio(one_minus_circular_var)
    return circular_mean, approximate_kappa


def fourier_to_von_mises(
    fourier_coefficients: jnp.ndarray, deflation: float = 1.0
) -> Tuple[float, float]:
    """
    Approximate a Fourier series distribution with a von Mises by circular moment matching.
    Find vonMises(phi | mu, kappa) ∝ exp(kappa * cos(phi - mu)) such that the first circular moment
    is equivalent to the Fourier series distribution.

    The first circular moment is defined as m1 = E[exp(i*phi)] = E[cos(phi) + i * sin(phi)]
    For Fourier series, we have m1=π(c1 + i * s1) where c1 and s1 are the first cosine and sine coefficients
    For vonMises(phi | mu, kappa), we have m1=A(kappa)(cos(mu) + i*sin(mu)),
        where A(kappa)=I_1(kappa)/I_0(kappa) is a ratio of modified Bessel functions.

    Args:
        fourier_coefficients: Array of shape (2, J) storing Fourier coefficients.
            First row for cosine, second for sine.
        deflation: Deflation parameter in (0, 1], for conversion from Fourier to von Mises
            (reduce kappa to deflation * kappa to account for approximation in conversion)


    Returns:
        Tuple of floats, mean mu in [0, 2π) and concentration kappa in [0, ∞)
    """
    circular_m1 = jnp.pi * (
        fourier_coefficients[0, 0] + 1j * fourier_coefficients[1, 0]
    )
    mu, kappa = _circular_mean_to_von_mises(circular_m1)
    return mu, deflation * kappa


def von_mises_to_fourier(mu: float, kappa: float, J: int) -> jnp.ndarray:
    """
    Convert von Mises distribution into Fourier series representation.

    Useful reference: https://quantum-journal.org/papers/q-2021-06-07-469/pdf/

    Args:
        mu: von Mises mean parameter, float in [0, 2π)
        kappa: von Mises concentration parameter, float in [0, ∞)
        J: number of terms in output Fourier series

    Returns:
        Array of shape (2, J) storing Fourier coefficients.
            First row for cosine, second for sine.
    """
    js = jnp.arange(1, J + 1)
    # in0k = bessel_ive(list(range(1, J + 1)), kappa) / bessel_ive(0, kappa)
    in0k = bessel_ive(jnp.arange(1, J + 1, dtype=float), kappa) / bessel_ive(0, kappa)

    cs = in0k * jnp.cos(js * mu) / jnp.pi
    ss = in0k * jnp.sin(js * mu) / jnp.pi
    return jnp.row_stack((cs, ss))


def von_mises_update(
    mu: float,
    kappa: float,
    m: Union[int, jnp.ndarray],
    k: Union[int, jnp.ndarray],
    beta: Union[float, jnp.ndarray],
    error_rate: Union[float, Callable[[int], float]] = 0.0,
) -> Tuple[float, float]:
    """
    Applies a posterior update based on the measurements of a phase estimation experiment.
    Assumes prior is a von Mises distribution with parameters mu and kappa.
    Ouputs a parameters of a von Mises distribtion approximating the posterior.

    p(phi | m) ∝ p(phi) * \prod_r^R (1 + (1-error_rate)*cos(k_r * phi + beta_r - m_r * pi)) / 2

    Prior p(phi) and posterior p(phi | m) are represented by a von Mises distribution
    p(phi) = exp(kappa * cos(phi - mu)) / (2 * pi * I_0(kappa))

    Args:
        mu: Prior von Mises mean parameter, float in [0, 2π)
        kappa: Prior von Mises concentration parameter, float in [0, ∞)
        m: Vector of measured bits
        k: Vector of integer exponents (or single integer if the same across measurements)
        beta: Vector of phase shifts in [0, 2π) (or single float if the same across measurements)
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k

    Returns:
        Tuple of floats, posterior mean mu in [0, 2π)
        and posterior concentration kappa in [0, ∞)

    """
    m = jnp.atleast_1d(jnp.array(m, dtype=int))
    k = jnp.array(k) * jnp.ones(m.size, dtype=float)
    beta = jnp.array(beta) * jnp.ones(m.size)

    if callable(error_rate):
        error_rate = vmap(error_rate)(k)
    error_rate = jnp.array(error_rate) * jnp.ones(m.size)

    kappa = jnp.array(kappa, dtype=float)
    posterior_mu, posterior_kappa = fori_loop(
        0,
        m.size,
        lambda r, mu_and_kappa: _update_single_von_mises(
            *mu_and_kappa, m[r], k[r], beta[r], error_rate[r]
        ),
        (mu, kappa),
    )
    return posterior_mu, posterior_kappa


def _update_single_von_mises(
    mu: float, kappa: float, m: int, k: int, beta: float, error_rate: float
) -> Tuple[float, float]:
    gamma_shift = beta + k * mu + m * jnp.pi
    cos_gamma_shift = jnp.cos(gamma_shift) * (1 - error_rate)
    sin_gamma_shift = jnp.sin(gamma_shift) * (1 - error_rate)

    I0 = special.i0e(kappa)

    k = jnp.array(k, dtype=float)

    norm_const = (1 + cos_gamma_shift * bessel_ive(k, kappa) / I0) / 2

    A_km1 = bessel_ive(k - 1, kappa) / I0
    A_kp1 = bessel_ive(k + 1, kappa) / I0

    m1 = (
        bessel_ratio(kappa)
        + 0.5 * cos_gamma_shift * (A_km1 + A_kp1)
        + 1.0j * 0.5 * sin_gamma_shift * (A_kp1 - A_km1)
    ) / (norm_const * 2)

    posterior_mu_shift, posterior_kappa = _circular_mean_to_von_mises(m1)

    posterior_kappa = jnp.where(posterior_kappa < 0, kappa, posterior_kappa)
    return posterior_mu_shift + mu, posterior_kappa


def von_mises_circular_m1(mu: float, kappa: float) -> complex:
    """
    First circular moment of a von Mises distribution.
    Equal to E[exp(i * phi)] = exp(i * mu) * I1(kappa) / I0(kappa),
    where I1 and I0 are the modified Bessel functions of the first kind.

    Args:
        mu: Prior von Mises mean parameter, float in [0, 2π)
        kappa: von Mises concentration parameter, float in [0, ∞)

    Returns:
        First circular moment - complex
    """
    return bessel_ratio(kappa) * jnp.exp(1j * mu)


def von_mises_circular_variance(kappa: float) -> float:
    """
    Circular variance of a von Mises distribution.
    Equal to 1 - |E[exp(i * phi)]| = 1 - I1(kappa) / I0(kappa),
    where I1 and I0 are the modified Bessel functions of the first kind.
    Note that the circular variance is different to the traditional variance, but is more natural
    for wrapped distributions.
    For sharply peaked distributions we have that 2 * circular_variance = standard variance.

    Args:
        kappa: von Mises concentration parameter, float in [0, ∞)

    Returns:
        Circular variance - float in [0, 1]
    """
    return 1 - bessel_ratio(kappa)


def von_mises_holevo_variance(
    kappa: float,
) -> float:
    """
    Holevo phase variance of a von Mises distribution.
    Equal to |E[exp(i * phi)]|^-2 - 1 = (I1(kappa) / I0(kappa))^-2 - 1,
    where I1 and I0 are the modified Bessel functions of the first kind.
    The Holevo phase variance is different to the traditional variance however they are
    equivalent for sharply peaked distributions, see https://www.dominicberry.org/research/thesis.pdf.

    Args:
        kappa: von Mises concentration parameter, float in [0, ∞)

    Returns:
        Holevo phase variance - float in [0, ∞)
    """
    return bessel_ratio(kappa) ** -2 - 1


def von_mises_cosine_distance(
    val: Union[float, jnp.ndarray],
    mu: float,
    kappa: float,
) -> Union[float, jnp.ndarray]:
    """
    Evaluates E[1 - cos(phi - val)] where the expectation value is taken
    with respect the von Mises distribution.

    Args:
        val: Float or vector of values for pdf to be evaluated at
        mu: Prior von Mises mean parameter, float in [0, 2π)
        kappa: Prior von Mises concentration parameter, float in [0, ∞)

    Returns:
        Float or vector of expected cosine distances, same shape as val
    """
    val = jnp.atleast_1d(val)
    A = bessel_ratio(kappa)
    return jnp.squeeze(
        1 - A * (jnp.cos(mu) * jnp.cos(val) + jnp.sin(mu) * jnp.sin(val))
    )


def von_mises_entropy(kappa: float) -> float:
    """
    Evaluates H(mu, kappa) = - kappa * I1(kappa) / I0(kappa) + log(2 * pi * I0(kappa)),
    the entropy of a von Mises distribution.

    Args:
        mu: Prior von Mises mean parameter, float in [0, 2π)
        kappa: Prior von Mises concentration parameter, float in [0, ∞)

    Returns:
        float in [0, ∞)

    """
    return (
        -kappa * bessel_ratio(kappa) + jnp.log(2 * jnp.pi * special.i0e(kappa)) + kappa
    )


def von_mises_pdf(
    val: Union[float, jnp.ndarray],
    mu: float,
    kappa: float,
) -> Union[float, jnp.ndarray]:
    """
    Evaluates P(val) where P is the pdf defined by fourier_coefficients

    Args:
        val: Float or vector of values for pdf to be evaluated at
        mu: Prior von Mises mean parameter, float in [0, 2π)
        kappa: Prior von Mises concentration parameter, float in [0, ∞)

    Returns:
        Float or vector of pdf evaluations, same shape as val
    """
    val = jnp.atleast_1d(val)
    return jnp.squeeze(
        jnp.exp(kappa * (jnp.cos(val - mu) - 1)) / (2 * jnp.pi * special.i0e(kappa))
    )


def von_mises_evidence(
    mu: jnp.ndarray,
    kappa: float,
    m: int,
    k: int,
    beta: float,
    error_rate: Union[float, Callable[[int], float]] = 0.0,
) -> float:
    """
    Evaluates p(m | k, beta) = ∫p(phi) p(m | phi, k, beta) dphi
    where p(phi) is a von Mises pdf with mean mu and concentration kappa.

    Args:
        mu: Prior von Mises mean parameter, float in [0, 2π)
        kappa: Prior von Mises concentration parameter, float in [0, ∞)
        m: Measured bit
        k: Integer exponents
        beta: Phase shifts in [0, 2π)
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k

    Returns:
        float value in [0, ∞) for p(m | k, beta)

    """
    k = jnp.array(k, dtype=float)
    if callable(error_rate):
        error_rate = error_rate(k)
    Ak = bessel_ive(k, kappa) / bessel_ive(0, kappa)
    return 0.5 + 0.5 * Ak * jnp.cos(k * mu + beta - m * jnp.pi) * (1 - error_rate)


def _posterior_m1_coeffs(
    mu: jnp.ndarray, kappa: float, k: int, error_rate: float
) -> Tuple[float, float, float, float]:
    k = jnp.array(k, dtype=float)
    A1 = bessel_ratio(kappa)
    Akmin1 = bessel_ive(k - 1, kappa) / bessel_ive(0, kappa)
    Akplus1 = bessel_ive(k + 1, kappa) / bessel_ive(0, kappa)

    c1 = jnp.cos(mu)
    s1 = jnp.sin(mu)
    ckmin1 = jnp.cos((k - 1) * mu)
    ckplus1 = jnp.cos((k + 1) * mu)
    skmin1 = jnp.sin((k - 1) * mu)
    skplus1 = jnp.sin((k + 1) * mu)

    success_rate = 1 - error_rate

    abar = A1 * c1 / 2
    bbar = (Akmin1 * ckmin1 + Akplus1 * ckplus1) / 4 * success_rate
    cbar = (-Akmin1 * skmin1 - Akplus1 * skplus1) / 4 * success_rate

    dbar = A1 * s1 / 2
    ebar = (-Akmin1 * skmin1 + Akplus1 * skplus1) / 4 * success_rate
    fbar = (-Akmin1 * ckmin1 + Akplus1 * ckplus1) / 4 * success_rate
    return abar, bbar, cbar, dbar, ebar, fbar


def von_mises_expected_posterior_circular_variance(
    mu: jnp.ndarray,
    kappa: float,
    k: int,
    beta: float,
    error_rate: Union[float, Callable[[int], float]] = 0.0,
) -> float:
    """
    Calculates expected circular variance of the single update posterior distribution.

    E[var_C(p(phi | m, k, beta))] = Σ_m p(m | k, beta) var_C(p(phi | m, k, beta))

    Args:
        mu: Prior von Mises mean parameter, float in [0, 2π)
        kappa: Prior von Mises concentration parameter, float in [0, ∞)
        k: Integer exponent
        beta: Phase shift in [0, 2π)
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k

    Returns:
        float value in [0, 1]

    """
    if callable(error_rate):
        error_rate = error_rate(k)

    abar, bbar, cbar, dbar, ebar, fbar = _posterior_m1_coeffs(mu, kappa, k, error_rate)

    cb = jnp.cos(beta)
    sb = jnp.sin(beta)

    M10p0 = jnp.abs(
        (abar + bbar * cb + cbar * sb) + 1.0j * (dbar + ebar * cb + fbar * sb)
    )
    M11p1 = jnp.abs(
        (abar - bbar * cb - cbar * sb) + 1.0j * (dbar - ebar * cb - fbar * sb)
    )
    return 1 - M10p0 - M11p1


def von_mises_expected_posterior_holevo_variance(
    mu: jnp.ndarray,
    kappa: float,
    k: int,
    beta: float,
    error_rate: Union[float, Callable[[int], float]] = 0.0,
) -> float:
    """
    Calculates expected Holevo variance of the single update posterior distribution.

    E[var_H(p(phi | m, k, beta))] = Σ_m p(m | k, beta) var_H(p(phi | m, k, beta))

    Args:
        mu: Prior von Mises mean parameter, float in [0, 2π)
        kappa: Prior von Mises concentration parameter, float in [0, ∞)
        k: Integer exponent
        beta: Phase shift in [0, 2π)
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k

    Returns:
        float value in [0, ∞)

    """
    if callable(error_rate):
        error_rate = error_rate(k)

    abar, bbar, cbar, dbar, ebar, fbar = _posterior_m1_coeffs(mu, kappa, k, error_rate)

    cb = jnp.cos(beta)
    sb = jnp.sin(beta)

    M10e0 = jnp.abs(
        (abar + bbar * cb + cbar * sb) + 1.0j * (dbar + ebar * cb + fbar * sb)
    )
    M11e1 = jnp.abs(
        (abar - bbar * cb - cbar * sb) + 1.0j * (dbar - ebar * cb - fbar * sb)
    )

    e0 = von_mises_evidence(mu, kappa, 0, k, beta, error_rate)
    e1 = von_mises_evidence(mu, kappa, 1, k, beta, error_rate)

    v0 = (M10e0 / e0) ** -2 - 1
    v1 = (M11e1 / e1) ** -2 - 1
    return v0 * e0 + v1 * e1


def von_mises_get_beta_given_k(
    mu: jnp.ndarray,
    kappa: float,
    k: int,
    error_rate: Union[float, Callable[[int], float]] = 0.0,
) -> Tuple[float, float]:
    """
    Calculates beta that minimises the expected circular variance of a single update, for a given k.

    Args:
        mu: Prior von Mises mean parameter, float in [0, 2π)
        kappa: Prior von Mises concentration parameter, float in [0, ∞)
        k: Integer exponent
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k

    Returns:
        Float beta in [0, π) and its corresponding expected negative circular variance

    """
    if callable(error_rate):
        error_rate = error_rate(k)

    abar, bbar, cbar, dbar, ebar, fbar = _posterior_m1_coeffs(mu, kappa, k, error_rate)
    return maximise_expected_circular_variance(abar, bbar, cbar, dbar, ebar, fbar)


def von_mises_get_k_and_beta(
    mu: jnp.ndarray,
    kappa: float,
    error_rate: Union[float, Callable[[int], float]] = 0.0,
    k_max: int = jnp.inf,
) -> Tuple[int, float]:
    """
    Calculates k and beta that minimise the expected circular variance of a single update.

    Args:
        mu: Prior von Mises mean parameter, float in [0, 2π)
        kappa: Prior von Mises concentration parameter, float in [0, ∞)
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k
        k_max: Maximum k to consider

    Returns:
        Tuple of (k, beta) with k in {1, ..., k_max} and beta in [0, π)

    """
    if callable(error_rate):
        error_rate_func = error_rate
    else:
        error_rate_func = lambda _: error_rate

    k_mid = 1 / jnp.sqrt(von_mises_holevo_variance(kappa))
    k_l = jnp.floor(k_mid).astype(float)
    k_u = k_l + 1.0

    beta_l, ecv_l = von_mises_get_beta_given_k(mu, kappa, k_l, error_rate_func(k_l))
    beta_u, ecv_u = von_mises_get_beta_given_k(mu, kappa, k_u, error_rate_func(k_u))

    k_out = jnp.where(ecv_l > ecv_u, k_l, k_u)
    beta_out = jnp.where(ecv_l > ecv_u, beta_l, beta_u)

    k_out = jnp.where(k_out > k_max, k_max, k_out).astype(int)
    beta_out = von_mises_get_beta_given_k(mu, kappa, k_out, error_rate_func(k_out))[0]
    return k_out, beta_out
