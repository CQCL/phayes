from typing import Tuple, Union, Callable
from jax import numpy as jnp, vmap
from jax.lax import fori_loop
from .maximise import maximise_expected_circular_variance


def fourier_update(
    prior_fourier_coefficients: jnp.ndarray,
    m: Union[int, jnp.ndarray],
    k: Union[int, jnp.ndarray],
    beta: Union[float, jnp.ndarray],
    error_rate: Union[float, Callable[[int], float]] = 0.0,
) -> jnp.ndarray:
    """
    Applies a posterior update based on the measurements of a phase estimation experiment.

    p(phi | m) ∝ p(phi) * \prod_r^R (1 + (1-error_rate) * cos(k_r * phi + beta_r - m_r * pi)) / 2

    Prior p(phi) and posterior p(phi | m) are stored as a Fourier series
    p(phi) = 1/(2π) + \sum_j^J c_j * cos(j * phi) + s_j * sin(j * phi)

    The distribution is therefore stored as an array of coefficients c_j and s_j.
    The exact update may lead to terms beyond the series length J, here these terms
    are ignored which introduces a bias.

    Reference: https://iopscience.iop.org/article/10.1088/1367-2630/aafb8e/pdf

    Args:
        prior_fourier_coefficients: Array of shape (2, J) storing prior coefficients
            First row for cosine, second for sine.
        m: Vector of measured bits
        k: Vector of integer exponents (or single integer if the same across measurements)
        beta: Vector of phase shifts in [0, 2π) (or single float if the same across measurements)
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k

    Returns:
        posterior Fourier coefficients: Array of shape (2, J)

    """
    m = jnp.atleast_1d(jnp.array(m, dtype=int))
    k = jnp.array(k) * jnp.ones(m.size, dtype=int)
    beta = jnp.array(beta) * jnp.ones(m.size)

    if callable(error_rate):
        error_rate = vmap(error_rate)(k)
    error_rate = jnp.array(error_rate) * jnp.ones(m.size)

    return fori_loop(
        0,
        m.size,
        lambda r, fcs: _update_single_fourier(fcs, m[r], k[r], beta[r], error_rate[r]),
        prior_fourier_coefficients,
    )


def _update_single_fourier(
    prior_fourier_coefficients: jnp.ndarray,
    m: int,
    k: int,
    beta: float,
    error_rate: float,
) -> jnp.ndarray:
    gamma = beta + m * jnp.pi
    success_rate = 1 - error_rate
    cos_gamma = jnp.cos(gamma) * success_rate
    sin_gamma = jnp.sin(gamma) * success_rate

    num_terms = prior_fourier_coefficients.shape[1]

    c0, fourier_coefficients = fori_loop(
        0,
        num_terms + 1,
        lambda j, co_fc: update_single_coeff_cos(
            jnp.where(j == 0, 1 / (2 * jnp.pi), prior_fourier_coefficients[0, j - 1]),
            co_fc[0],
            co_fc[1],
            cos_gamma,
            sin_gamma,
            k,
            j,
        ),
        (0, jnp.zeros_like(prior_fourier_coefficients)),
    )

    c0, fourier_coefficients = fori_loop(
        1,
        num_terms + 1,
        lambda j, co_fc: update_single_coeff_sin(
            prior_fourier_coefficients[1, j - 1],
            co_fc[0],
            co_fc[1],
            cos_gamma,
            sin_gamma,
            k,
            j,
        ),
        (c0, fourier_coefficients),
    )

    fourier_coefficients /= c0 * 2 * jnp.pi

    return fourier_coefficients


def update_single_coeff_cos(
    prior_cj: float,
    c0: float,
    fourier_coefficients: jnp.ndarray,
    cos_gamma: float,
    sin_gamma: float,
    k: int,
    j: int,
) -> Tuple[float, jnp.ndarray]:
    n_coeff = fourier_coefficients.shape[1]

    c0 = jnp.where(j > 0, c0, c0 + prior_cj * 0.5)
    fourier_coefficients = jnp.where(
        j > 0,
        fourier_coefficients.at[0, j - 1].add(0.5 * prior_cj),
        fourier_coefficients,
    )

    j_plus_k_bool = (0 < j + k) & (j + k <= n_coeff)
    fourier_coefficients = jnp.where(
        j_plus_k_bool,
        fourier_coefficients.at[0, j + k - 1].add(0.25 * cos_gamma * prior_cj),
        fourier_coefficients,
    )
    fourier_coefficients = jnp.where(
        j_plus_k_bool,
        fourier_coefficients.at[1, j + k - 1].add(-0.25 * sin_gamma * prior_cj),
        fourier_coefficients,
    )

    j_minus_k = j - k
    j_minus_k_pos = (0 < j_minus_k) & (j_minus_k <= n_coeff)
    fourier_coefficients = jnp.where(
        j_minus_k_pos,
        fourier_coefficients.at[0, j_minus_k - 1].add(0.25 * cos_gamma * prior_cj),
        fourier_coefficients,
    )
    fourier_coefficients = jnp.where(
        j_minus_k_pos,
        fourier_coefficients.at[1, j_minus_k - 1].add(0.25 * sin_gamma * prior_cj),
        fourier_coefficients,
    )
    c0 = jnp.where(j - k == 0, c0 + 0.25 * cos_gamma * prior_cj, c0)
    j_minus_k_nef = (0 < -j_minus_k) & (-j_minus_k <= n_coeff)
    fourier_coefficients = jnp.where(
        j_minus_k_nef,
        fourier_coefficients.at[0, -j_minus_k - 1].add(0.25 * cos_gamma * prior_cj),
        fourier_coefficients,
    )
    fourier_coefficients = jnp.where(
        j_minus_k_nef,
        fourier_coefficients.at[1, -j_minus_k - 1].add(-0.25 * sin_gamma * prior_cj),
        fourier_coefficients,
    )

    return c0, fourier_coefficients


def update_single_coeff_sin(
    prior_sj: float,
    c0: float,
    fourier_coefficients: jnp.ndarray,
    cos_gamma: float,
    sin_gamma: float,
    k: int,
    j: int,
) -> Tuple[float, jnp.ndarray]:
    n_coeff = fourier_coefficients.shape[1]

    fourier_coefficients = jnp.where(
        j > 0,
        fourier_coefficients.at[1, j - 1].add(0.5 * prior_sj),
        fourier_coefficients,
    )

    j_plus_k_bool = (0 < j + k) & (j + k <= n_coeff)
    fourier_coefficients = jnp.where(
        j_plus_k_bool,
        fourier_coefficients.at[0, j + k - 1].add(0.25 * sin_gamma * prior_sj),
        fourier_coefficients,
    )
    fourier_coefficients = jnp.where(
        j_plus_k_bool,
        fourier_coefficients.at[1, j + k - 1].add(0.25 * cos_gamma * prior_sj),
        fourier_coefficients,
    )

    j_minus_k = j - k
    j_minus_k_bool = (0 < j_minus_k) & (j_minus_k <= n_coeff)
    fourier_coefficients = jnp.where(
        j_minus_k_bool,
        fourier_coefficients.at[0, j_minus_k - 1].add(-0.25 * sin_gamma * prior_sj),
        fourier_coefficients,
    )
    fourier_coefficients = jnp.where(
        j_minus_k_bool,
        fourier_coefficients.at[1, j_minus_k - 1].add(0.25 * cos_gamma * prior_sj),
        fourier_coefficients,
    )
    c0 = jnp.where(j - k == 0, c0 - 0.25 * sin_gamma * prior_sj, c0)
    j_minus_k_nef = (0 < -j_minus_k) & (-j_minus_k <= n_coeff)
    fourier_coefficients = jnp.where(
        j_minus_k_nef,
        fourier_coefficients.at[0, -j_minus_k - 1].add(-0.25 * sin_gamma * prior_sj),
        fourier_coefficients,
    )
    fourier_coefficients = jnp.where(
        j_minus_k_nef,
        fourier_coefficients.at[1, -j_minus_k - 1].add(-0.25 * cos_gamma * prior_sj),
        fourier_coefficients,
    )

    return c0, fourier_coefficients


def _single_fourier_pdf(
    val: Union[float, jnp.ndarray], fourier_coefficients: jnp.ndarray
) -> Union[float, jnp.ndarray]:
    c0 = 1 / (2 * jnp.pi)
    n_coeff = fourier_coefficients.shape[1]
    cs, ss = fourier_coefficients
    return (
        c0
        + (
            cs * jnp.cos(jnp.arange(1, n_coeff + 1) * val)
            + ss * jnp.sin(jnp.arange(1, n_coeff + 1) * val)
        ).sum()
    )


def fourier_pdf(
    val: Union[float, jnp.ndarray], fourier_coefficients: jnp.ndarray
) -> Union[float, jnp.ndarray]:
    """
    Evaluates P(val) where P is the pdf defined by fourier_coefficients

    Args:
        val: Float or vector of values for pdf to be evaluated at
        fourier_coefficients: Array of shape (2, J) storing Fourier coefficients.
            First row for cosine, second for sine.

    Returns:
        Float or vector of pdf evaluations, same shape as val
    """
    val = jnp.atleast_1d(val)
    return jnp.squeeze(
        vmap(_single_fourier_pdf, in_axes=(0, None))(val, fourier_coefficients)
    )


def fourier_circular_m1(fourier_coefficients: jnp.ndarray) -> complex:
    """
    First circular moment of a Fourier distribution.
    Equal to E[exp(i * phi)] =  c1 + i s1.

    Args:
        fourier_coefficients: Array of shape (2, J) storing Fourier coefficients.
            First row for cosine, second for sine.

    Returns:
        First circular moment - complex
    """
    return jnp.pi * (fourier_coefficients[0, 0] + 1j * fourier_coefficients[1, 0])


def fourier_circular_mean(fourier_coefficients: jnp.ndarray) -> float:
    """
    Extracts the circular mean of the distribution represented by the inputted Fourier coefficients.
    Equal to arg(E[exp(i * phi)]) =  arg(c1 + i s1).

    Args:
        fourier_coefficients: Array of shape (2, J) storing Fourier coefficients.
            First row for cosine, second for sine.

    Returns:
        Mean - float in [0, 2π)
    """
    return jnp.angle(fourier_circular_m1(fourier_coefficients))


def fourier_circular_variance(fourier_coefficients: jnp.ndarray) -> float:
    """
    Extracts the circular variance of the distribution represented by the inputted Fourier coefficients.
    Equal to 1 - |E[exp(i * phi)]| = 1 - π|c1 + i s1|.
    Note that the circular variance is different to the traditional variance, but is more natural
    for wrapped distributions.
    For sharply peaked distributions we have that 2 * circular_variance = standard variance.

    Args:
        fourier_coefficients: Array of shape (2, J) storing Fourier coefficients.
            First row for cosine, second for sine.

    Returns:
        Circular variance - float in [0, 1]
    """
    return 1 - jnp.abs(fourier_circular_m1(fourier_coefficients))


def fourier_holevo_variance(fourier_coefficients: jnp.ndarray) -> float:
    """
    Extracts the Holevo phase variance of the distribution represented by the inputted Fourier coefficients.
    Equal to |E[exp(i * phi)]|^-2 - 1 = 1/(π^2 * (c1^2 + s1^2)) - 1.
    The Holevo phase variance is different to the traditional variance however they are
    equivalent for sharply peaked distributions, see https://www.dominicberry.org/research/thesis.pdf.

    Args:
        fourier_coefficients: Array of shape (2, J) storing Fourier coefficients.
            First row for cosine, second for sine.

    Returns:
        Holevo phase variance - float in [0, ∞)
    """
    return jnp.abs(fourier_circular_m1(fourier_coefficients)) ** -2 - 1


def _fourier_cosine_distance(val: float, fourier_coefficients: jnp.ndarray) -> float:
    c1 = fourier_coefficients[0, 0]
    s1 = fourier_coefficients[1, 0]
    return 1 - jnp.pi * (c1 * jnp.cos(val) + s1 * jnp.sin(val))


def fourier_cosine_distance(
    val: Union[float, jnp.ndarray], fourier_coefficients: jnp.ndarray
) -> float:
    """
    Evaluates E[1 - cos(phi - val)] where the expectation value is taken
    with respect the Fourier distribution.

    Args:
        val: Float or vector of values to calculate expected cosine distance from
        fourier_coefficients: Array of shape (2, J) storing Fourier coefficients.
            First row for cosine, second for sine.

    Returns:
        Float or vector of expected cosine distances, same shape as val
    """
    val = jnp.atleast_1d(val)
    return jnp.squeeze(
        vmap(_fourier_cosine_distance, in_axes=(0, None))(val, fourier_coefficients)
    )


def fourier_evidence(
    fourier_coefficients: jnp.ndarray,
    m: int,
    k: int,
    beta: float,
    error_rate: Union[float, Callable[[int], float]] = 0.0,
) -> float:
    """
    Evaluates p(m | k, beta) = ∫p(phi) p(m | phi, k, beta) dphi
    where p(phi) is the pdf defined by fourier_coefficients.

    Args:
        fourier_coefficients: Array of shape (2, J) storing prior coefficients
            First row for cosine, second for sine.
        m: Measured bit
        k: Integer exponent
        beta: Phase shift in [0, 2π)
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k

    Returns:
        Float value in [0, ∞) for p(m | k, beta)

    """
    if callable(error_rate):
        error_rate = error_rate(k)

    k = jnp.array(k, dtype=int)
    gamma = beta - m * jnp.pi
    cos_gamma = jnp.cos(gamma) * (1 - error_rate)
    sin_gamma = jnp.sin(gamma) * (1 - error_rate)
    return (
        0.5
        + 0.5 * jnp.pi * cos_gamma * fourier_coefficients[0, k - 1]
        - 0.5 * jnp.pi * sin_gamma * fourier_coefficients[1, k - 1]
    )


def _c1s1_coeffs(
    prior_fourier_coefficients: jnp.ndarray, k: int, error_rate: float
) -> Tuple[float, float, float, float]:
    k = jnp.array(k, dtype=int)
    J = prior_fourier_coefficients.shape[1] - 1

    success_rate = 1 - error_rate
    ckm1 = jnp.where((k >= 2) & (k - 2 <= J), prior_fourier_coefficients[0, k - 2], 0)
    ckm1 = jnp.where(k == 1, 1 / jnp.pi, ckm1) * success_rate
    ckp1 = (
        jnp.where((k >= 0) & (k <= J), prior_fourier_coefficients[0, k], 0)
        * success_rate
    )
    skm1 = (
        jnp.where((k >= 2) & (k - 2 <= J), prior_fourier_coefficients[1, k - 2], 0)
        * success_rate
    )
    skp1 = (
        jnp.where((k >= 0) * (k <= J), prior_fourier_coefficients[1, k], 0)
        * success_rate
    )
    return ckm1, ckp1, skm1, skp1


def fourier_posterior_c1s1(
    prior_fourier_coefficients: jnp.ndarray,
    m: int,
    k: int,
    beta: float,
    error_rate: Union[float, Callable[[int], float]] = 0.0,
) -> Tuple[float, float]:
    """
    Evaluates only the first two coefficients of a single update posterior p(phi | m, k, beta).

    Args:
        fourier_coefficients: Array of shape (2, J) storing prior coefficients
            First row for cosine, second for sine.
        m: Measured bit
        k: Integer exponent
        beta: Phase shift in [0, 2π)
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k

    Returns:
        float value in [0, ∞) for p(m | k, beta)

    """
    if callable(error_rate):
        error_rate = error_rate(k)

    ckm1, ckp1, skm1, skp1 = _c1s1_coeffs(prior_fourier_coefficients, k, error_rate)

    gamma = beta - m * jnp.pi
    cos_gamma = jnp.cos(gamma)
    sin_gamma = jnp.sin(gamma)
    cbar1 = (
        0.5 * prior_fourier_coefficients[0, 0]
        + 0.25 * cos_gamma * ckm1
        + 0.25 * cos_gamma * ckp1
        - 0.25 * sin_gamma * skm1
        - 0.25 * sin_gamma * skp1
    )
    sbar1 = (
        0.5 * prior_fourier_coefficients[1, 0]
        - 0.25 * sin_gamma * ckm1
        + 0.25 * sin_gamma * ckp1
        - 0.25 * cos_gamma * skm1
        + 0.25 * cos_gamma * skp1
    )

    evidence = fourier_evidence(prior_fourier_coefficients, m, k, beta, error_rate)
    return jnp.array([[cbar1], [sbar1]]) / evidence


def fourier_expected_posterior_circular_variance(
    prior_fourier_coefficients: jnp.ndarray,
    k: int,
    beta: float,
    error_rate: Union[float, Callable[[int], float]] = 0.0,
) -> float:
    """
    Calculates expected circular variance of the single update posterior distribution.

    E[var_C(p(phi | m, k, beta))] = Σ_m p(m | k, beta) var_C(p(phi | m, k, beta))

    Args:
        fourier_coefficients: Array of shape (2, J) storing prior coefficients
            First row for cosine, second for sine.
        k: Integer exponent
        beta: Phase shift in [0, 2π)
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k

    Returns:
        float value in [0, 1]

    """
    if callable(error_rate):
        error_rate = error_rate(k)

    m0_c1s1 = fourier_posterior_c1s1(prior_fourier_coefficients, 0, k, beta, error_rate)
    m0_cvar = fourier_circular_variance(m0_c1s1)
    m0_evidence = fourier_evidence(prior_fourier_coefficients, 0, k, beta, error_rate)

    m1_c1s1 = fourier_posterior_c1s1(prior_fourier_coefficients, 1, k, beta, error_rate)
    m1_cvar = fourier_circular_variance(m1_c1s1)
    m1_evidence = fourier_evidence(prior_fourier_coefficients, 1, k, beta, error_rate)
    return m0_cvar * m0_evidence + m1_cvar * m1_evidence


def fourier_expected_posterior_holevo_variance(
    prior_fourier_coefficients: jnp.ndarray,
    k: int,
    beta: float,
    error_rate: Union[float, Callable[[int], float]] = 0.0,
) -> float:
    """
    Calculates expected Holevo variance of the single update posterior distribution.

    E[var_H(p(phi | m, k, beta))] = Σ_m p(m | k, beta) var_H(p(phi | m, k, beta))

    Args:
        fourier_coefficients: Array of shape (2, J) storing prior coefficients
            First row for cosine, second for sine.
        k: Integer exponent
        beta: Phase shift in [0, 2π)
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k

    Returns:
        float value in [0, ∞)

    """
    if callable(error_rate):
        error_rate = error_rate(k)

    m0_c1s1 = fourier_posterior_c1s1(prior_fourier_coefficients, 0, k, beta, error_rate)
    m0_hvar = fourier_holevo_variance(m0_c1s1)
    m0_evidence = fourier_evidence(prior_fourier_coefficients, 0, k, beta, error_rate)

    m1_c1s1 = fourier_posterior_c1s1(prior_fourier_coefficients, 1, k, beta, error_rate)
    m1_hvar = fourier_holevo_variance(m1_c1s1)
    m1_evidence = fourier_evidence(prior_fourier_coefficients, 1, k, beta, error_rate)
    return m0_hvar * m0_evidence + m1_hvar * m1_evidence


def fourier_get_beta_given_k(
    prior_fourier_coefficients: jnp.ndarray,
    k: int,
    error_rate: Union[float, Callable[[int], float]] = 0.0,
) -> Tuple[float, float]:
    """
    Calculates beta that minimises the expected circular variance of a single update, for a given k.

    Args:
        prior_fourier_coefficients: Array of shape (2, J) storing prior coefficients
            First row for cosine, second for sine.
        k: Integer exponent
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k

    Returns:
        Float beta in [0, π) and its corresponding negative expected circular variance

    """
    k = jnp.array(k, dtype=int)
    if callable(error_rate):
        error_rate = error_rate(k)

    abar = prior_fourier_coefficients[0, 0] / 2 * jnp.pi
    dbar = prior_fourier_coefficients[1, 0] / 2 * jnp.pi
    ckm1, ckp1, skm1, skp1 = _c1s1_coeffs(prior_fourier_coefficients, k, error_rate)

    bbar = 0.25 * (ckm1 + ckp1) * jnp.pi
    cbar = -0.25 * (skm1 + skp1) * jnp.pi

    ebar = 0.25 * (skp1 - skm1) * jnp.pi
    fbar = 0.25 * (ckp1 - ckm1) * jnp.pi
    return maximise_expected_circular_variance(abar, bbar, cbar, dbar, ebar, fbar)


def fourier_get_k_and_beta(
    prior_fourier_coefficients: jnp.ndarray,
    error_rate: Union[float, Callable[[int], float]] = 0.0,
    k_max: int = jnp.inf,
) -> Tuple[int, float]:
    """
    Calculates k and beta that minimise the expected circular variance of a single update.

    Args:
        prior_fourier_coefficients: Array of shape (2, J) storing prior coefficients
            First row for cosine, second for sine.
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k
        k_max: Maximum k to consider

    Returns:
        Tuple of (k, beta) with k in {1, ..., min(J, k_max)} and beta in [0, π)

    """
    if callable(error_rate):
        error_rate_func = error_rate
    else:
        error_rate_func = lambda _: error_rate

    J = prior_fourier_coefficients.shape[1] + 1
    error_rates = vmap(error_rate_func)(jnp.arange(1, J))
    betas, utils = vmap(fourier_get_beta_given_k, in_axes=(None, 0, 0))(
        prior_fourier_coefficients, jnp.arange(1, J), error_rates
    )
    utils = jnp.where(jnp.arange(1, J) > k_max, -jnp.inf, utils)

    k_ind = jnp.argmax(utils)
    return k_ind + 1, betas[k_ind]
