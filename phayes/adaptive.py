from typing import NamedTuple, Tuple, Union, Callable

from jax import numpy as jnp, vmap
from jax.lax import cond, fori_loop

from phayes import fourier
from phayes import von_mises


class PhayesState(NamedTuple):
    """
    Encodes a circular distribution.

    If fourier_mode is True, then the distribution is encoded as a Fourier series,
    with coefficients stored in fourier_coefficients.
    I.e.  p(phi) = 1/(2π) + \sum_j^J c_j * cos(j * phi) + s_j * sin(j * phi)
    with fourier_coefficients encoding c_j in the first row
    and s_j in the second row (for column j).

    If fourier_mode is False, then the distribution is encoded as a von Mises distribution,
    with mean and concentration stored as a tuple in von_mises_parameters.
    """

    fourier_mode: bool
    fourier_coefficients: jnp.ndarray
    von_mises_parameters: Tuple[float, float]


def init(J: int = 1000) -> PhayesState:
    """
    Initiate adaptive state, to the uniform distribution.

    Args:
        J: Number of Fourier coefficients to store,
            (will convert to von Mises when number of coefficients exceeds J).

    Returns:
        PhayesState (namedtuple with fields: fourier_mode,
                                             fourier_coefficients,
                                             von_mises_parameters)
            fourier_mode initiated to True and fourier_coefficients initiated to zeros.
    """
    return PhayesState(
        fourier_mode=True,
        fourier_coefficients=jnp.zeros((2, J)),
        von_mises_parameters=(0.0, 0.0),
    )


def _update_single_fourier(
    prior_state: PhayesState, m: int, k: int, beta: float, error_rate: float
) -> PhayesState:
    return PhayesState(
        fourier_mode=True,
        fourier_coefficients=fourier._update_single_fourier(
            prior_state.fourier_coefficients, m, k, beta, error_rate
        ),
        von_mises_parameters=(0.0, 0.0),
    )


def _update_single_von_mises(
    prior_state: PhayesState, m: int, k: int, beta: float, error_rate: float
) -> PhayesState:
    return PhayesState(
        fourier_mode=False,
        fourier_coefficients=jnp.zeros_like(prior_state.fourier_coefficients),
        von_mises_parameters=von_mises._update_single_von_mises(
            *prior_state.von_mises_parameters, m, k, beta, error_rate
        ),
    )


def _last_true_index(arr: jnp.ndarray, treat_as_zero: float) -> int:
    arr_bool = jnp.abs(arr) > treat_as_zero
    ind = len(arr) - jnp.argmax(jnp.flip(arr_bool)) - 1
    return jnp.where(jnp.all(~arr_bool), -1, ind)


def _update_single(
    prior_state: PhayesState,
    m: int,
    k: int,
    beta: float,
    error_rate: float,
    treat_as_zero: float,
    deflation: float,
) -> PhayesState:
    J = prior_state.fourier_coefficients.shape[1]

    # J_prior = last index of non-zero coefficient + 1
    J_prior = (
        jnp.maximum(
            _last_true_index(prior_state.fourier_coefficients[0], treat_as_zero),
            _last_true_index(prior_state.fourier_coefficients[1], treat_as_zero),
        )
        + 1
    )
    convert = prior_state.fourier_mode * ((J_prior + k) > J)

    prior_state = cond(
        convert,
        lambda _: PhayesState(
            fourier_mode=False,
            fourier_coefficients=jnp.zeros_like(prior_state.fourier_coefficients),
            von_mises_parameters=von_mises.fourier_to_von_mises(
                prior_state.fourier_coefficients, deflation
            ),
        ),
        lambda _: prior_state,
        None,
    )

    fourier_update = (~convert) & prior_state.fourier_mode

    return cond(
        fourier_update,
        lambda ps: _update_single_fourier(ps, m, k, beta, error_rate),
        lambda ps: _update_single_von_mises(ps, m, k, beta, error_rate),
        prior_state,
    )


def update(
    prior_state: PhayesState,
    m: jnp.ndarray,
    k: jnp.ndarray,
    beta: float,
    error_rate: Union[float, Callable[[int], float]] = 0.0,
    treat_as_zero: float = 0.0,
    deflation: float = 1.0,
) -> PhayesState:
    """
    Applies a posterior update based on shot measurements.

    p(phi | m) ∝ p(phi) * \prod_r^R (1 + (1-error_rate) * cos(k_r * phi + beta_r - m_r * pi)) / 2

    Prior p(phi) and posterior p(phi | m) are stored as a Fourier series
        p(phi) = 1/(2π) + \sum_j^J c_j * cos(j * phi) + s_j * sin(j * phi)
    Until the number of non-zero coefficients exceeds J=the width of the
    PhayesState.fourier_coefficients array, upon which it will convert to a
    von Mises representation - stored in PhayesState.von_mises_parameters.
    The storage mode can be checked via the PhayesState.fourier_mode boolean attribute.

    Args:
        prior_state: PhayesState
            (namedtuple with fields: fourier_mode, fourier_coefficients, von_mises_parameters)
        m: Vector of measured bits
        k: Vector of integer exponents (or single integer if the same across measurements)
        beta: Vector of phase shifts in [0, 2π) (or single float if the same across measurements)
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k
        treat_as_zero: For conversion from Fourier to von Mises
            Fourier cofficients less than the treat_as_zero parameter will be treated as 0
        deflation: Deflation parameter in (0, 1], for conversion from Fourier to von Mises
            (reduce kappa to deflation * kappa to account for approximation in conversion)

    Returns:
        Posterior updated state (may have converted from Fourier to von Mises)

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
        lambda r, ps: _update_single(
            ps, m[r], k[r], beta[r], error_rate[r], treat_as_zero, deflation
        ),
        prior_state,
    )


def circular_m1(state: PhayesState) -> float:
    """
    Extracts the first circular moment of the Fourier or von Mises distribution.
    Equal to arg(E[exp(i * phi)]).

    Args:
        prior_state: PhayesState
            (namedtuple with fields: fourier_mode, fourier_coefficients, von_mises_parameters)

    Returns:
        Mean - float in [0, 2π)
    """
    return cond(
        state.fourier_mode,
        lambda s: fourier.fourier_circular_m1(s.fourier_coefficients),
        lambda s: von_mises.von_mises_circular_m1(*s.von_mises_parameters),
        state,
    )


def circular_mean(state: PhayesState) -> float:
    """
    Extracts the circular mean of the Fourier or von Mises distribution.
    Equal to arg(E[exp(i * phi)]).

    Args:
        prior_state: PhayesState
            (namedtuple with fields: fourier_mode, fourier_coefficients, von_mises_parameters)

    Returns:
        Mean - float in [0, 2π)
    """
    return cond(
        state.fourier_mode,
        lambda s: fourier.fourier_circular_mean(s.fourier_coefficients),
        lambda s: s.von_mises_parameters[0],
        state,
    )


def circular_variance(state: PhayesState) -> float:
    """
    Circular variance of a Fourier or von Mises distribution.
    Equal to 1 - |E[exp(i * phi)]|
    Note that the circular variance is different to the traditional variance, but is more natural
    for wrapped distributions.
    For sharply peaked distributions we have that 2 * circular_variance = standard variance.

    Args:
        PhayesState
            (namedtuple with fields: fourier_mode, fourier_coefficients, von_mises_parameters)

    Returns:
        Circular variance - float in [0, 1]
    """
    return cond(
        state.fourier_mode,
        lambda s: fourier.fourier_circular_variance(s.fourier_coefficients),
        lambda s: von_mises.von_mises_circular_variance(s.von_mises_parameters[1]),
        state,
    )


def holevo_variance(state: PhayesState) -> float:
    """
    Holevo phase variance of a Fourier or von Mises distribution.
    Equal to |E[exp(i * phi)]|^-2 - 1.
    The Holevo phase variance is different to the traditional variance however they are
    equivalent for sharply peaked distributions, see https://www.dominicberry.org/research/thesis.pdf.

    Args:
        PhayesState
            (namedtuple with fields: fourier_mode, fourier_coefficients, von_mises_parameters)

    Returns:
        Holevo phase variance - float in [0, ∞)
    """
    return cond(
        state.fourier_mode,
        lambda s: fourier.fourier_holevo_variance(s.fourier_coefficients),
        lambda s: von_mises.von_mises_holevo_variance(s.von_mises_parameters[1]),
        state,
    )


def cosine_distance(
    val: Union[float, jnp.ndarray], state: PhayesState
) -> Union[float, jnp.ndarray]:
    """
    Evaluates E[1 - cos(phi - val)] where the expectation value is taken
    with respect the Fourier or von Mises distribution.

    Args:
        val: Float or vector of values for pdf to be evaluated at
        PhayesState
            (namedtuple with fields: fourier_mode, fourier_coefficients, von_mises_parameters)

    Returns:
        Float or vector of expected cosine distances, same shape as val

    """
    return cond(
        state.fourier_mode,
        lambda s: fourier.fourier_cosine_distance(val, s.fourier_coefficients),
        lambda s: von_mises.von_mises_cosine_distance(val, *s.von_mises_parameters),
        state,
    )


def evidence(
    state: PhayesState,
    m: int,
    k: int,
    beta: float,
    error_rate: Union[float, Callable[[int], float]] = 0.0,
) -> float:
    """
    Evaluates p(m | k, beta) = ∫p(phi) p(m | phi, k, beta) dphi
    where p(phi) is the prior pdf.

    Args:
        PhayesState
            (namedtuple with fields: fourier_mode, fourier_coefficients, von_mises_parameters)
        m: Measured bit
        k: Integer exponent
        beta: Phase shift in [0, 2π)
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k

    Returns:
        Float value in [0, ∞) for p(m | k, beta)

    """
    return cond(
        state.fourier_mode,
        lambda s: fourier.fourier_evidence(
            s.fourier_coefficients, m, k, beta, error_rate
        ),
        lambda s: von_mises.von_mises_evidence(
            *s.von_mises_parameters, m, k, beta, error_rate
        ),
        state,
    )


def pdf(
    val: Union[float, jnp.ndarray], state: PhayesState
) -> Union[float, jnp.ndarray]:
    """
    Evaluates Fourier or von Mises pdf.

    Args:
        val: Float or vector of values for pdf to be evaluated at
        PhayesState
            (namedtuple with fields: fourier_mode, fourier_coefficients, von_mises_parameters)

    Returns:
        Float or vector of pdf evaluations, same shape as val

    """
    return cond(
        state.fourier_mode,
        lambda s: fourier.fourier_pdf(val, s.fourier_coefficients),
        lambda s: von_mises.von_mises_pdf(val, *s.von_mises_parameters),
        state,
    )


def expected_posterior_circular_variance(
    state: PhayesState,
    k: int,
    beta: float,
    error_rate: Union[float, Callable[[int], float]] = 0.0,
) -> float:
    """
    Calculates expected circular variance of the single update posterior distribution.

    E[var_C(p(phi | m, k, beta))] = Σ_m p(m | k, beta) var_C(p(phi | m, k, beta))

    Args:
        PhayesState
            (namedtuple with fields: fourier_mode, fourier_coefficients, von_mises_parameters)
        k: Integer exponent
        beta: Phase shift in [0, 2π)
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k

    Returns:
        float value in [0, 1]

    """
    return cond(
        state.fourier_mode,
        lambda s: fourier.fourier_expected_posterior_circular_variance(
            s.fourier_coefficients, k, beta, error_rate
        ),
        lambda s: von_mises.von_mises_expected_posterior_circular_variance(
            *s.von_mises_parameters, k, beta, error_rate
        ),
        state,
    )


def expected_posterior_holevo_variance(
    state: PhayesState,
    k: int,
    beta: float,
    error_rate: Union[float, Callable[[int], float]] = 0.0,
) -> float:
    """
    Calculates expected Holevo variance of the single update posterior distribution.

    E[var_H(p(phi | m, k, beta))] = Σ_m p(m | k, beta) var_H(p(phi | m, k, beta))

    Args:
        PhayesState
            (namedtuple with fields: fourier_mode, fourier_coefficients, von_mises_parameters)
        k: Integer exponent
        beta: Phase shift in [0, 2π)
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k

    Returns:
        float value in [0, ∞)

    """
    return cond(
        state.fourier_mode,
        lambda s: fourier.fourier_expected_posterior_holveo_variance(
            s.fourier_coefficients, k, beta, error_rate
        ),
        lambda s: von_mises.von_mises_expected_posterior_holevo_variance(
            *s.von_mises_parameters, k, beta, error_rate
        ),
        state,
    )


def get_beta_given_k(
    state: PhayesState,
    k: int,
    error_rate: Union[float, Callable[[int], float]] = 0.0,
) -> float:
    """
    Calculates beta that minimises the expected circular variance of a single update, for a given k.

    Args:
        PhayesState
            (namedtuple with fields: fourier_mode, fourier_coefficients, von_mises_parameters)
        k: Integer exponent
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k

    Returns:
        Float beta in [0, π) and its corresponding negative expected circular variance

    """
    return cond(
        state.fourier_mode,
        lambda s: fourier.fourier_get_beta_given_k(
            s.fourier_coefficients, k, error_rate
        ),
        lambda s: von_mises.von_mises_get_beta_given_k(
            *s.von_mises_parameters, k, error_rate
        ),
        state,
    )


def get_k_and_beta(
    state: PhayesState,
    error_rate: Union[float, Callable[[int], float]] = 0.0,
    k_max: int = jnp.inf,
) -> float:
    """
    Calculates k and beta that minimise the expected circular variance of a single update.

    Args:
        PhayesState
            (namedtuple with fields: fourier_mode, fourier_coefficients, von_mises_parameters)
        error_rate: Noise parameter, e.g. error_rate = 1 - exp(-k/T2)
            Can be either a float or a Callable function of k
        k_max: Maximum k to consider

    Returns:
        Float beta in [0, π) and its corresponding negative expected circular variance

    """
    return cond(
        state.fourier_mode,
        lambda s: fourier.fourier_get_k_and_beta(
            s.fourier_coefficients, error_rate, k_max
        ),
        lambda s: von_mises.von_mises_get_k_and_beta(
            *s.von_mises_parameters, error_rate, k_max
        ),
        state,
    )
