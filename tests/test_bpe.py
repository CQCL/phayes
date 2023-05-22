from jax import numpy as jnp, random, vmap, jit

import phayes


def loglikelihood(val, m, k, beta):
    k *= jnp.ones(m.size)
    beta *= jnp.ones(m.size)
    return vmap(
        lambda mm, kk, bb: jnp.log(jnp.cos(kk * val / 2 + (bb + mm * jnp.pi) / 2) ** 2)
    )(m, k, beta).sum()


def test_versus_trapeziod_rule():
    prior_fc = jnp.zeros((2, 100))
    m = jnp.array([0, 1, 1, 0, 0, 1])
    k = jnp.array([1, 4, 3, 8, 5, 10])
    beta = jnp.array([1.4, 0.6, 1.2, 1.1, 1.9, 0.3])

    # Exact update from Fourier
    posterior_fc = phayes.fourier_update(prior_fc, m, k, beta)

    # Trapeziodal rule approximation
    linsp = jnp.linspace(-jnp.pi, jnp.pi, 10000)
    liks = jnp.exp(
        vmap(loglikelihood, in_axes=(0, None, None, None))(linsp, m, k, beta)
    )
    norm_constant = jnp.trapz(liks, linsp)
    posterior_evals = liks / norm_constant

    assert jnp.allclose(
        posterior_evals, phayes.fourier_pdf(linsp, posterior_fc), atol=1e-5
    )

    # Check error rate=1 doesn't change posterior
    posterior_fc_full_error = phayes.fourier_update(posterior_fc, m, k, beta, 1.0)
    assert jnp.allclose(posterior_fc_full_error, posterior_fc)

    posterior_fc2 = phayes.fourier_update(posterior_fc, m, k, beta)
    posterior_fc_at_once = phayes.fourier_update(
        prior_fc, jnp.tile(m, 2), jnp.tile(k, 2), jnp.tile(beta, 2)
    )

    assert jnp.allclose(posterior_fc2, posterior_fc_at_once)

    liks2 = jnp.exp(
        vmap(loglikelihood, in_axes=(0, None, None, None))(
            linsp, jnp.tile(m, 2), jnp.tile(k, 2), jnp.tile(beta, 2)
        )
    )
    norm_constant2 = jnp.trapz(liks2, linsp)
    posterior_evals2 = liks2 / norm_constant2

    assert jnp.allclose(
        posterior_evals2, phayes.fourier_pdf(linsp, posterior_fc2), atol=1e-5
    )

    # Check adaptive
    adapt_init = phayes.init(100)
    adapt_posterior = phayes.update(adapt_init, m, k, beta)

    assert jnp.allclose(adapt_posterior.fourier_coefficients, posterior_fc)

    # # Plot comparison of exact Fourier posterior and numeric approximation
    # import matplotlib.pyplot as plt
    # plt.plot(linsp, posterior_evals, color='green', alpha=0.7)
    # pdf = phayes.fourier_pdf(linsp, posterior_fc)
    # plt.plot(linsp, pdf, c='purple')


def test_versus_importance_sampling():
    prior_fc = jnp.zeros((2, 1000))

    m = jnp.zeros(15)
    k = 1
    beta = 1.0

    # Exact update from Fourier
    posterior_fc = phayes.fourier_update(prior_fc, m, k, beta)

    # Importance sampling approximation
    samps = random.uniform(random.PRNGKey(0), shape=(100000,)) * 2 * jnp.pi - jnp.pi

    log_weights = vmap(loglikelihood, in_axes=(0, None, None, None))(samps, m, k, beta)
    weights = jnp.exp(log_weights)
    normalised_weights = weights / weights.sum()
    res_samps = samps[
        random.choice(
            random.PRNGKey(1),
            a=jnp.arange(samps.size),
            shape=samps.shape,
            p=normalised_weights,
        )
    ]
    empirical_circular_m1 = jnp.exp(res_samps * 1.0j).mean()

    assert jnp.isclose(
        phayes.fourier_circular_mean(posterior_fc),
        jnp.angle(empirical_circular_m1),
        atol=1e-1,
    )
    assert jnp.isclose(
        phayes.fourier_holevo_variance(posterior_fc),
        jnp.abs(empirical_circular_m1) ** -2 - 1,
        atol=1e-1,
    )
    assert jnp.isclose(
        phayes.fourier_circular_variance(posterior_fc),
        1 - jnp.abs(empirical_circular_m1),
        atol=1e-1,
    )

    # # Plot comparison of exact Fourier posterior and importance sampling approximation
    # import matplotlib.pyplot as plt
    # plt.hist(res_samps, bins=80, density=True, color='green', alpha=0.7)
    # pdf = phayes.fourier_pdf(linsp, posterior_fc)
    # plt.plot(linsp, pdf, c='purple')


def test_vonMises_conversion():
    vm_mu = 0.4
    vm_kappa = 1.0

    fc = jit(phayes.von_mises_to_fourier, static_argnames="J")(vm_mu, vm_kappa, 1000)

    new_mu, new_kappa = jit(phayes.fourier_to_von_mises)(fc)

    assert jnp.isclose(vm_mu, new_mu)
    assert jnp.isclose(vm_kappa, new_kappa)

    init_state = phayes.init(3)
    m = 1
    k = 10
    beta = 1.4
    update_state = jit(phayes.update)(init_state, m, k, beta)
    vm_post = phayes.von_mises_update(0.0, 0.0, m, k, beta)
    assert not update_state.fourier_mode
    assert jnp.isclose(update_state.von_mises_parameters[0], vm_post[0])
    assert jnp.isclose(update_state.von_mises_parameters[1], vm_post[1])


def test_unitary():

    u_alpha = 0.3
    u_beta = 0.4
    u_xi = 1.0
    u_eta = 0.6
    unitary = (
        jnp.array(
            [
                [jnp.cos(u_alpha), -jnp.sin(u_alpha)],
                [jnp.sin(u_alpha), jnp.cos(u_alpha)],
            ]
        )
        @ jnp.diag(jnp.array([jnp.exp(1.0j * u_xi), jnp.exp(1.0j * u_eta)]))
        @ jnp.array(
            [[jnp.cos(u_beta), jnp.sin(u_beta)], [-jnp.sin(u_beta), jnp.cos(u_beta)]]
        )
    )

    # check matrix is unitary
    assert jnp.allclose(unitary @ unitary.conj().T, jnp.eye(2), atol=1e-7)

    true_eigvals, true_eigvecs = jnp.linalg.eig(unitary)
    phi_ind = 1
    phi_vec = true_eigvecs[:, phi_ind]
    true_phi = jnp.arccos(true_eigvals[phi_ind].real)

    # check eigenvector is correct
    assert jnp.allclose(
        unitary @ phi_vec, jnp.exp(1.0j * true_phi) * phi_vec, atol=1e-7
    )

    controlled_unitary = jnp.block(
        [[jnp.eye(2), jnp.zeros((2, 2))], [jnp.zeros((2, 2)), unitary]]
    )

    H = jnp.array([[1.0, 1.0], [1.0, -1]]) / jnp.sqrt(2)
    big_H = jnp.kron(H, jnp.eye(2))

    def Rz(param: float) -> jnp.ndarray:
        return jnp.array([[jnp.exp(-0.5j * param), 0], [0, jnp.exp(0.5j * param)]])

    def sample_measurement(k, beta, rk):
        statevector = jnp.kron(jnp.array([1, 0]), phi_vec)
        big_Rz = jnp.kron(Rz(beta), jnp.eye(2))

        controlled_unitary_k = jnp.linalg.matrix_power(controlled_unitary, k)

        statevector = big_H @ controlled_unitary_k @ big_Rz @ big_H @ statevector
        measurement_probs = jnp.abs(statevector) ** 2
        measurement_probs_q1 = measurement_probs.reshape(2, 2).sum(axis=1)
        return random.choice(rk, a=2, p=measurement_probs_q1)

    # Check sampling matches analytical likelihood
    beta_single = 1.3
    k_single = 3
    n_measurements = 10000
    measurements_single_k_beta = vmap(sample_measurement, in_axes=(None, None, 0))(
        k_single, beta_single, random.split(random.PRNGKey(0), n_measurements)
    )
    assert jnp.isclose(
        jnp.bincount(measurements_single_k_beta)[0] / n_measurements,
        jnp.cos(k_single * true_phi / 2 + beta_single / 2) ** 2,
        atol=1e-2,
    )

    # Check posterior converges to true_phi
    betas = jnp.linspace(0.0, 2 * jnp.pi, 100)
    ks = (jnp.arange(betas.size) % 10) + 1

    measurements = jnp.array(
        [
            sample_measurement(int(k), b, rk)
            for k, b, rk in zip(ks, betas, random.split(random.PRNGKey(0), betas.size))
        ]
    )

    posterior_fc = phayes.fourier_update(jnp.zeros((2, 1000)), measurements, ks, betas)
    approx_phi = phayes.fourier_circular_mean(posterior_fc)
    assert jnp.isclose(approx_phi, true_phi, atol=1e-1)

    # Test von Mises update
    vm_prior = [0.0, 0.0]
    vm_posterior = phayes.von_mises_update(*vm_prior, measurements, ks, betas)
    assert jnp.isclose(vm_posterior[0], true_phi, atol=1e-1)

    vm_prior_state = phayes.PhayesState(
        fourier_mode=False,
        fourier_coefficients=jnp.zeros((2, 10)),
        von_mises_parameters=tuple(vm_prior),
    )
    vm_posterior_state = phayes.update(vm_prior_state, measurements, ks, betas)
    assert jnp.isclose(vm_posterior_state.von_mises_parameters[0], vm_posterior[0])
    assert jnp.isclose(vm_posterior_state.von_mises_parameters[1], vm_posterior[1])

    # Test error_rate = 1 leaves posterior untouched
    vm_posterior_full_error = phayes.von_mises_update(
        *vm_posterior, measurements, ks, betas, 1
    )
    assert jnp.isclose(vm_posterior[0], vm_posterior_full_error[0])
    assert jnp.isclose(vm_posterior[1], vm_posterior_full_error[1], rtol=1e-2)

    # import matplotlib.pyplot as plt
    # linsp = jnp.linspace(-jnp.pi, jnp.pi, 1000)
    # # linsp = jnp.linspace(true_phi - 1e-1, true_phi + 1e-1, 1000)
    # pdf = phayes.fourier_pdf(linsp, posterior_fc)
    # plt.plot(linsp, pdf, c="purple")
    # plt.axvline(true_phi, c="green")
    # plt.axvline(approx_phi, c="blue")
    # pdf_vm = phayes.fourier_pdf(linsp, phayes.von_mises_to_fourier(*vm_posterior, 1000))
    # plt.plot(linsp, pdf_vm, c="red")


def test_fourier_posterior_c1s1():
    m_init = jnp.array([0, 1, 1, 0, 0, 1])
    k_init = jnp.array([1, 4, 3, 8, 5, 10])
    beta_init = jnp.array([1.4, 0.6, 1.2, 1.1, 1.9, 0.3])

    prior_fc = phayes.fourier_update(jnp.zeros((2, 100)), m_init, k_init, beta_init)

    m = 1
    k = 4
    beta = 0.34

    posterior_fc = phayes.fourier_update(prior_fc, m, k, beta)

    # Check evidence with importance sampling
    samps = random.uniform(random.PRNGKey(0), shape=(100000,)) * 2 * jnp.pi - jnp.pi
    log_lik_weights = vmap(loglikelihood, in_axes=(0, None, None, None))(
        samps, jnp.atleast_1d(m), k, beta
    )
    prior_weights = phayes.fourier_pdf(samps, prior_fc)
    prior_weights = jnp.where(prior_weights < 1e-5, 1e-5, prior_weights)
    weights = jnp.exp(log_lik_weights + jnp.log(prior_weights)) * 2 * jnp.pi
    assert jnp.isclose(
        weights.mean(), phayes.fourier_evidence(prior_fc, m, k, beta), atol=1e-2
    )

    # Check first posterior coeffs are correct
    posterior_c1s1 = phayes.fourier_posterior_c1s1(prior_fc, m, k, beta)
    assert jnp.allclose(posterior_fc[:, :1], posterior_c1s1)

    # Check with k = 1 edge case
    posterior_fc_k1 = phayes.fourier_update(prior_fc, m, 1, beta)
    posterior_c1s1_k1 = phayes.fourier_posterior_c1s1(prior_fc, m, 1, beta)
    assert jnp.allclose(posterior_fc_k1[:, :1], posterior_c1s1_k1)


def test_expected_vars():
    m_init = jnp.array([0, 1, 1, 0, 0, 1])
    k_init = jnp.array([1, 4, 3, 8, 5, 10])
    beta_init = jnp.array([1.4, 0.6, 1.2, 1.1, 1.9, 0.3])

    prior_fc = phayes.fourier_update(jnp.zeros((2, 100)), m_init, k_init, beta_init)

    k = 4
    beta = 0.34

    e_c_var = phayes.fourier_expected_posterior_circular_variance(prior_fc, k, beta)
    c_var_0 = phayes.fourier_circular_variance(
        phayes.fourier_update(prior_fc, 0, k, beta)
    )
    evid_0 = phayes.fourier_evidence(prior_fc, 0, k, beta)
    c_var_1 = phayes.fourier_circular_variance(
        phayes.fourier_update(prior_fc, 1, k, beta)
    )
    evid_1 = phayes.fourier_evidence(prior_fc, 1, k, beta)

    assert jnp.isclose(e_c_var, c_var_0 * evid_0 + c_var_1 * evid_1, atol=1e-2)

    e_h_var = phayes.fourier_expected_posterior_holevo_variance(prior_fc, k, beta)
    h_var_0 = phayes.fourier_holevo_variance(
        phayes.fourier_update(prior_fc, 0, k, beta)
    )
    h_var_1 = phayes.fourier_holevo_variance(
        phayes.fourier_update(prior_fc, 1, k, beta)
    )

    assert jnp.isclose(e_h_var, h_var_0 * evid_0 + h_var_1 * evid_1, atol=1e-2)

    # von Mises
    prior_mu_kappa = phayes.fourier_to_von_mises(prior_fc)
    vm_e_c_var = phayes.von_mises_expected_posterior_circular_variance(
        *prior_mu_kappa, k, beta
    )
    vm_recov_fc = phayes.von_mises_to_fourier(*prior_mu_kappa, 100)
    vm_c_var_0 = phayes.fourier_circular_variance(
        phayes.fourier_update(vm_recov_fc, 0, k, beta)
    )
    vm_evid_0 = phayes.fourier_evidence(vm_recov_fc, 0, k, beta)
    vm_c_var_1 = phayes.fourier_circular_variance(
        phayes.fourier_update(vm_recov_fc, 1, k, beta)
    )
    vm_evid_1 = phayes.fourier_evidence(vm_recov_fc, 1, k, beta)
    assert jnp.isclose(
        vm_e_c_var, vm_c_var_0 * vm_evid_0 + vm_c_var_1 * vm_evid_1, atol=1e-2
    )


def test_get_k_beta_fourier():

    prior_fcs = jnp.array([[0.04, -0.02, 0.08], [0.01, -0.01, 0.02]])

    beta_linsp = jnp.linspace(0, 2 * jnp.pi, 1000)
    k_range = jnp.arange(1, prior_fcs.shape[1] + 1)

    def get_expected_circ_vars(k):
        return vmap(
            phayes.fourier_expected_posterior_circular_variance, in_axes=(None, None, 0)
        )(prior_fcs, k, beta_linsp)

    neg_expected_circ_vars = -jnp.array([get_expected_circ_vars(k) for k in k_range])

    beta_and_circ_vars_given_k = vmap(
        phayes.fourier_get_beta_given_k, in_axes=(None, 0)
    )(prior_fcs, k_range)

    for i in range(len(k_range)):
        assert jnp.isclose(
            beta_and_circ_vars_given_k[1][i], neg_expected_circ_vars[i].max(), atol=1e-3
        )

        smallest_inds = jnp.argpartition(-neg_expected_circ_vars[i], 5)[:5]

        assert jnp.any(
            jnp.isclose(
                beta_and_circ_vars_given_k[0][i], beta_linsp[smallest_inds], atol=1e-1
            )
        )

    k_max, beta_max = phayes.fourier_get_k_and_beta(prior_fcs)
    max_k_ind = beta_and_circ_vars_given_k[1].argmax()

    assert k_max == k_range[max_k_ind]
    assert jnp.isclose(beta_max, beta_and_circ_vars_given_k[0][max_k_ind])


def test_k_beta_von_mises():
    prior_mu_kappa = [1.8, 200.0]
    prior_fcs = phayes.von_mises_to_fourier(*prior_mu_kappa, 1000)

    k_range = jnp.arange(1, 100)

    beta = 1.3

    f_circ_vars = vmap(
        phayes.fourier_expected_posterior_circular_variance, in_axes=(None, 0, None)
    )(prior_fcs, k_range, beta)
    vm_circ_vars = vmap(
        phayes.von_mises_expected_posterior_circular_variance,
        in_axes=(None, None, 0, None),
    )(*prior_mu_kappa, k_range, beta)

    assert jnp.allclose(f_circ_vars, vm_circ_vars, atol=1e-3)

    fourier_beta_and_circ_vars_given_k = vmap(
        phayes.fourier_get_beta_given_k, in_axes=(None, 0)
    )(prior_fcs, k_range)

    vm_beta_and_circ_vars_given_k = vmap(
        phayes.von_mises_get_beta_given_k, in_axes=(None, None, 0)
    )(*prior_mu_kappa, k_range)

    assert jnp.allclose(
        fourier_beta_and_circ_vars_given_k[1],
        vm_beta_and_circ_vars_given_k[1],
        atol=1e-3,
    )

    fourier_k, fourier_beta = phayes.fourier_get_k_and_beta(prior_fcs)
    vm_k, vm_beta = phayes.von_mises_get_k_and_beta(*prior_mu_kappa)

    assert jnp.isclose(fourier_k, vm_k, atol=1e-3)
    assert jnp.isclose(fourier_beta, vm_beta, atol=1e-3)
