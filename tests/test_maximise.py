from jax import numpy as jnp, random

from phayes.maximise import fplus_coeffs, uvarc, maximise_expected_circular_variance


def test_maximise_coeffs():
    for seed in range(10):
        coeffs = random.uniform(random.PRNGKey(seed), (6,))

        fplus_c = fplus_coeffs(*coeffs)

        bs = jnp.linspace(0, jnp.pi, 1000)

        uvcs = uvarc(bs, *coeffs)
        fpfms = (
            -1
            + jnp.sqrt(
                fplus_c[0]
                + fplus_c[1] * jnp.cos(bs)
                + fplus_c[2] * jnp.sin(bs)
                + fplus_c[3] * jnp.cos(2 * bs)
                + fplus_c[4] * jnp.sin(2 * bs)
            )
            + jnp.sqrt(
                fplus_c[0]
                - fplus_c[1] * jnp.cos(bs)
                - fplus_c[2] * jnp.sin(bs)
                + fplus_c[3] * jnp.cos(2 * bs)
                + fplus_c[4] * jnp.sin(2 * bs)
            )
        )

        assert jnp.allclose(uvcs, fpfms, atol=1e-2)

        max_beta, max_eval = maximise_expected_circular_variance(*coeffs)

        assert jnp.isclose(max_eval, uvcs.max())
        assert jnp.isclose(max_beta, bs[uvcs.argmax()], atol=1e-2)
