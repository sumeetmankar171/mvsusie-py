import numpy as np

from mvsusie_py import create_mixture_prior, fit_susie_univariate, mvsusie


def test_create_mixture_prior_normalizes_and_computes_mean_variance():
    prior = create_mixture_prior(weights=[2.0, 1.0], variances=[0.5, 2.0])

    assert np.allclose(prior.weights, np.array([2.0 / 3.0, 1.0 / 3.0]))
    assert np.allclose(prior.variances, np.array([0.5, 2.0]))
    assert np.isclose(prior.mean_variance, (2.0 / 3.0) * 0.5 + (1.0 / 3.0) * 2.0)


def test_create_mixture_prior_validation():
    try:
        create_mixture_prior(weights=[0.5, 0.5], variances=[1.0])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "same length" in str(e)

    try:
        create_mixture_prior(weights=[0.5, -0.5], variances=[1.0, 2.0])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "non-negative" in str(e)

    try:
        create_mixture_prior(weights=[0.5, 0.5], variances=[1.0, -2.0])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "positive" in str(e)


def test_mvsusie_mixture_prior_matches_effective_scalar_prior():
    rng = np.random.default_rng(202)
    X = rng.normal(size=(140, 35))
    y = rng.normal(size=140)

    prior = create_mixture_prior(weights=[0.25, 0.75], variances=[0.2, 1.4])

    fit_mix = mvsusie(
        X,
        y,
        L=4,
        mixture_prior=prior,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=90,
        tol=1e-7,
    )

    fit_scalar = fit_susie_univariate(
        X,
        y,
        L=4,
        prior_variance=prior.mean_variance,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=90,
        tol=1e-7,
    )

    assert np.allclose(fit_mix.alpha, fit_scalar.alpha)
    assert np.allclose(fit_mix.mu, fit_scalar.mu)
    assert np.allclose(fit_mix.pip, fit_scalar.pip)
