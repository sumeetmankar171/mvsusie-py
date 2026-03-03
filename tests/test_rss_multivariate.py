import numpy as np

from mvsusie_py import coef, mvsusie_rss, predict


def test_mvsusie_rss_multivariate_shapes_and_signal_recovery():
    p, r = 80, 3
    z = np.zeros((p, r))
    causal = [7, 23, 51]
    z[causal[0], 0] = 8.0
    z[causal[1], 1] = -7.5
    z[causal[2], 2] = 9.0

    fit = mvsusie_rss(
        z=z,
        R=np.eye(p),
        n=3000,
        L=6,
        prior_variance=0.8,
        residual_variance=np.ones(r),
        estimate_residual_variance=False,
        max_iter=120,
        tol=1e-7,
    )

    Bhat = coef(fit)
    assert Bhat.shape == (p, r)

    for j, cj in enumerate(causal):
        pip_j = fit.fits[j].pip
        assert int(np.argmax(pip_j)) == cj


def test_mvsusie_rss_multivariate_predict_shape():
    rng = np.random.default_rng(77)
    n, p, r = 90, 40, 2
    X = rng.normal(size=(n, p))
    z = rng.normal(size=(p, r))

    fit = mvsusie_rss(
        z=z,
        R=np.eye(p),
        n=2000,
        L=4,
        prior_variance=0.5,
        residual_variance=np.ones(r),
        estimate_residual_variance=False,
        max_iter=100,
        tol=1e-7,
    )

    Yhat = predict(fit, X)
    assert Yhat.shape == (n, r)


def test_mvsusie_rss_multivariate_validation():
    z = np.ones((5, 3))
    R = np.eye(5)

    try:
        mvsusie_rss(z=z, R=R, n=100, residual_variance=np.ones(2))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "residual_variance length" in str(e)

    try:
        mvsusie_rss(z=np.ones((4, 2)), R=R, n=100)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "z row count" in str(e)
