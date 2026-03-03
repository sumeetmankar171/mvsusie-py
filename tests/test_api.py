import numpy as np

from mvsusie_py import coef, fit_susie_univariate, mvsusie, mvsusie_suff_stat


def test_mvsusie_alias_matches_fit_susie_univariate():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(120, 30))
    y = rng.normal(size=120)

    fit_a = fit_susie_univariate(
        X,
        y,
        L=4,
        prior_variance=0.8,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=80,
        tol=1e-7,
    )
    fit_b = mvsusie(
        X,
        y,
        L=4,
        prior_variance=0.8,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=80,
        tol=1e-7,
    )

    assert np.allclose(fit_a.alpha, fit_b.alpha)
    assert np.allclose(fit_a.mu, fit_b.mu)
    assert np.allclose(fit_a.pip, fit_b.pip)


def test_mvsusie_suff_stat_matches_dense_on_centered_data():
    rng = np.random.default_rng(7)
    n, p = 180, 50
    X = rng.normal(size=(n, p))
    b = np.zeros(p)
    b[[6, 21]] = [2.5, -1.9]
    y = X @ b + rng.normal(scale=1.1, size=n)

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()

    fit_dense = fit_susie_univariate(
        X,
        y,
        L=6,
        prior_variance=1.0,
        residual_variance=1.21,
        estimate_residual_variance=False,
        max_iter=150,
        tol=1e-7,
    )

    XtX = Xc.T @ Xc
    Xty = Xc.T @ yc
    yty = float(yc @ yc)

    fit_ss = mvsusie_suff_stat(
        XtX,
        Xty,
        yty,
        n=n,
        L=6,
        prior_variance=1.0,
        residual_variance=1.21,
        estimate_residual_variance=False,
        max_iter=150,
        tol=1e-7,
    )

    assert int(np.argmax(fit_dense.pip)) == int(np.argmax(fit_ss.pip))
    assert np.allclose(coef(fit_dense), coef(fit_ss), atol=2e-2, rtol=5e-2)


def test_mvsusie_suff_stat_validation():
    XtX = np.eye(3)
    Xty = np.ones(2)

    try:
        mvsusie_suff_stat(XtX, Xty, yty=1.0, n=10)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "Xty length" in str(e)
