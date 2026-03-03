import numpy as np

from mvsusie_py import mvsusie_rss, mvsusie_rss_suff_stat


def test_rss_suff_stat_matches_rss_univariate():
    rng = np.random.default_rng(123)
    p = 70
    z = rng.normal(size=p)
    R = np.eye(p)
    n = 1500

    fit_a = mvsusie_rss(
        z=z,
        R=R,
        n=n,
        L=6,
        prior_variance=0.8,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=120,
        tol=1e-7,
    )

    z_tilde = z / np.sqrt(1.0 + (z * z) / float(n))
    fit_b = mvsusie_rss_suff_stat(
        XtX=float(n) * R,
        Xty=np.sqrt(float(n)) * z_tilde,
        n=n,
        yty=float(n),
        L=6,
        prior_variance=0.8,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=120,
        tol=1e-7,
    )

    assert np.allclose(fit_a.pip, fit_b.pip)


def test_rss_suff_stat_matches_rss_multivariate():
    p, r = 60, 2
    z = np.zeros((p, r))
    z[5, 0] = 7.0
    z[33, 1] = -8.0
    R = np.eye(p)
    n = 2200

    fit_a = mvsusie_rss(
        z=z,
        R=R,
        n=n,
        L=5,
        prior_variance=1.0,
        residual_variance=np.ones(r),
        estimate_residual_variance=False,
        max_iter=120,
        tol=1e-7,
    )

    z_tilde = z / np.sqrt(1.0 + (z * z) / float(n))
    fit_b = mvsusie_rss_suff_stat(
        XtX=float(n) * R,
        Xty=np.sqrt(float(n)) * z_tilde,
        n=n,
        yty=np.full(r, float(n)),
        L=5,
        prior_variance=1.0,
        residual_variance=np.ones(r),
        estimate_residual_variance=False,
        max_iter=120,
        tol=1e-7,
    )

    for j in range(r):
        assert int(np.argmax(fit_a.fits[j].pip)) == int(np.argmax(fit_b.fits[j].pip))


def test_rss_suff_stat_default_yty_is_set_from_n():
    p = 30
    fit = mvsusie_rss_suff_stat(
        XtX=1000.0 * np.eye(p),
        Xty=np.zeros(p),
        n=1000,
        L=3,
        estimate_residual_variance=False,
    )

    assert fit.pip.shape == (p,)
