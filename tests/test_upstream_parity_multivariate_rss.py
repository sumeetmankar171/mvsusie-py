import numpy as np

from mvsusie_py import calc_z, mvsusie, mvsusie_rss


def test_upstream_like_multivariate_rss_top_signals_match_dense():
    rng = np.random.default_rng(222)
    n, p, r = 420, 100, 3

    X = rng.normal(size=(n, p))
    B = np.zeros((p, r))
    B[[8, 15], 0] = [2.0, -1.4]
    B[[12, 31], 1] = [-2.2, 1.1]
    B[[5, 70], 2] = [1.8, 1.6]
    Y = X @ B + rng.normal(scale=1.0, size=(n, r))

    fit_dense = mvsusie(
        X,
        Y,
        L=10,
        prior_variance=1.0,
        residual_variance=np.ones(r),
        estimate_residual_variance=False,
        max_iter=220,
        tol=1e-7,
    )

    Xc = X - X.mean(axis=0)
    R = np.corrcoef(Xc, rowvar=False)
    z = np.column_stack([calc_z(X, Y[:, j]) for j in range(r)])

    fit_rss = mvsusie_rss(
        z=z,
        R=R,
        n=n,
        L=10,
        prior_variance=1.0,
        residual_variance=np.ones(r),
        estimate_residual_variance=False,
        max_iter=220,
        tol=1e-7,
    )

    for j in range(r):
        assert int(np.argmax(fit_dense.fits[j].pip)) == int(np.argmax(fit_rss.fits[j].pip))
