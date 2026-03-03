import numpy as np

from mvsusie_py import coef, mvsusie, mvsusie_suff_stat


def test_mvsusie_suff_stat_multivariate_matches_dense_top_signal():
    rng = np.random.default_rng(909)
    n, p, r = 260, 60, 3

    X = rng.normal(size=(n, p))
    B = np.zeros((p, r))
    B[[4, 15], 0] = [2.1, -1.5]
    B[[8, 30], 1] = [1.8, 1.2]
    B[[11, 22], 2] = [-2.0, 1.0]
    Y = X @ B + rng.normal(scale=1.0, size=(n, r))

    fit_dense = mvsusie(
        X,
        Y,
        L=8,
        prior_variance=1.0,
        residual_variance=np.ones(r),
        estimate_residual_variance=False,
        max_iter=180,
        tol=1e-7,
    )

    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    XtX = Xc.T @ Xc
    Xty = Xc.T @ Yc
    yty = np.sum(Yc * Yc, axis=0)

    fit_ss = mvsusie_suff_stat(
        XtX=XtX,
        Xty=Xty,
        yty=yty,
        n=n,
        L=8,
        prior_variance=1.0,
        residual_variance=np.ones(r),
        estimate_residual_variance=False,
        max_iter=180,
        tol=1e-7,
    )

    for j in range(r):
        assert int(np.argmax(fit_dense.fits[j].pip)) == int(np.argmax(fit_ss.fits[j].pip))

    assert coef(fit_ss).shape == (p, r)


def test_mvsusie_suff_stat_multivariate_validation():
    XtX = np.eye(5)
    Xty = np.ones((5, 3))

    try:
        mvsusie_suff_stat(XtX=XtX, Xty=Xty, yty=np.ones(2), n=100)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "yty length" in str(e)

    try:
        mvsusie_suff_stat(XtX=XtX, Xty=Xty, yty=np.ones(3), n=100, residual_variance=np.ones(2))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "residual_variance length" in str(e)
