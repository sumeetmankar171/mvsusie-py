import numpy as np

from mvsusie_py import coef, mvsusie, mvsusie_rss, mvsusie_suff_stat, predict


def _simulate_genotype_like(seed: int = 1, n: int = 500, p: int = 100):
    rng = np.random.default_rng(seed)
    maf = np.concatenate([np.array([0.5, 0.2, 0.1, 0.05]), 0.05 + 0.45 * rng.random(p - 4)])

    # Binomial(2, maf_j) per SNP j, per sample i.
    X = (rng.random((n, p)) < maf).astype(float) + (rng.random((n, p)) < maf).astype(float)

    b = np.zeros(p)
    b[:4] = 3.0
    y = -1.0 + X @ b + rng.normal(size=n)
    return X, y, b


def test_upstream_like_coef_recovers_signal_with_high_correlation():
    X, y, b = _simulate_genotype_like(seed=1)

    fit = mvsusie(
        X,
        y,
        L=10,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=250,
        tol=1e-7,
    )

    b_with_intercept = coef(fit, include_intercept=True)
    target = np.concatenate((np.array([-1.0]), b))

    corr = np.corrcoef(b_with_intercept, target)[0, 1]
    assert corr > 0.98


def test_upstream_like_predict_has_low_rmse():
    X, y, _ = _simulate_genotype_like(seed=1)

    fit = mvsusie(
        X,
        y,
        L=10,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=250,
        tol=1e-7,
    )

    yhat = predict(fit, X)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    assert rmse < 1.25


def test_upstream_like_suff_stat_matches_dense_univariate():
    X, y, _ = _simulate_genotype_like(seed=7, n=400, p=80)
    n = X.shape[0]

    fit_dense = mvsusie(
        X,
        y,
        L=8,
        prior_variance=1.0,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=200,
        tol=1e-7,
    )

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    XtX = Xc.T @ Xc
    Xty = Xc.T @ yc
    yty = float(yc @ yc)

    fit_ss = mvsusie_suff_stat(
        XtX,
        Xty,
        yty,
        n=n,
        L=8,
        prior_variance=1.0,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=200,
        tol=1e-7,
    )

    assert np.allclose(fit_dense.alpha, fit_ss.alpha, atol=1e-7, rtol=1e-7)
    assert np.allclose(coef(fit_dense), coef(fit_ss), atol=1e-7, rtol=1e-7)


def test_upstream_like_rss_top_signal_matches_dense():
    X, y, _ = _simulate_genotype_like(seed=11, n=450, p=90)
    n = X.shape[0]

    fit_dense = mvsusie(
        X,
        y,
        L=10,
        prior_variance=1.0,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=200,
        tol=1e-7,
    )

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    bhat = (Xc.T @ yc) / np.sum(Xc * Xc, axis=0)
    se = np.sqrt(np.var(yc, ddof=1) / np.sum(Xc * Xc, axis=0))
    z = bhat / se
    R = np.corrcoef(Xc, rowvar=False)

    fit_rss = mvsusie_rss(
        z=z,
        R=R,
        n=n,
        L=10,
        prior_variance=1.0,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=200,
        tol=1e-7,
    )

    assert int(np.argmax(fit_dense.pip)) == int(np.argmax(fit_rss.pip))
