import numpy as np

from mvsusie_py import coef, mvsusie, predict


def test_multivariate_coef_and_predict_shapes():
    rng = np.random.default_rng(10)
    n, p, r = 180, 70, 3
    X = rng.normal(size=(n, p))
    B = np.zeros((p, r))
    B[:3, 0] = [2.5, -2.0, 1.5]
    B[:3, 1] = [-1.0, 2.2, 1.2]
    B[:3, 2] = [1.8, 0.8, -2.4]
    intercept = np.array([-1.0, 1.0, -2.0])
    Y = intercept.reshape(1, -1) + X @ B + rng.normal(scale=1.0, size=(n, r))

    fit = mvsusie(
        X,
        Y,
        L=10,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=250,
        tol=1e-7,
    )

    Bhat = coef(fit)
    Bhat_i = coef(fit, include_intercept=True)
    Yhat = predict(fit, X)

    assert Bhat.shape == (p, r)
    assert Bhat_i.shape == (p + 1, r)
    assert Yhat.shape == (n, r)
    assert np.allclose(Bhat_i[1:, :], Bhat)


def test_multivariate_upstream_like_recovery():
    rng = np.random.default_rng(1)
    n, p, r = 500, 100, 3

    maf = np.concatenate([np.array([0.5, 0.2, 0.1, 0.05]), 0.05 + 0.45 * rng.random(96)])
    X = (rng.random((n, p)) < maf).astype(float) + (rng.random((n, p)) < maf).astype(float)

    B = np.zeros((p, r))
    B[:3, 0] = 3.0 * rng.normal(size=3)
    B[:3, 1] = 3.0 * rng.normal(size=3)
    B[:3, 2] = 3.0 * rng.normal(size=3)
    intercept = np.array([-1.0, 1.0, -2.0])

    Y = intercept.reshape(1, -1) + X @ B + rng.normal(size=(n, r))

    fit = mvsusie(
        X,
        Y,
        L=10,
        residual_variance=np.ones(r),
        estimate_residual_variance=False,
        max_iter=250,
        tol=1e-7,
    )

    Bhat = coef(fit)
    Yhat = predict(fit, X)

    corr = np.corrcoef(B[:4, :].ravel(), Bhat[:4, :].ravel())[0, 1]
    rmse = float(np.sqrt(np.mean((Y - Yhat) ** 2)))

    assert corr > 0.95
    assert rmse < 1.25
