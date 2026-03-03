import numpy as np

from mvsusie_py import coef, fit_susie_univariate, predict


def test_shapes_and_ranges():
    rng = np.random.default_rng(0)
    n, p = 120, 40
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)

    fit = fit_susie_univariate(X, y, L=5, max_iter=50)

    assert fit.alpha.shape == (5, p)
    assert fit.mu.shape == (5, p)
    assert fit.mu2.shape == (5, p)
    assert fit.pip.shape == (p,)
    assert np.allclose(fit.alpha.sum(axis=1), 1.0, atol=1e-6)
    assert np.all((fit.pip >= 0) & (fit.pip <= 1))


def test_identifies_strong_single_signal():
    rng = np.random.default_rng(1)
    n, p = 300, 80
    causal = 13

    X = rng.normal(size=(n, p))
    b = np.zeros(p)
    b[causal] = 3.5
    y = X @ b + rng.normal(scale=1.0, size=n)

    fit = fit_susie_univariate(
        X,
        y,
        L=5,
        prior_variance=1.0,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=200,
        tol=1e-6,
    )

    assert int(np.argmax(fit.pip)) == causal


def test_predict_reasonable_fit():
    rng = np.random.default_rng(2)
    n, p = 200, 60
    X = rng.normal(size=(n, p))
    b = np.zeros(p)
    b[[2, 9, 25]] = [2.2, -1.8, 1.5]
    y = X @ b + rng.normal(scale=1.2, size=n)

    fit = fit_susie_univariate(X, y, L=8, max_iter=150)
    yhat = predict(fit, X)

    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    assert rmse < float(np.std(y))

    bhat = coef(fit)
    assert bhat.shape == (p,)


def test_coef_include_intercept_matches_predict_form():
    rng = np.random.default_rng(3)
    n, p = 110, 25
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)

    fit = fit_susie_univariate(X, y, L=4, max_iter=80)

    b0 = coef(fit, include_intercept=False)
    b1 = coef(fit, include_intercept=True)

    assert b0.shape == (p,)
    assert b1.shape == (p + 1,)
    assert np.isclose(b1[0], fit.intercept)
    assert np.allclose(b1[1:], b0)

    X_aug = np.column_stack([np.ones(n), X])
    yhat_aug = X_aug @ b1
    yhat_predict = predict(fit, X)
    assert np.allclose(yhat_aug, yhat_predict)
