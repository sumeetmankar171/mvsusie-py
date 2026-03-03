from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SusieResult:
    alpha: np.ndarray
    mu: np.ndarray
    mu2: np.ndarray
    pip: np.ndarray
    intercept: float
    residual_variance: float
    prior_variance: float
    converged: bool
    n_iter: int
    X_mean: np.ndarray
    y_mean: float


@dataclass
class MultiSusieResult:
    fits: list[SusieResult]
    intercept: np.ndarray
    prior_variance: float


@dataclass
class MixturePrior:
    weights: np.ndarray
    variances: np.ndarray
    mean_variance: float


def _softmax(logw: np.ndarray) -> np.ndarray:
    z = logw - np.max(logw)
    w = np.exp(z)
    return w / np.sum(w)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    s = float(np.sum(weights))
    if s <= 0:
        raise ValueError("weights must sum to a positive value")
    return weights / s


def create_mixture_prior(
    weights: np.ndarray,
    variances: np.ndarray,
    normalize_weights: bool = True,
) -> MixturePrior:
    w = np.asarray(weights, dtype=float).reshape(-1)
    v = np.asarray(variances, dtype=float).reshape(-1)

    if w.size == 0 or v.size == 0:
        raise ValueError("weights and variances must be non-empty")
    if w.shape[0] != v.shape[0]:
        raise ValueError("weights and variances must have same length")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if np.any(v <= 0):
        raise ValueError("variances must be positive")

    if normalize_weights:
        w = _normalize_weights(w)
    else:
        if not np.isclose(np.sum(w), 1.0):
            raise ValueError("weights must sum to 1 when normalize_weights=False")

    return MixturePrior(weights=w, variances=v, mean_variance=float(np.sum(w * v)))


def _resolve_prior_variance(
    prior_variance: float,
    mixture_prior: MixturePrior | None,
) -> float:
    if mixture_prior is None:
        return prior_variance
    if not isinstance(mixture_prior, MixturePrior):
        raise ValueError("mixture_prior must be a MixturePrior object")
    return float(mixture_prior.mean_variance)


def coef(
    fit: SusieResult | MultiSusieResult,
    include_intercept: bool = False,
) -> np.ndarray:
    if isinstance(fit, MultiSusieResult):
        b = np.column_stack([np.sum(f.alpha * f.mu, axis=0) for f in fit.fits])
        if include_intercept:
            return np.vstack([fit.intercept.reshape(1, -1), b])
        return b

    b = np.sum(fit.alpha * fit.mu, axis=0)
    if include_intercept:
        return np.concatenate((np.array([fit.intercept]), b))
    return b


def predict(fit: SusieResult | MultiSusieResult, X_new: np.ndarray) -> np.ndarray:
    X_new = np.asarray(X_new, dtype=float)

    if isinstance(fit, MultiSusieResult):
        b = coef(fit, include_intercept=False)
        return fit.intercept.reshape(1, -1) + X_new @ b

    b = coef(fit)
    return fit.intercept + X_new @ b


def fit_susie_univariate(
    X: np.ndarray,
    y: np.ndarray,
    L: int = 10,
    prior_variance: float = 1.0,
    residual_variance: float | None = None,
    estimate_residual_variance: bool = True,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> SusieResult:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("y length must match X rows")
    if L <= 0:
        raise ValueError("L must be positive")
    if prior_variance <= 0:
        raise ValueError("prior_variance must be positive")

    X_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    Xc = X - X_mean
    yc = y - y_mean

    d = np.sum(Xc * Xc, axis=0)
    if np.any(d == 0):
        raise ValueError("X contains zero-variance columns")

    sigma2 = float(np.var(yc)) if residual_variance is None else float(residual_variance)
    if sigma2 <= 0:
        raise ValueError("residual_variance must be positive")

    alpha = np.full((L, p), 1.0 / p)
    mu = np.zeros((L, p))
    mu2 = np.zeros((L, p))

    b = np.sum(alpha * mu, axis=0)
    r = yc - Xc @ b

    converged = False
    prev_b = b.copy()

    for it in range(1, max_iter + 1):
        for l in range(L):
            b_l_old = alpha[l] * mu[l]
            r_l = r + Xc @ b_l_old

            shat2 = sigma2 / d
            bhat = (Xc.T @ r_l) / d
            V = prior_variance

            logbf = 0.5 * (
                np.log(shat2 / (shat2 + V))
                + (bhat * bhat) * (V / (shat2 * (shat2 + V)))
            )
            alpha_l = _softmax(logbf)

            post_mean = (V / (V + shat2)) * bhat
            post_var = (V * shat2) / (V + shat2)

            alpha[l] = alpha_l
            mu[l] = post_mean
            mu2[l] = post_var + post_mean * post_mean

            b_l_new = alpha[l] * mu[l]
            r = r_l - Xc @ b_l_new

        b = np.sum(alpha * mu, axis=0)
        db = np.linalg.norm(b - prev_b)
        if estimate_residual_variance:
            sigma2 = float(np.mean((yc - Xc @ b) ** 2))
            sigma2 = max(sigma2, 1e-12)
        if db < tol:
            converged = True
            break
        prev_b = b.copy()

    pip = 1.0 - np.prod(1.0 - alpha, axis=0)
    intercept = y_mean - float(X_mean @ b)

    return SusieResult(
        alpha=alpha,
        mu=mu,
        mu2=mu2,
        pip=pip,
        intercept=intercept,
        residual_variance=sigma2,
        prior_variance=prior_variance,
        converged=converged,
        n_iter=it,
        X_mean=X_mean,
        y_mean=y_mean,
    )


def fit_susie_multivariate_independent(
    X: np.ndarray,
    Y: np.ndarray,
    L: int = 10,
    prior_variance: float = 1.0,
    residual_variance: float | np.ndarray | None = None,
    estimate_residual_variance: bool = True,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> MultiSusieResult:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if Y.ndim != 2:
        raise ValueError("Y must be 2D for multivariate fitting")
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError("Y row count must match X rows")

    r = Y.shape[1]
    if residual_variance is None or np.isscalar(residual_variance):
        rv = [residual_variance] * r
    else:
        rv_arr = np.asarray(residual_variance, dtype=float).reshape(-1)
        if rv_arr.shape[0] != r:
            raise ValueError("residual_variance length must match Y columns")
        rv = [float(x) for x in rv_arr]

    fits: list[SusieResult] = []
    for j in range(r):
        fits.append(
            fit_susie_univariate(
                X=X,
                y=Y[:, j],
                L=L,
                prior_variance=prior_variance,
                residual_variance=rv[j],
                estimate_residual_variance=estimate_residual_variance,
                max_iter=max_iter,
                tol=tol,
            )
        )

    intercept = np.array([f.intercept for f in fits], dtype=float)
    return MultiSusieResult(fits=fits, intercept=intercept, prior_variance=prior_variance)


def mvsusie(
    X: np.ndarray,
    y: np.ndarray,
    L: int = 10,
    prior_variance: float = 1.0,
    mixture_prior: MixturePrior | None = None,
    residual_variance: float | np.ndarray | None = None,
    estimate_residual_variance: bool = True,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> SusieResult | MultiSusieResult:
    eff_prior_variance = _resolve_prior_variance(prior_variance, mixture_prior)
    y_arr = np.asarray(y)

    if y_arr.ndim == 1:
        return fit_susie_univariate(
            X=X,
            y=y_arr,
            L=L,
            prior_variance=eff_prior_variance,
            residual_variance=residual_variance,
            estimate_residual_variance=estimate_residual_variance,
            max_iter=max_iter,
            tol=tol,
        )

    if y_arr.ndim == 2:
        return fit_susie_multivariate_independent(
            X=X,
            Y=y_arr,
            L=L,
            prior_variance=eff_prior_variance,
            residual_variance=residual_variance,
            estimate_residual_variance=estimate_residual_variance,
            max_iter=max_iter,
            tol=tol,
        )

    raise ValueError("y must be 1D or 2D")


def _mvsusie_suff_stat_univariate(
    XtX: np.ndarray,
    Xty: np.ndarray,
    yty: float,
    n: int,
    L: int,
    prior_variance: float,
    residual_variance: float | None,
    estimate_residual_variance: bool,
    max_iter: int,
    tol: float,
) -> SusieResult:
    p = XtX.shape[0]
    d = np.diag(XtX)

    Xty = np.asarray(Xty, dtype=float).reshape(-1)
    yty = float(yty)

    if Xty.shape[0] != p:
        raise ValueError("Xty length must match XtX dimension")

    sigma2 = (yty / n) if residual_variance is None else float(residual_variance)
    if sigma2 <= 0:
        raise ValueError("residual_variance must be positive")

    alpha = np.full((L, p), 1.0 / p)
    mu = np.zeros((L, p))
    mu2 = np.zeros((L, p))

    b = np.sum(alpha * mu, axis=0)
    xtr = Xty - XtX @ b

    converged = False
    prev_b = b.copy()

    for it in range(1, max_iter + 1):
        for l in range(L):
            b_l_old = alpha[l] * mu[l]
            xtr_l = xtr + XtX @ b_l_old

            shat2 = sigma2 / d
            bhat = xtr_l / d
            V = prior_variance

            logbf = 0.5 * (
                np.log(shat2 / (shat2 + V))
                + (bhat * bhat) * (V / (shat2 * (shat2 + V)))
            )
            alpha_l = _softmax(logbf)

            post_mean = (V / (V + shat2)) * bhat
            post_var = (V * shat2) / (V + shat2)

            alpha[l] = alpha_l
            mu[l] = post_mean
            mu2[l] = post_var + post_mean * post_mean

            b_l_new = alpha[l] * mu[l]
            xtr = xtr_l - XtX @ b_l_new

        b = np.sum(alpha * mu, axis=0)
        db = np.linalg.norm(b - prev_b)
        if estimate_residual_variance:
            rss = yty - 2.0 * float(b @ Xty) + float(b @ XtX @ b)
            sigma2 = max(rss / n, 1e-12)
        if db < tol:
            converged = True
            break
        prev_b = b.copy()

    pip = 1.0 - np.prod(1.0 - alpha, axis=0)

    return SusieResult(
        alpha=alpha,
        mu=mu,
        mu2=mu2,
        pip=pip,
        intercept=0.0,
        residual_variance=sigma2,
        prior_variance=prior_variance,
        converged=converged,
        n_iter=it,
        X_mean=np.zeros(p),
        y_mean=0.0,
    )


def mvsusie_suff_stat(
    XtX: np.ndarray,
    Xty: np.ndarray,
    yty: float | np.ndarray,
    n: int,
    L: int = 10,
    prior_variance: float = 1.0,
    residual_variance: float | np.ndarray | None = None,
    estimate_residual_variance: bool = True,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> SusieResult | MultiSusieResult:
    XtX = np.asarray(XtX, dtype=float)
    Xty_arr = np.asarray(Xty, dtype=float)

    if XtX.ndim != 2 or XtX.shape[0] != XtX.shape[1]:
        raise ValueError("XtX must be square 2D")
    p = XtX.shape[0]
    if n <= 0:
        raise ValueError("n must be positive")
    if L <= 0:
        raise ValueError("L must be positive")
    if prior_variance <= 0:
        raise ValueError("prior_variance must be positive")

    d = np.diag(XtX)
    if np.any(d <= 0):
        raise ValueError("XtX diagonal must be positive")

    if Xty_arr.ndim == 1:
        return _mvsusie_suff_stat_univariate(
            XtX=XtX,
            Xty=Xty_arr,
            yty=float(yty),
            n=n,
            L=L,
            prior_variance=prior_variance,
            residual_variance=residual_variance if residual_variance is None else float(residual_variance),
            estimate_residual_variance=estimate_residual_variance,
            max_iter=max_iter,
            tol=tol,
        )

    if Xty_arr.ndim == 2:
        if Xty_arr.shape[0] != p:
            raise ValueError("Xty row count must match XtX dimension")
        r = Xty_arr.shape[1]

        yty_arr = np.asarray(yty, dtype=float).reshape(-1)
        if yty_arr.shape[0] != r:
            raise ValueError("yty length must match Xty columns")

        if residual_variance is None or np.isscalar(residual_variance):
            rv = [residual_variance] * r
        else:
            rv_arr = np.asarray(residual_variance, dtype=float).reshape(-1)
            if rv_arr.shape[0] != r:
                raise ValueError("residual_variance length must match Xty columns")
            rv = [float(x) for x in rv_arr]

        fits: list[SusieResult] = []
        for j in range(r):
            fits.append(
                _mvsusie_suff_stat_univariate(
                    XtX=XtX,
                    Xty=Xty_arr[:, j],
                    yty=float(yty_arr[j]),
                    n=n,
                    L=L,
                    prior_variance=prior_variance,
                    residual_variance=rv[j],
                    estimate_residual_variance=estimate_residual_variance,
                    max_iter=max_iter,
                    tol=tol,
                )
            )

        intercept = np.array([f.intercept for f in fits], dtype=float)
        return MultiSusieResult(fits=fits, intercept=intercept, prior_variance=prior_variance)

    raise ValueError("Xty must be 1D or 2D")


def mvsusie_rss(
    z: np.ndarray,
    R: np.ndarray,
    n: int,
    L: int = 10,
    prior_variance: float = 1.0,
    residual_variance: float | np.ndarray = 1.0,
    estimate_residual_variance: bool = False,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> SusieResult | MultiSusieResult:
    z_arr = np.asarray(z, dtype=float)
    R = np.asarray(R, dtype=float)

    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be square 2D")
    p = R.shape[0]
    if n <= 0:
        raise ValueError("n must be positive")

    XtX = float(n) * R

    if z_arr.ndim == 1:
        if z_arr.shape[0] != p:
            raise ValueError("z length must match R dimension")
        z_tilde = z_arr / np.sqrt(1.0 + (z_arr * z_arr) / float(n))
        Xty = np.sqrt(float(n)) * z_tilde
        yty = float(n)
        return mvsusie_suff_stat(
            XtX=XtX,
            Xty=Xty,
            yty=yty,
            n=n,
            L=L,
            prior_variance=prior_variance,
            residual_variance=float(residual_variance),
            estimate_residual_variance=estimate_residual_variance,
            max_iter=max_iter,
            tol=tol,
        )

    if z_arr.ndim == 2:
        if z_arr.shape[0] != p:
            raise ValueError("z row count must match R dimension")
        r = z_arr.shape[1]
        z_tilde = z_arr / np.sqrt(1.0 + (z_arr * z_arr) / float(n))
        Xty = np.sqrt(float(n)) * z_tilde
        yty = np.full(r, float(n))

        if np.isscalar(residual_variance):
            rv = np.full(r, float(residual_variance))
        else:
            rv = np.asarray(residual_variance, dtype=float).reshape(-1)
            if rv.shape[0] != r:
                raise ValueError("residual_variance length must match z columns")

        return mvsusie_suff_stat(
            XtX=XtX,
            Xty=Xty,
            yty=yty,
            n=n,
            L=L,
            prior_variance=prior_variance,
            residual_variance=rv,
            estimate_residual_variance=estimate_residual_variance,
            max_iter=max_iter,
            tol=tol,
        )

    raise ValueError("z must be 1D or 2D")


def mvsusie_rss_suff_stat(
    XtX: np.ndarray,
    Xty: np.ndarray,
    n: int,
    yty: float | np.ndarray | None = None,
    L: int = 10,
    prior_variance: float = 1.0,
    residual_variance: float | np.ndarray = 1.0,
    estimate_residual_variance: bool = False,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> SusieResult | MultiSusieResult:
    Xty_arr = np.asarray(Xty)
    if yty is None:
        if Xty_arr.ndim == 1:
            yty_val: float | np.ndarray = float(n)
        elif Xty_arr.ndim == 2:
            yty_val = np.full(Xty_arr.shape[1], float(n))
        else:
            raise ValueError("Xty must be 1D or 2D")
    else:
        yty_val = yty

    return mvsusie_suff_stat(
        XtX=XtX,
        Xty=Xty_arr,
        yty=yty_val,
        n=n,
        L=L,
        prior_variance=prior_variance,
        residual_variance=residual_variance,
        estimate_residual_variance=estimate_residual_variance,
        max_iter=max_iter,
        tol=tol,
    )


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    from math import erf

    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def calc_z(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n, _ = X.shape
    if y.shape[0] != n:
        raise ValueError("y length must match X rows")

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()

    d = np.sum(Xc * Xc, axis=0)
    if np.any(d <= 0):
        raise ValueError("X contains zero-variance columns")

    bhat = (Xc.T @ yc) / d
    y_var = float(np.var(yc, ddof=1))
    se = np.sqrt(np.maximum(y_var / d, 1e-18))
    z = bhat / se
    return z


def _single_fit_lfsr(fit: SusieResult) -> np.ndarray:
    alpha = np.asarray(fit.alpha, dtype=float)
    mu = np.asarray(fit.mu, dtype=float)
    mu2 = np.asarray(fit.mu2, dtype=float)

    post_var = np.maximum(mu2 - mu * mu, 1e-18)
    z = mu / np.sqrt(post_var)
    ppos = _norm_cdf(z)
    pneg = 1.0 - ppos

    p_beta_ge_0 = (1.0 - alpha) + alpha * ppos
    p_beta_le_0 = (1.0 - alpha) + alpha * pneg
    return np.minimum(p_beta_ge_0, p_beta_le_0)


def mvsusie_single_effect_lfsr(
    fit: SusieResult | MultiSusieResult,
) -> np.ndarray:
    if isinstance(fit, MultiSusieResult):
        arr = [_single_fit_lfsr(f) for f in fit.fits]
        return np.stack(arr, axis=2)

    return _single_fit_lfsr(fit)


def mvsusie_get_lfsr(fit: SusieResult | MultiSusieResult) -> np.ndarray:
    se_lfsr = mvsusie_single_effect_lfsr(fit)
    return np.min(se_lfsr, axis=0)
