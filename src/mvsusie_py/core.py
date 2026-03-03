from __future__ import annotations

from dataclasses import dataclass, field
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
class JointMultiSusieResult:
    """Result from full joint multivariate SuSiE with shared residual covariance."""
    alpha: np.ndarray          # (L, p) — shared across outcomes
    mu: np.ndarray             # (L, p, r) — posterior means per outcome
    mu2: np.ndarray            # (L, p, r) — posterior second moments
    pip: np.ndarray            # (p,) — marginal PIPs (shared)
    intercept: np.ndarray      # (r,)
    residual_variance: np.ndarray  # (r, r) covariance matrix Sigma
    prior_variance: float
    converged: bool
    n_iter: int
    X_mean: np.ndarray         # (p,)
    Y_mean: np.ndarray         # (r,)
    prior_variance_trace: list[float] = field(default_factory=list)  # EM V trace


@dataclass
class MashCovariance:
    """Learned mash-style covariance mixture (pure Python/numpy, no R)."""
    weights: np.ndarray        # (K,) mixture weights
    components: np.ndarray     # (K, r, r) covariance matrices
    loglik_trace: list[float] = field(default_factory=list)


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



# ---------------------------------------------------------------------------
# Helper: EM update for scalar prior variance V
# ---------------------------------------------------------------------------

def em_update_prior_variance(
    alpha: np.ndarray,
    mu: np.ndarray,
    mu2: np.ndarray,
) -> float:
    """One EM step to update scalar prior variance V.

    For each single-effect l, the marginal likelihood w.r.t. V is maximised
    analytically via the formula:

        V_new = sum_l sum_j alpha[l,j] * mu2_marginal[l,j]

    where mu2_marginal[l,j] is the mean of the diagonal of the posterior
    second-moment matrix for SNP j, effect l. The result is averaged over
    L effects and r outcomes.

    Parameters
    ----------
    alpha : (L, p)
    mu    : (L, p) or (L, p, r)
    mu2   : (L, p) or (L, p, r)

    Returns
    -------
    V_new : float  (positive)
    """
    alpha = np.asarray(alpha, dtype=float)
    mu    = np.asarray(mu,    dtype=float)
    mu2   = np.asarray(mu2,   dtype=float)

    if mu.ndim == 3:
        # Multivariate: average over r outcomes
        mu2_marginal = mu2.mean(axis=2)   # (L, p)
    else:
        mu2_marginal = mu2

    # E[b^2] per effect = sum_j alpha_lj * mu2_lj
    # V_new = mean over L of E[b^2]
    V_new = float(np.mean(np.sum(alpha * mu2_marginal, axis=1)))
    return max(V_new, 1e-10)


# ---------------------------------------------------------------------------
# Helper: mash posterior reweighting for a joint fit
# ---------------------------------------------------------------------------

def mash_reweight_joint(
    fit: "JointMultiSusieResult",
    mc: "MashCovariance",
    X: np.ndarray,
    Y: np.ndarray,
) -> "JointMultiSusieResult":
    """Apply learned mash covariance mixture as a structured prior in a joint fit.

    After `learn_mash_covariances` produces mixture weights pi_k and covariance
    components U_k, this function re-runs one pass of the IBSS-M update replacing
    the scalar prior V*I with the mixture prior  sum_k pi_k * s_k * U_k, where
    s_k is a grid of scaling factors.  The resulting posterior is the
    mixture-weighted average across components.

    This implements the key step missing from the original mvsusie-py: feeding
    learned cross-trait covariance structure back into the fine-mapping.

    Parameters
    ----------
    fit : JointMultiSusieResult  (from fit_susie_multivariate_joint)
    mc  : MashCovariance         (from learn_mash_covariances)
    X   : (n, p) original genotype matrix
    Y   : (n, r) original trait matrix

    Returns
    -------
    Updated JointMultiSusieResult with mash-reweighted posteriors
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, p = X.shape
    r = Y.shape[1]
    L = fit.alpha.shape[0]

    Xc = X - fit.X_mean
    Yc = Y - fit.Y_mean

    # Current residual covariance from the joint fit
    Sigma = fit.residual_variance.copy()
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma)

    alpha = fit.alpha.copy()
    mu    = fit.mu.copy()
    mu2   = fit.mu2.copy()

    B = np.einsum('lp,lpr->pr', alpha, mu)
    R = Yc - Xc @ B

    # Grid of scales (mash uses a grid of sigma^2 values)
    scales = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    K = mc.components.shape[0]
    weights = mc.weights          # (K,)
    components = mc.components    # (K, r, r)

    # Build full prior mixture: K components x S scales = K*S prior matrices
    prior_covs = []
    prior_ws   = []
    for k in range(K):
        for s in scales:
            U = s * components[k]
            # Skip degenerate (zero) components
            if np.max(np.abs(U)) < 1e-12:
                continue
            prior_covs.append(U)
            prior_ws.append(weights[k] / len(scales))

    prior_covs = np.array(prior_covs)   # (M, r, r)
    prior_ws   = np.array(prior_ws, dtype=float)
    prior_ws  /= prior_ws.sum()
    M = prior_covs.shape[0]

    for l in range(L):
        B_l_old = mu[l] * alpha[l, :, np.newaxis]
        R_l = R + Xc @ B_l_old
        bhat = (Xc.T @ R_l) / np.sum(Xc * Xc, axis=0)[:, np.newaxis]  # (p, r)

        logbf    = np.zeros(p)
        post_mean = np.zeros((p, r))
        post_var  = np.zeros((p, r))

        d = np.sum(Xc * Xc, axis=0)

        for j in range(p):
            dj = d[j]
            bj = bhat[j]   # (r,)

            # For each prior component, compute posterior and log BF
            comp_logbf   = np.zeros(M)
            comp_mu      = np.zeros((M, r))
            comp_sigma   = np.zeros((M, r))

            for m, U_m in enumerate(prior_covs):
                prec_prior = np.linalg.pinv(U_m + 1e-10 * np.eye(r))
                prec_post  = prec_prior + dj * Sigma_inv
                try:
                    Sp = np.linalg.inv(prec_post)
                except np.linalg.LinAlgError:
                    Sp = np.linalg.pinv(prec_post)

                mu_p = Sp @ (Sigma_inv @ (dj * bj))

                _, ld_post  = np.linalg.slogdet(Sp)
                _, ld_prior = np.linalg.slogdet(U_m + 1e-10 * np.eye(r))
                quad = float(mu_p @ prec_post @ mu_p)
                comp_logbf[m]  = 0.5 * (ld_post - ld_prior + quad)
                comp_mu[m]     = mu_p
                comp_sigma[m]  = np.diag(Sp)

            # Mixture posterior: weight components by prior_ws * exp(logBF)
            log_w = np.log(prior_ws + 1e-300) + comp_logbf
            log_w -= log_w.max()
            w = np.exp(log_w)
            w /= w.sum()

            # Mixture log BF (marginalise over components) — log-sum-exp trick
            log_ws_bf = np.log(prior_ws + 1e-300) + comp_logbf  # (M,)
            lmax = log_ws_bf.max()
            logbf[j] = lmax + np.log(np.sum(np.exp(log_ws_bf - lmax)))

            # Mixture posterior mean and variance
            post_mean[j] = w @ comp_mu         # weighted mean
            # Law of total variance: Var = E[Var] + Var[E]
            mean_var  = w @ comp_sigma
            var_mean  = w @ (comp_mu ** 2) - post_mean[j] ** 2
            post_var[j] = mean_var + var_mean

        alpha_l = _softmax(logbf)
        alpha[l] = alpha_l
        mu[l]    = post_mean
        mu2[l]   = post_var + post_mean ** 2

        B_l_new = mu[l] * alpha_l[:, np.newaxis]
        R = R_l - Xc @ B_l_new

    B   = np.einsum('lp,lpr->pr', alpha, mu)
    pip = 1.0 - np.prod(1.0 - alpha, axis=0)
    intercept = fit.Y_mean - fit.X_mean @ B

    return JointMultiSusieResult(
        alpha=alpha,
        mu=mu,
        mu2=mu2,
        pip=pip,
        intercept=intercept,
        residual_variance=Sigma,
        prior_variance=fit.prior_variance,
        converged=fit.converged,
        n_iter=fit.n_iter,
        X_mean=fit.X_mean,
        Y_mean=fit.Y_mean,
        prior_variance_trace=fit.prior_variance_trace,
    )


def fit_susie_multivariate_joint(
    X: np.ndarray,
    Y: np.ndarray,
    L: int = 10,
    prior_variance: float = 1.0,
    residual_variance: np.ndarray | None = None,
    estimate_residual_variance: bool = True,
    estimate_prior_variance: bool = False,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> JointMultiSusieResult:
    """Full joint multivariate SuSiE with a shared (r x r) residual covariance.

    Each single-effect update uses the inverse of the joint covariance Sigma so
    that cross-trait signals are modelled together rather than independently.
    Optionally performs EM updates of the scalar prior variance V.

    Parameters
    ----------
    X : (n, p) genotype / predictor matrix
    Y : (n, r) multi-trait response matrix
    L : number of single effects
    prior_variance : initial scalar prior variance V for each effect
    residual_variance : (r, r) initial Sigma; defaults to diag(var(Y))
    estimate_residual_variance : update Sigma (r x r) each IBSS iteration
    estimate_prior_variance : update scalar V via EM each IBSS iteration
    max_iter, tol : convergence controls

    Returns
    -------
    JointMultiSusieResult  (.prior_variance_trace records V at each iteration)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if Y.ndim != 2:
        raise ValueError("Y must be 2D for joint multivariate fitting")
    n, p = X.shape
    r = Y.shape[1]
    if Y.shape[0] != n:
        raise ValueError("Y row count must match X rows")
    if L <= 0:
        raise ValueError("L must be positive")
    if prior_variance <= 0:
        raise ValueError("prior_variance must be positive")

    # Centre
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    Xc = X - X_mean
    Yc = Y - Y_mean

    d = np.sum(Xc * Xc, axis=0)          # (p,) column squared-norms
    if np.any(d == 0):
        raise ValueError("X contains zero-variance columns")

    # Initialise Sigma
    if residual_variance is None:
        Sigma = np.diag(np.var(Yc, axis=0))
    else:
        Sigma = np.asarray(residual_variance, dtype=float)
        if Sigma.shape != (r, r):
            raise ValueError("residual_variance must be (r, r)")
    Sigma = Sigma.copy()

    # Initialise parameters  alpha:(L,p)  mu:(L,p,r)  mu2:(L,p,r)
    alpha = np.full((L, p), 1.0 / p)
    mu    = np.zeros((L, p, r))
    mu2   = np.zeros((L, p, r))

    # B_hat (p, r): posterior expected coefficients
    B = np.einsum('lp,lpr->pr', alpha, mu)   # (p, r)
    R = Yc - Xc @ B                           # (n, r) residuals

    converged = False
    prev_B = B.copy()
    pv_trace: list[float] = []

    for it in range(1, max_iter + 1):
        try:
            Sigma_inv = np.linalg.inv(Sigma)  # (r, r)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(Sigma)

        for l in range(L):
            # Restore residual for this effect
            B_l_old = mu[l] * alpha[l, :, np.newaxis]   # (p, r)
            R_l = R + Xc @ B_l_old                       # (n, r)

            # Sufficient stats for each SNP: bhat_j = (x_j' R_l) / d_j  shape (p, r)
            bhat = (Xc.T @ R_l) / d[:, np.newaxis]      # (p, r)

            # Multivariate Bayes factor per SNP using shared Sigma
            # log BF_j = 0.5 * [ log|I + V * d_j * Sigma_inv|^{-1}
            #                   + bhat_j' (V^{-1}/d_j * I + Sigma_inv)^{-1} Sigma_inv^2 bhat_j ]
            # Simplified scalar-prior form (V scalar):
            logbf = np.zeros(p)
            post_mean = np.zeros((p, r))
            post_var  = np.zeros((p, r))

            for j in range(p):
                dj = d[j]
                # Posterior precision: (1/V + dj * Sigma_inv)^{-1}
                prec_prior = (1.0 / prior_variance) * np.eye(r)
                prec_data  = dj * Sigma_inv
                prec_post  = prec_prior + prec_data       # (r, r)
                try:
                    Sigma_post = np.linalg.inv(prec_post)
                except np.linalg.LinAlgError:
                    Sigma_post = np.linalg.pinv(prec_post)

                mu_post = Sigma_post @ (Sigma_inv @ (dj * bhat[j]))  # (r,)

                # log |Sigma_post| - log|prior_cov|
                sign_post, ld_post = np.linalg.slogdet(Sigma_post)
                ld_prior = r * np.log(prior_variance)
                log_det_ratio = ld_post - ld_prior   # log|Sigma_post/Sigma_prior|

                # Quadratic term: mu_post' prec_post mu_post  (positive definite)
                quad = float(mu_post @ prec_post @ mu_post)

                logbf[j] = 0.5 * (log_det_ratio + quad)
                post_mean[j] = mu_post
                post_var[j]  = np.diag(Sigma_post)

            alpha_l = _softmax(logbf)

            alpha[l] = alpha_l
            mu[l]    = post_mean                               # (p, r)
            mu2[l]   = post_var + post_mean ** 2              # (p, r)

            B_l_new = mu[l] * alpha_l[:, np.newaxis]         # (p, r)
            R = R_l - Xc @ B_l_new

        B = np.einsum('lp,lpr->pr', alpha, mu)
        dB = np.linalg.norm(B - prev_B)

        if estimate_residual_variance:
            E = Yc - Xc @ B                           # (n, r)
            Sigma = (E.T @ E) / n
            # Regularise: ridge toward diagonal to avoid singularity
            Sigma += 1e-6 * np.eye(r)

        if estimate_prior_variance:
            prior_variance = em_update_prior_variance(alpha, mu, mu2)
            pv_trace.append(prior_variance)

        if dB < tol:
            converged = True
            break
        prev_B = B.copy()

    pip = 1.0 - np.prod(1.0 - alpha, axis=0)   # (p,)
    intercept = Y_mean - X_mean @ B             # (r,)

    return JointMultiSusieResult(
        alpha=alpha,
        mu=mu,
        mu2=mu2,
        pip=pip,
        intercept=intercept,
        residual_variance=Sigma,
        prior_variance=prior_variance,
        converged=converged,
        n_iter=it,
        X_mean=X_mean,
        Y_mean=Y_mean,
        prior_variance_trace=pv_trace,
    )


def learn_mash_covariances(
    B_hat: np.ndarray,
    S_hat: np.ndarray | None = None,
    n_data_driven: int = 3,
    max_iter: int = 500,
    tol: float = 1e-6,
    min_weight: float = 1e-8,
) -> MashCovariance:
    """Learn mash-style covariance mixture via EM (pure numpy, no R).

    Fits a mixture  Y ~ sum_k pi_k * N(0, U_k)  where U_k are (r x r)
    covariance components.  Components include:

    * Canonical: identity I, rank-1 per condition (e_j e_j')
    * Data-driven: top `n_data_driven` PCs of B_hat (mash default strategy)

    Parameters
    ----------
    B_hat : (p, r) matrix of effect estimates (e.g. from coef(joint_fit))
    S_hat : (p, r) standard errors; if given, effects are scaled before PCA
    n_data_driven : number of data-driven covariance components from PCA
    max_iter, tol : EM convergence controls
    min_weight : floor for mixture weights (numerical stability)

    Returns
    -------
    MashCovariance with .weights (K,), .components (K, r, r)
    """
    B = np.asarray(B_hat, dtype=float)
    p, r = B.shape
    if p < 2 or r < 2:
        raise ValueError("B_hat must be at least (2, 2)")

    # ---- Build canonical components ----------------------------------------
    components = []

    # 1. Null (zero matrix — implicit in mash, here kept for completeness)
    components.append(np.zeros((r, r)))

    # 2. Identity (all traits equally affected)
    components.append(np.eye(r))

    # 3. Rank-1 per condition  (e_j e_j')
    for j in range(r):
        U = np.zeros((r, r))
        U[j, j] = 1.0
        components.append(U)

    # ---- Data-driven components via PCA on B_hat ---------------------------
    B_scaled = B.copy()
    if S_hat is not None:
        S = np.asarray(S_hat, dtype=float)
        S = np.where(S > 0, S, 1.0)
        B_scaled = B / S

    # Row-centre before SVD
    B_c = B_scaled - B_scaled.mean(axis=0)
    try:
        _, sv, Vt = np.linalg.svd(B_c, full_matrices=False)
        n_dd = min(n_data_driven, Vt.shape[0])
        for k in range(n_dd):
            v = Vt[k]                    # (r,) top PC direction
            weight = sv[k] ** 2 / p
            U_dd = weight * np.outer(v, v)
            components.append(U_dd)
    except np.linalg.LinAlgError:
        pass  # skip data-driven if SVD fails

    components = np.stack(components, axis=0)   # (K, r, r)
    K = components.shape[0]

    # ---- EM algorithm -------------------------------------------------------
    # We treat each row of B_hat as an observation ~ N(0, U_k)
    # E-step: responsibilities  r_{ik} = pi_k * p(b_i | U_k) / sum_k
    # M-step: pi_k = mean(r_{ik})

    pi = np.full(K, 1.0 / K)
    loglik_trace: list[float] = []

    # Precompute log-likelihoods  log p(b_i | U_k)  shape (p, K)
    def _log_mvn(B: np.ndarray, U: np.ndarray) -> np.ndarray:
        """log N(b; 0, U) for each row b of B.  Returns (p,) vector."""
        reg = U + 1e-8 * np.eye(r)
        try:
            L_chol = np.linalg.cholesky(reg)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse for degenerate components
            eigvals, eigvecs = np.linalg.eigh(reg)
            eigvals = np.maximum(eigvals, 1e-10)
            inv_U = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            log_det = np.sum(np.log(eigvals))
            quad = np.einsum('pi,ij,pj->p', B, inv_U, B)
            return -0.5 * (r * np.log(2 * np.pi) + log_det + quad)

        inv_L = np.linalg.inv(L_chol)
        log_det = 2.0 * np.sum(np.log(np.diag(L_chol)))
        # Mahalanobis: ||inv_L @ b||^2 for each row b
        z = B @ inv_L.T                           # (p, r)
        quad = np.sum(z ** 2, axis=1)             # (p,)
        return -0.5 * (r * np.log(2 * np.pi) + log_det + quad)

    log_liks = np.column_stack([_log_mvn(B, components[k]) for k in range(K)])  # (p, K)

    for em_it in range(max_iter):
        # E-step: log responsibilities
        log_resp = log_liks + np.log(pi + 1e-300)    # (p, K)
        log_norm = log_resp - log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_norm)
        resp /= resp.sum(axis=1, keepdims=True)      # (p, K)

        # Log-likelihood
        ll = float(np.sum(
            log_resp.max(axis=1) + np.log(np.exp(log_norm).sum(axis=1))
        ))
        loglik_trace.append(ll)

        # M-step: update weights
        pi_new = resp.mean(axis=0)
        pi_new = np.maximum(pi_new, min_weight)
        pi_new /= pi_new.sum()

        delta = np.max(np.abs(pi_new - pi))
        pi = pi_new

        if em_it > 0 and delta < tol:
            break

    return MashCovariance(
        weights=pi,
        components=components,
        loglik_trace=loglik_trace,
    )


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
    joint: bool = False,
) -> SusieResult | MultiSusieResult | JointMultiSusieResult:
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
        if joint:
            rv = None if residual_variance is None else np.asarray(residual_variance, dtype=float)
            if rv is not None and rv.ndim == 0:
                rv = float(rv) * np.eye(y_arr.shape[1])
            return fit_susie_multivariate_joint(
                X=X,
                Y=y_arr,
                L=L,
                prior_variance=eff_prior_variance,
                residual_variance=rv,
                estimate_residual_variance=estimate_residual_variance,
                max_iter=max_iter,
                tol=tol,
            )
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
