"""Tests for Fix 1 (joint multivariate inference) and Fix 2 (mash covariance learning)."""
import numpy as np
import pytest

from mvsusie_py import (
    JointMultiSusieResult,
    MashCovariance,
    coef,
    fit_susie_multivariate_joint,
    learn_mash_covariances,
    mvsusie,
    predict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_multitrait_data(n=300, p=80, r=3, n_causal=3, seed=42):
    """Simulate X, Y with shared causal SNPs and correlated residuals."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))

    # Shared causal effect matrix
    B = np.zeros((p, r))
    causal_idx = [0, 1, 2][:n_causal]
    for j in causal_idx:
        B[j] = rng.normal(scale=2.0, size=r)

    # Correlated residual noise (cross-trait covariance)
    L_noise = np.array([[1.0, 0.0, 0.0],
                         [0.6, 0.8, 0.0],
                         [0.3, 0.4, 0.866]])[:r, :r]
    Sigma_true = L_noise @ L_noise.T
    eps = rng.multivariate_normal(np.zeros(r), Sigma_true, size=n)
    Y = X @ B + eps
    return X, Y, B, Sigma_true


# ---------------------------------------------------------------------------
# Fix 1: Joint multivariate IBSS
# ---------------------------------------------------------------------------

class TestJointMultivariateFit:

    def test_returns_correct_type(self):
        X, Y, B, _ = _make_multitrait_data()
        fit = fit_susie_multivariate_joint(X, Y, L=5, max_iter=50)
        assert isinstance(fit, JointMultiSusieResult)

    def test_output_shapes(self):
        n, p, r = 300, 80, 3
        X, Y, B, _ = _make_multitrait_data(n=n, p=p, r=r)
        fit = fit_susie_multivariate_joint(X, Y, L=5, max_iter=50)

        assert fit.alpha.shape == (5, p)
        assert fit.mu.shape    == (5, p, r)
        assert fit.mu2.shape   == (5, p, r)
        assert fit.pip.shape   == (p,)
        assert fit.intercept.shape == (r,)
        assert fit.residual_variance.shape == (r, r)
        assert fit.X_mean.shape == (p,)
        assert fit.Y_mean.shape == (r,)

    def test_alpha_sums_to_one_per_effect(self):
        X, Y, B, _ = _make_multitrait_data()
        fit = fit_susie_multivariate_joint(X, Y, L=5, max_iter=50)
        for l in range(fit.alpha.shape[0]):
            assert np.isclose(fit.alpha[l].sum(), 1.0, atol=1e-6)

    def test_pip_in_unit_interval(self):
        X, Y, B, _ = _make_multitrait_data()
        fit = fit_susie_multivariate_joint(X, Y, L=5, max_iter=50)
        assert np.all(fit.pip >= 0) and np.all(fit.pip <= 1)

    def test_residual_covariance_is_psd(self):
        X, Y, B, _ = _make_multitrait_data()
        fit = fit_susie_multivariate_joint(X, Y, L=5, max_iter=100,
                                           estimate_residual_variance=True)
        eigvals = np.linalg.eigvalsh(fit.residual_variance)
        assert np.all(eigvals >= 0), "Sigma should be PSD"

    def test_causal_snps_have_high_pip(self):
        X, Y, B, _ = _make_multitrait_data(n=400, p=80, r=3, seed=7)
        fit = fit_susie_multivariate_joint(X, Y, L=8, max_iter=150, tol=1e-5)
        causal = [0, 1, 2]
        pip_causal = fit.pip[causal]
        pip_null   = np.delete(fit.pip, causal)
        assert pip_causal.mean() > pip_null.mean(), \
            "Causal SNP PIPs should exceed null SNP PIPs on average"

    def test_predict_shape(self):
        n, p, r = 300, 80, 3
        X, Y, B, _ = _make_multitrait_data(n=n, p=p, r=r)
        fit = fit_susie_multivariate_joint(X, Y, L=5, max_iter=50)
        Yhat = X @ np.einsum('lp,lpr->pr', fit.alpha, fit.mu) + fit.intercept
        assert Yhat.shape == (n, r)

    def test_convergence_flag(self):
        X, Y, B, _ = _make_multitrait_data(n=200, p=40, r=2, seed=99)
        fit = fit_susie_multivariate_joint(X, Y, L=3, max_iter=500, tol=1e-6)
        assert fit.converged, "Should converge on simple simulated data"

    def test_mvsusie_joint_flag_routes_correctly(self):
        X, Y, B, _ = _make_multitrait_data()
        fit = mvsusie(X, Y, L=5, max_iter=50, joint=True)
        assert isinstance(fit, JointMultiSusieResult)

    def test_joint_vs_independent_different_results(self):
        """Joint inference should give different (not identical) PIPs vs independent."""
        X, Y, B, _ = _make_multitrait_data(n=300, p=60, r=3, seed=123)
        fit_joint = fit_susie_multivariate_joint(X, Y, L=5, max_iter=100)
        fit_indep = mvsusie(X, Y, L=5, max_iter=100, joint=False)
        # PIPs won't be identical when covariance structure is non-trivial
        pip_indep = np.stack([f.pip for f in fit_indep.fits], axis=1).mean(axis=1)
        assert not np.allclose(fit_joint.pip, pip_indep, atol=1e-3), \
            "Joint and independent fits should produce different PIPs"

    def test_fixed_residual_variance_respected(self):
        n, p, r = 200, 50, 3
        X, Y, B, _ = _make_multitrait_data(n=n, p=p, r=r)
        Sigma_fixed = np.eye(r) * 2.0
        fit = fit_susie_multivariate_joint(
            X, Y, L=5, max_iter=50,
            residual_variance=Sigma_fixed,
            estimate_residual_variance=False,
        )
        assert np.allclose(fit.residual_variance, Sigma_fixed), \
            "Sigma should not change when estimate_residual_variance=False"


# ---------------------------------------------------------------------------
# Fix 2: Mash-style covariance learning
# ---------------------------------------------------------------------------

class TestLearnMashCovariances:

    def test_returns_correct_type(self):
        rng = np.random.default_rng(0)
        B = rng.normal(size=(50, 3))
        mc = learn_mash_covariances(B)
        assert isinstance(mc, MashCovariance)

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(1)
        B = rng.normal(size=(100, 4))
        mc = learn_mash_covariances(B)
        assert np.isclose(mc.weights.sum(), 1.0, atol=1e-6)

    def test_weights_non_negative(self):
        rng = np.random.default_rng(2)
        B = rng.normal(size=(80, 3))
        mc = learn_mash_covariances(B)
        assert np.all(mc.weights >= 0)

    def test_component_shapes(self):
        r = 4
        rng = np.random.default_rng(3)
        B = rng.normal(size=(60, r))
        mc = learn_mash_covariances(B, n_data_driven=2)
        K = mc.components.shape[0]
        assert mc.components.shape == (K, r, r)
        assert mc.weights.shape    == (K,)

    def test_canonical_components_present(self):
        """Should have at least identity + rank-1 per condition + null."""
        r = 3
        rng = np.random.default_rng(4)
        B = rng.normal(size=(50, r))
        mc = learn_mash_covariances(B, n_data_driven=0)
        # 1 null + 1 identity + r rank-1 = 1+1+r components minimum
        assert mc.components.shape[0] >= 1 + 1 + r

    def test_data_driven_components_added(self):
        r, n_dd = 3, 2
        rng = np.random.default_rng(5)
        B = rng.normal(size=(60, r))
        mc_no_dd = learn_mash_covariances(B, n_data_driven=0)
        mc_dd    = learn_mash_covariances(B, n_data_driven=n_dd)
        assert mc_dd.components.shape[0] == mc_no_dd.components.shape[0] + n_dd

    def test_loglik_trace_non_empty(self):
        rng = np.random.default_rng(6)
        B = rng.normal(size=(50, 3))
        mc = learn_mash_covariances(B, max_iter=20)
        assert len(mc.loglik_trace) > 0

    def test_with_standard_errors(self):
        rng = np.random.default_rng(7)
        B = rng.normal(size=(60, 3))
        S = np.abs(rng.normal(loc=1.0, scale=0.2, size=(60, 3)))
        mc = learn_mash_covariances(B, S_hat=S)
        assert np.isclose(mc.weights.sum(), 1.0, atol=1e-6)

    def test_raises_on_bad_input(self):
        with pytest.raises(ValueError):
            learn_mash_covariances(np.ones((1, 3)))   # p < 2
        with pytest.raises(ValueError):
            learn_mash_covariances(np.ones((10, 1)))  # r < 2

    def test_shared_signal_upweights_identity(self):
        """When all traits share the same effect, identity component should dominate."""
        rng = np.random.default_rng(8)
        p, r = 200, 3
        # All traits see the same signal direction
        shared = rng.normal(size=p)
        B = np.column_stack([shared + 0.05 * rng.normal(size=p) for _ in range(r)])
        mc = learn_mash_covariances(B, n_data_driven=3, max_iter=300)
        # Identity component is index 1
        identity_weight = mc.weights[1]
        assert identity_weight > 0.05, \
            "Identity component should receive meaningful weight for shared signal"

    def test_end_to_end_joint_then_mash(self):
        """Full workflow: joint fit → extract B_hat → learn mash covariances."""
        X, Y, B, Sigma_true = _make_multitrait_data(n=300, p=80, r=3, seed=11)
        fit = fit_susie_multivariate_joint(X, Y, L=8, max_iter=150, tol=1e-5)
        B_hat = np.einsum('lp,lpr->pr', fit.alpha, fit.mu)  # (p, r)
        mc = learn_mash_covariances(B_hat, n_data_driven=3)
        assert isinstance(mc, MashCovariance)
        assert np.isclose(mc.weights.sum(), 1.0, atol=1e-6)
        assert mc.components.shape[1] == 3
        assert mc.components.shape[2] == 3
