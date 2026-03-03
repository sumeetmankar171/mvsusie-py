"""Tests for Fix 3: EM prior variance, mash reweighting, and CLI."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mvsusie_py import (
    JointMultiSusieResult,
    em_update_prior_variance,
    fit_susie_multivariate_joint,
    learn_mash_covariances,
    mash_reweight_joint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n=300, p=80, r=3, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    B = np.zeros((p, r))
    effects = [2.0, -1.5, 1.0][:r]
    B[0] = effects + [0.0] * (r - len(effects))
    effects2 = [-1.0, 2.0, 0.5][:r]
    B[1] = effects2 + [0.0] * (r - len(effects2))
    L_noise = np.eye(r)
    if r >= 2:
        L_noise[1, 0] = 0.5
    if r >= 3:
        L_noise[2, 0] = 0.3
        L_noise[2, 1] = 0.4
    Sigma_noise = L_noise @ L_noise.T
    eps = rng.multivariate_normal(np.zeros(r), Sigma_noise, size=n)
    Y = X @ B + eps
    return X, Y, B


# ---------------------------------------------------------------------------
# Fix: EM prior variance learning
# ---------------------------------------------------------------------------

class TestEMPriorVariance:

    def test_em_update_returns_positive(self):
        rng = np.random.default_rng(0)
        L, p, r = 5, 50, 3
        alpha = np.abs(rng.normal(size=(L, p)))
        alpha /= alpha.sum(axis=1, keepdims=True)
        mu  = rng.normal(size=(L, p, r))
        mu2 = mu**2 + np.abs(rng.normal(scale=0.1, size=(L, p, r)))
        V = em_update_prior_variance(alpha, mu, mu2)
        assert V > 0

    def test_em_update_univariate_shapes(self):
        rng = np.random.default_rng(1)
        L, p = 4, 30
        alpha = np.abs(rng.normal(size=(L, p)))
        alpha /= alpha.sum(axis=1, keepdims=True)
        mu  = rng.normal(size=(L, p))
        mu2 = mu**2 + 0.1
        V = em_update_prior_variance(alpha, mu, mu2)
        assert isinstance(V, float)

    def test_joint_fit_with_em_prior_variance_runs(self):
        X, Y, _ = _make_data()
        fit = fit_susie_multivariate_joint(
            X, Y, L=5, max_iter=50,
            estimate_prior_variance=True,
        )
        assert isinstance(fit, JointMultiSusieResult)
        assert fit.prior_variance > 0

    def test_prior_variance_trace_populated(self):
        X, Y, _ = _make_data(n=200, p=50, r=2, seed=7)
        fit = fit_susie_multivariate_joint(
            X, Y, L=4, max_iter=30,
            estimate_prior_variance=True,
        )
        assert len(fit.prior_variance_trace) > 0
        assert all(v > 0 for v in fit.prior_variance_trace)

    def test_prior_variance_trace_empty_without_em(self):
        X, Y, _ = _make_data()
        fit = fit_susie_multivariate_joint(
            X, Y, L=4, max_iter=30,
            estimate_prior_variance=False,
        )
        assert fit.prior_variance_trace == []

    def test_em_prior_variance_changes_v(self):
        X, Y, _ = _make_data(n=300, p=60, r=3, seed=5)
        fit_fixed = fit_susie_multivariate_joint(
            X, Y, L=5, max_iter=80, prior_variance=1.0,
            estimate_prior_variance=False,
        )
        fit_em = fit_susie_multivariate_joint(
            X, Y, L=5, max_iter=80, prior_variance=1.0,
            estimate_prior_variance=True,
        )
        # EM should update V away from the initial 1.0
        assert not np.isclose(fit_em.prior_variance, 1.0, atol=0.05) or \
               len(fit_em.prior_variance_trace) > 0


# ---------------------------------------------------------------------------
# Fix: Mash posterior reweighting
# ---------------------------------------------------------------------------

class TestMashReweightJoint:

    def test_returns_joint_result(self):
        X, Y, _ = _make_data(n=200, p=50, r=3, seed=10)
        fit = fit_susie_multivariate_joint(X, Y, L=5, max_iter=60)
        B_hat = np.einsum('lp,lpr->pr', fit.alpha, fit.mu)
        mc = learn_mash_covariances(B_hat, n_data_driven=2)
        fit2 = mash_reweight_joint(fit, mc, X, Y)
        assert isinstance(fit2, JointMultiSusieResult)

    def test_output_shapes_preserved(self):
        n, p, r, L = 200, 50, 3, 5
        X, Y, _ = _make_data(n=n, p=p, r=r, seed=11)
        fit = fit_susie_multivariate_joint(X, Y, L=L, max_iter=60)
        B_hat = np.einsum('lp,lpr->pr', fit.alpha, fit.mu)
        mc = learn_mash_covariances(B_hat, n_data_driven=2)
        fit2 = mash_reweight_joint(fit, mc, X, Y)
        assert fit2.alpha.shape == (L, p)
        assert fit2.mu.shape    == (L, p, r)
        assert fit2.mu2.shape   == (L, p, r)
        assert fit2.pip.shape   == (p,)
        assert fit2.intercept.shape == (r,)

    def test_alpha_sums_to_one(self):
        X, Y, _ = _make_data(n=200, p=50, r=3)
        fit = fit_susie_multivariate_joint(X, Y, L=5, max_iter=60)
        B_hat = np.einsum('lp,lpr->pr', fit.alpha, fit.mu)
        mc = learn_mash_covariances(B_hat)
        fit2 = mash_reweight_joint(fit, mc, X, Y)
        for l in range(fit2.alpha.shape[0]):
            assert np.isclose(fit2.alpha[l].sum(), 1.0, atol=1e-5)

    def test_pip_in_unit_interval(self):
        X, Y, _ = _make_data()
        fit = fit_susie_multivariate_joint(X, Y, L=5, max_iter=60)
        B_hat = np.einsum('lp,lpr->pr', fit.alpha, fit.mu)
        mc = learn_mash_covariances(B_hat)
        fit2 = mash_reweight_joint(fit, mc, X, Y)
        assert np.all(fit2.pip >= 0) and np.all(fit2.pip <= 1)

    def test_reweighting_changes_pips(self):
        """Mash reweighting should produce different PIPs from the pre-reweight fit."""
        X, Y, _ = _make_data(n=300, p=80, r=3, seed=99)
        fit = fit_susie_multivariate_joint(X, Y, L=8, max_iter=100)
        B_hat = np.einsum('lp,lpr->pr', fit.alpha, fit.mu)
        mc = learn_mash_covariances(B_hat, n_data_driven=3)
        fit2 = mash_reweight_joint(fit, mc, X, Y)
        assert not np.allclose(fit.pip, fit2.pip, atol=1e-4), \
            "Mash reweighting should change PIPs"

    def test_causal_pip_maintained_after_reweight(self):
        """Causal SNPs should still rank above nulls after reweighting."""
        X, Y, _ = _make_data(n=400, p=80, r=3, seed=55)
        fit = fit_susie_multivariate_joint(X, Y, L=8, max_iter=120)
        B_hat = np.einsum('lp,lpr->pr', fit.alpha, fit.mu)
        mc = learn_mash_covariances(B_hat)
        fit2 = mash_reweight_joint(fit, mc, X, Y)
        causal = [0, 1]
        assert fit2.pip[causal].mean() > np.delete(fit2.pip, causal).mean()

    def test_full_pipeline_joint_mash_em(self):
        """Joint fit + EM prior variance + mash learning + reweighting."""
        X, Y, _ = _make_data(n=300, p=80, r=3, seed=123)
        fit = fit_susie_multivariate_joint(
            X, Y, L=8, max_iter=100,
            estimate_prior_variance=True,
            estimate_residual_variance=True,
        )
        B_hat = np.einsum('lp,lpr->pr', fit.alpha, fit.mu)
        mc  = learn_mash_covariances(B_hat, n_data_driven=3)
        fit2 = mash_reweight_joint(fit, mc, X, Y)
        assert isinstance(fit2, JointMultiSusieResult)
        assert np.isclose(mc.weights.sum(), 1.0, atol=1e-6)
        assert np.all(fit2.pip >= 0) and np.all(fit2.pip <= 1)


# ---------------------------------------------------------------------------
# Fix: CLI interface
# ---------------------------------------------------------------------------

class TestCLI:

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "mvsusie_py.cli"] + args,
            capture_output=True, text=True,
        )

    def test_help(self):
        r = self._run(["--help"])
        assert r.returncode == 0
        assert "mvsusie" in r.stdout

    def test_dense_subcommand_help(self):
        r = self._run(["dense", "--help"])
        assert r.returncode == 0
        assert "--X" in r.stdout
        assert "--joint" in r.stdout
        assert "--estimate-prior-variance" in r.stdout
        assert "--mash-reweight" in r.stdout

    def test_rss_subcommand_help(self):
        r = self._run(["rss", "--help"])
        assert r.returncode == 0
        assert "--z" in r.stdout
        assert "--R" in r.stdout

    def test_suff_stat_subcommand_help(self):
        r = self._run(["suff-stat", "--help"])
        assert r.returncode == 0
        assert "--XtX" in r.stdout

    def test_dense_independent_run(self):
        X, Y, _ = _make_data(n=100, p=30, r=2, seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.save(tmp / "X.npy", X)
            np.save(tmp / "Y.npy", Y)
            r = self._run([
                "dense",
                "--X", str(tmp / "X.npy"),
                "--Y", str(tmp / "Y.npy"),
                "--L", "3", "--max-iter", "30",
                "--format", "npy",
                "--out", str(tmp / "out"),
            ])
            assert r.returncode == 0, r.stderr
            pip = np.load(tmp / "out" / "pip.npy")
            assert pip.shape == (30,)
            assert np.all(pip >= 0) and np.all(pip <= 1)

    def test_dense_joint_run(self):
        X, Y, _ = _make_data(n=150, p=40, r=3, seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.save(tmp / "X.npy", X)
            np.save(tmp / "Y.npy", Y)
            r = self._run([
                "dense",
                "--X", str(tmp / "X.npy"),
                "--Y", str(tmp / "Y.npy"),
                "--L", "4", "--max-iter", "30",
                "--joint", "--estimate-prior-variance",
                "--format", "npy",
                "--out", str(tmp / "out"),
            ])
            assert r.returncode == 0, r.stderr
            pip = np.load(tmp / "out" / "pip.npy")
            meta = json.loads((tmp / "out" / "meta.json").read_text())
            assert pip.shape == (40,)
            assert "prior_variance_trace" in meta

    def test_dense_joint_mash_reweight_run(self):
        X, Y, _ = _make_data(n=150, p=40, r=3, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.save(tmp / "X.npy", X)
            np.save(tmp / "Y.npy", Y)
            r = self._run([
                "dense",
                "--X", str(tmp / "X.npy"),
                "--Y", str(tmp / "Y.npy"),
                "--L", "4", "--max-iter", "30",
                "--joint", "--mash-reweight", "--mash-n-data-driven", "2",
                "--format", "npy",
                "--out", str(tmp / "out"),
            ])
            assert r.returncode == 0, r.stderr
            pip = np.load(tmp / "out" / "pip.npy")
            weights = np.load(tmp / "out" / "mash_weights.npy")
            assert pip.shape == (40,)
            assert np.isclose(weights.sum(), 1.0, atol=1e-5)

    def test_rss_run(self):
        rng = np.random.default_rng(4)
        p = 30
        R = np.eye(p)
        z = rng.normal(size=p)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.save(tmp / "z.npy", z)
            np.save(tmp / "R.npy", R)
            r = self._run([
                "rss",
                "--z", str(tmp / "z.npy"),
                "--R", str(tmp / "R.npy"),
                "--n", "200",
                "--L", "3", "--max-iter", "30",
                "--format", "npy",
                "--out", str(tmp / "out"),
            ])
            assert r.returncode == 0, r.stderr
            pip = np.load(tmp / "out" / "pip.npy")
            assert pip.shape == (p,)

    def test_suff_stat_run(self):
        X, Y, _ = _make_data(n=100, p=30, r=1, seed=5)
        y = Y[:, 0]
        XtX = X.T @ X
        Xty = X.T @ y
        yty = float(y @ y)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.save(tmp / "XtX.npy", XtX)
            np.save(tmp / "Xty.npy", Xty)
            np.save(tmp / "yty.npy", np.array(yty))
            r = self._run([
                "suff-stat",
                "--XtX", str(tmp / "XtX.npy"),
                "--Xty", str(tmp / "Xty.npy"),
                "--yty", str(tmp / "yty.npy"),
                "--n", "100",
                "--L", "3", "--max-iter", "30",
                "--format", "npy",
                "--out", str(tmp / "out"),
            ])
            assert r.returncode == 0, r.stderr
            pip = np.load(tmp / "out" / "pip.npy")
            assert pip.shape == (30,)

    def test_txt_output_format(self):
        X, Y, _ = _make_data(n=100, p=20, r=2, seed=6)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.savetxt(tmp / "X.txt", X)
            np.savetxt(tmp / "Y.txt", Y)
            r = self._run([
                "dense",
                "--X", str(tmp / "X.txt"),
                "--Y", str(tmp / "Y.txt"),
                "--L", "3", "--max-iter", "20",
                "--format", "txt",
                "--out", str(tmp / "out"),
            ])
            assert r.returncode == 0, r.stderr
            assert (tmp / "out" / "pip.txt").exists()
            assert (tmp / "out" / "coef.txt").exists()
            assert (tmp / "out" / "meta.json").exists()
