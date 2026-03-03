"""mvsusie-py command-line interface.

Supports three analysis modes matching mvsusieR:
  dense      -- raw X/Y matrices (numpy .npy or whitespace-delimited .txt/.csv)
  suff-stat  -- XtX, Xty, yty sufficient statistics
  rss        -- z-scores and LD matrix

Outputs PIPs and posterior coefficients as .npy or .txt files.

Usage examples
--------------
# Dense joint multivariate (recommended for multi-trait plant eQTL)
mvsusie dense --X geno.npy --Y pheno.npy --L 10 --joint \\
              --estimate-prior-variance --out results/

# RSS fine-mapping from GWAS summary stats
mvsusie rss --z zscores.npy --R ld.npy --n 500 --out results/

# Sufficient statistics (large datasets)
mvsusie suff-stat --XtX XtX.npy --Xty Xty.npy --yty yty.npy --n 500 --out results/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_matrix(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    elif p.suffix in (".txt", ".csv", ".tsv"):
        delimiter = "," if p.suffix == ".csv" else None
        return np.loadtxt(p, delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}. Use .npy, .txt, .csv, or .tsv")


def _save(arr: np.ndarray, out_dir: Path, name: str, fmt: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(out_dir / f"{name}.npy", arr)
    else:
        np.savetxt(out_dir / f"{name}.txt", arr, fmt="%.8g")


def _save_json(obj: dict, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def _cmd_dense(args: argparse.Namespace) -> None:
    from mvsusie_py import (
        fit_susie_multivariate_joint,
        fit_susie_univariate,
        learn_mash_covariances,
        mash_reweight_joint,
        mvsusie,
        coef,
    )

    print(f"[mvsusie dense] Loading X from {args.X}")
    X = _load_matrix(args.X)
    print(f"[mvsusie dense] Loading Y from {args.Y}")
    Y = _load_matrix(args.Y)

    print(f"  X shape: {X.shape}  Y shape: {Y.shape}")
    out_dir = Path(args.out)

    rv = None
    if args.residual_variance:
        rv = _load_matrix(args.residual_variance)

    if args.joint and Y.ndim == 2:
        print(f"[mvsusie dense] Running joint multivariate fit (L={args.L}, "
              f"estimate_prior_variance={args.estimate_prior_variance})")
        fit = fit_susie_multivariate_joint(
            X=X,
            Y=Y,
            L=args.L,
            prior_variance=args.prior_variance,
            residual_variance=rv,
            estimate_residual_variance=not args.no_estimate_residual_variance,
            estimate_prior_variance=args.estimate_prior_variance,
            max_iter=args.max_iter,
            tol=args.tol,
        )

        if args.mash_reweight:
            print("[mvsusie dense] Learning mash covariances and reweighting...")
            B_hat = np.einsum('lp,lpr->pr', fit.alpha, fit.mu)
            mc = learn_mash_covariances(
                B_hat,
                n_data_driven=args.mash_n_data_driven,
            )
            _save(mc.weights,     out_dir, "mash_weights",     args.format)
            _save(mc.components,  out_dir, "mash_components",  args.format)
            fit = mash_reweight_joint(fit, mc, X, Y)
            print(f"  Mash components: {mc.components.shape[0]}  "
                  f"Top weight: {mc.weights.max():.4f}")

        B_hat = np.einsum('lp,lpr->pr', fit.alpha, fit.mu)
        pip = fit.pip
        meta = {
            "converged": fit.converged,
            "n_iter": fit.n_iter,
            "prior_variance": fit.prior_variance,
            "prior_variance_trace": fit.prior_variance_trace,
            "mode": "joint",
            "L": args.L,
            "X_shape": list(X.shape),
            "Y_shape": list(Y.shape),
        }

    else:
        print(f"[mvsusie dense] Running {'univariate' if Y.ndim == 1 else 'independent multivariate'} "
              f"fit (L={args.L})")
        fit = mvsusie(
            X=X,
            y=Y,
            L=args.L,
            prior_variance=args.prior_variance,
            residual_variance=rv if rv is not None else None,
            estimate_residual_variance=not args.no_estimate_residual_variance,
            max_iter=args.max_iter,
            tol=args.tol,
            joint=False,
        )
        B_hat = coef(fit)
        # For independent fits, compute PIP as max across outcomes
        from mvsusie_py import MultiSusieResult as _MSR
        if isinstance(fit, _MSR):
            pip = np.max(np.stack([f.pip for f in fit.fits], axis=1), axis=1)
        else:
            pip = fit.pip
        meta = {
            "converged": getattr(fit, "converged", getattr(fit.fits[0] if hasattr(fit, "fits") else fit, "converged", None)),
            "n_iter": getattr(fit, "n_iter", getattr(fit.fits[0] if hasattr(fit, "fits") else fit, "n_iter", None)),
            "mode": "independent",
            "L": args.L,
        }

    _save(pip,      out_dir, "pip",  args.format)
    _save(B_hat,    out_dir, "coef", args.format)
    alpha_out = fit.alpha if hasattr(fit, 'alpha') else fit.fits[0].alpha
    _save(alpha_out, out_dir, "alpha", args.format)
    _save_json(meta, out_dir, "meta")

    print(f"[mvsusie dense] Done. Results written to {out_dir}/")
    print(f"  converged={meta.get('converged')}  n_iter={meta.get('n_iter')}")
    print(f"  PIPs > 0.5: {int(np.sum(pip > 0.5))} SNPs")


def _cmd_rss(args: argparse.Namespace) -> None:
    from mvsusie_py import mvsusie_rss, coef

    print(f"[mvsusie rss] Loading z from {args.z}")
    z = _load_matrix(args.z)
    print(f"[mvsusie rss] Loading R from {args.R}")
    R = _load_matrix(args.R)
    out_dir = Path(args.out)

    print(f"  z shape: {z.shape}  R shape: {R.shape}  n={args.n}")
    fit = mvsusie_rss(
        z=z,
        R=R,
        n=args.n,
        L=args.L,
        prior_variance=args.prior_variance,
        residual_variance=args.residual_variance_scalar,
        estimate_residual_variance=not args.no_estimate_residual_variance,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    B_hat = coef(fit)
    _save(fit.pip, out_dir, "pip",  args.format)
    _save(B_hat,   out_dir, "coef", args.format)
    _save(fit.alpha, out_dir, "alpha", args.format)

    meta = {
        "converged": getattr(fit, "converged", None),
        "n_iter": getattr(fit, "n_iter", None),
        "mode": "rss",
        "n": args.n,
        "L": args.L,
    }
    _save_json(meta, out_dir, "meta")

    print(f"[mvsusie rss] Done. Results written to {out_dir}/")
    print(f"  PIPs > 0.5: {int(np.sum(fit.pip > 0.5))} SNPs")


def _cmd_suff_stat(args: argparse.Namespace) -> None:
    from mvsusie_py import mvsusie_suff_stat, coef

    print(f"[mvsusie suff-stat] Loading XtX from {args.XtX}")
    XtX = _load_matrix(args.XtX)
    print(f"[mvsusie suff-stat] Loading Xty from {args.Xty}")
    Xty = _load_matrix(args.Xty)
    print(f"[mvsusie suff-stat] Loading yty from {args.yty}")
    yty_raw = _load_matrix(args.yty)
    yty = float(yty_raw) if yty_raw.ndim == 0 else yty_raw
    out_dir = Path(args.out)

    print(f"  XtX shape: {XtX.shape}  Xty shape: {Xty.shape}  n={args.n}")
    fit = mvsusie_suff_stat(
        XtX=XtX,
        Xty=Xty,
        yty=yty,
        n=args.n,
        L=args.L,
        prior_variance=args.prior_variance,
        estimate_residual_variance=not args.no_estimate_residual_variance,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    B_hat = coef(fit)
    _save(fit.pip, out_dir, "pip",  args.format)
    _save(B_hat,   out_dir, "coef", args.format)
    _save(fit.alpha, out_dir, "alpha", args.format)

    meta = {
        "converged": getattr(fit, "converged", None),
        "n_iter": getattr(fit, "n_iter", None),
        "mode": "suff_stat",
        "n": args.n,
        "L": args.L,
    }
    _save_json(meta, out_dir, "meta")

    print(f"[mvsusie suff-stat] Done. Results written to {out_dir}/")
    print(f"  PIPs > 0.5: {int(np.sum(fit.pip > 0.5))} SNPs")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mvsusie",
        description="mvsusie-py: Python implementation of mvsusieR fine-mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version="mvsusie-py 0.3.0")

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- Common args -------------------------------------------------------
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--L",  type=int,   default=10,    help="Number of single effects (default: 10)")
        p.add_argument("--prior-variance", type=float, default=1.0, help="Initial prior variance V (default: 1.0)")
        p.add_argument("--max-iter", type=int, default=200, help="Max IBSS iterations (default: 200)")
        p.add_argument("--tol", type=float, default=1e-4,  help="Convergence tolerance (default: 1e-4)")
        p.add_argument("--no-estimate-residual-variance", action="store_true",
                       help="Fix residual variance (do not update)")
        p.add_argument("--out", type=str, default="mvsusie_out/",
                       help="Output directory (default: mvsusie_out/)")
        p.add_argument("--format", choices=["npy", "txt"], default="txt",
                       help="Output format: npy (binary) or txt (default: txt)")

    # ---- dense -------------------------------------------------------------
    p_dense = sub.add_parser("dense", help="Fit from raw X/Y matrices")
    p_dense.add_argument("--X", required=True, help="Genotype matrix (.npy/.txt/.csv)")
    p_dense.add_argument("--Y", required=True, help="Phenotype matrix (.npy/.txt/.csv)")
    p_dense.add_argument("--joint", action="store_true",
                         help="Use full joint multivariate inference (recommended)")
    p_dense.add_argument("--estimate-prior-variance", action="store_true",
                         help="Learn prior variance V via EM (joint mode only)")
    p_dense.add_argument("--mash-reweight", action="store_true",
                         help="Learn mash covariances and reweight posteriors (joint mode only)")
    p_dense.add_argument("--mash-n-data-driven", type=int, default=3,
                         help="Number of data-driven mash components (default: 3)")
    p_dense.add_argument("--residual-variance", type=str, default=None,
                         help="Path to initial (r x r) residual covariance matrix (.npy/.txt)")
    add_common(p_dense)
    p_dense.set_defaults(func=_cmd_dense)

    # ---- rss ---------------------------------------------------------------
    p_rss = sub.add_parser("rss", help="Fit from z-scores and LD matrix")
    p_rss.add_argument("--z", required=True, help="Z-score vector/matrix (.npy/.txt)")
    p_rss.add_argument("--R", required=True, help="LD correlation matrix (.npy/.txt)")
    p_rss.add_argument("--n", required=True, type=int, help="Sample size")
    p_rss.add_argument("--residual-variance-scalar", type=float, default=1.0,
                       help="Scalar residual variance (default: 1.0)")
    add_common(p_rss)
    p_rss.set_defaults(func=_cmd_rss)

    # ---- suff-stat ---------------------------------------------------------
    p_ss = sub.add_parser("suff-stat", help="Fit from sufficient statistics")
    p_ss.add_argument("--XtX", required=True, help="X'X matrix (.npy/.txt)")
    p_ss.add_argument("--Xty", required=True, help="X'y vector/matrix (.npy/.txt)")
    p_ss.add_argument("--yty", required=True, help="y'y scalar/vector (.npy/.txt)")
    p_ss.add_argument("--n",   required=True, type=int, help="Sample size")
    add_common(p_ss)
    p_ss.set_defaults(func=_cmd_suff_stat)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
