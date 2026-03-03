# Changelog

## 0.3.0 - 2026-03-03

- Added full joint multivariate inference (`fit_susie_multivariate_joint`) with shared (r×r) residual covariance Σ updated each IBSS-M iteration.
- Added mash-style covariance mixture learning (`learn_mash_covariances`) via EM over canonical and data-driven (PCA) components.
- Added mash posterior reweighting (`mash_reweight_joint`) feeding learned cross-trait covariance back into IBSS as a structured mixture prior.
- Added EM-based prior variance learning (`em_update_prior_variance`) with `estimate_prior_variance=True` flag in joint fit.
- Added `JointMultiSusieResult` dataclass with `prior_variance_trace` field.
- Added `MashCovariance` dataclass.
- Added CLI (`mvsusie dense / rss / suff-stat`) with `--joint`, `--estimate-prior-variance`, `--mash-reweight` flags.
- Expanded test suite from 52 to 75 tests covering all new APIs and CLI end-to-end.
- Added `.gitignore`, GitHub Actions CI (Python 3.9–3.12).

## 0.2.0 - 2026-03-03

- Added multivariate response support via independent per-outcome fitting.
- Added multivariate support for `coef`, `predict`, `mvsusie_get_lfsr`, and `mvsusie_single_effect_lfsr`.
- Added multivariate `mvsusie_rss` and multivariate `mvsusie_suff_stat` support.
- Added `mvsusie_rss_suff_stat` convenience API.
- Added `create_mixture_prior` and optional usage in `mvsusie`.
- Added API docs and parity smoke script.
- Expanded test suite to cover dense/suff-stat/rss univariate and multivariate paths.

## 0.1.0 - 2026-03-03

- Initial univariate MVP implementation.
- Added core APIs: `fit_susie_univariate`, `mvsusie`, `mvsusie_suff_stat`, `mvsusie_rss`.
- Added utilities: `calc_z`, `coef`, `predict`, `mvsusie_get_lfsr`, `mvsusie_single_effect_lfsr`.
