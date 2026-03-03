# Changelog

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
