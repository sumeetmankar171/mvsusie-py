import numpy as np

from mvsusie_py import mvsusie_rss


def test_mvsusie_rss_identity_ld_picks_signal():
    p = 60
    causal = 17
    z = np.zeros(p)
    z[causal] = 8.0

    fit = mvsusie_rss(
        z=z,
        R=np.eye(p),
        n=2000,
        L=5,
        prior_variance=0.8,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=100,
        tol=1e-7,
    )

    assert int(np.argmax(fit.pip)) == causal


def test_mvsusie_rss_validation():
    z = np.ones(4)
    R = np.eye(5)

    try:
        mvsusie_rss(z=z, R=R, n=100)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "z length" in str(e)
