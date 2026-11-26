"""Shared pipeline for E5/E6/E12/E13: Emb-S kernels (symmetrised), shot sampling, method evaluation (fixed, CV, CNR)."""
import numpy as np
from .channels import kraus
from .embeddings import SliceFamily
from .kernel import noisy_kernels
from .krr import ridge, cv_lam, cnr_lambda, shot_sample, center_train
from .metrics import auc, accuracy, brier, ece, dispersion, spread


def embS_kernel(fam, ch, gamma, L):
    """Symmetrised noisy Emb-S kernel (P x P) for L slices; ch=None -> noiseless."""
    noise = None if ch is None else ("local", kraus(ch, gamma))
    K = noisy_kernels(np.array(fam.slices(L)), noise, fam.N, [L])[L]
    return (K + K.T) / 2, K


def _safe_ridge(Ktr, Kte, y, lam, center):
    try:
        return ridge(Ktr, Kte, y, lam, center)
    except np.linalg.LinAlgError:
        n = len(y)
        A = Ktr + lam * np.eye(n)
        return Kte @ np.linalg.lstsq(A, y, rcond=None)[0]


def evaluate_methods(Ks, Kref_s, n, ytr, yte, S, rng, lam0=0.1, c=1.0, with_cv=True):
    """Ks, Kref_s: symmetrised (n+nte)x(n+nte) kernels (noisy at depth L, noisy at reference depth).
    Returns dict method -> metrics (shared shot realisation for all methods)."""
    Ktr, Kte = Ks[:n, :n], Ks[n:, :n]
    Rtr, Rte = Kref_s[:n, :n], Kref_s[n:, :n]
    Ktr_h, Kte_h = shot_sample(Ktr, Kte, S, rng)
    Rtr_h, _ = shot_sample(Rtr, Rte, S, rng)
    ytr = ytr.astype(float)
    out = {}

    def rec(name, h, lam=None):
        h = np.clip(h, -1, 1)
        out[name] = dict(auc=auc(h, yte), acc=accuracy(h, yte), brier=brier(h, yte), ece=ece(h, yte),
                         disp=dispersion(h), spread=spread(h), lam=lam)

    rec("fixed", _safe_ridge(Ktr_h, Kte_h, ytr, lam0, False), lam0)
    rec("fixed_centered", _safe_ridge(Ktr_h, Kte_h, ytr, lam0, True), lam0)
    if with_cv:
        lu = cv_lam(Ktr_h, ytr, False)
        rec("cv_unc", _safe_ridge(Ktr_h, Kte_h, ytr, lu, False), lu)
        lc = cv_lam(Ktr_h, ytr, True)
        rec("cv_cen", _safe_ridge(Ktr_h, Kte_h, ytr, lc, True), lc)
    lam_c, a = cnr_lambda(Ktr_h, Rtr_h, n, S, lam0, c)
    rec("cnr", _safe_ridge(Ktr_h, Kte_h, ytr, lam_c, True), lam_c)
    out["a_L_est"] = a
    return out
