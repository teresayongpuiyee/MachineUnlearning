import warnings
import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def concentration_basis(H: torch.Tensor, neg_tol: float = 1e-9):
    """
    H : (N, d) retain features at theta_o, from get_representations.
    Returns eigendecompositions of both the centered covariance Cov(h)
    and the uncentered second moment E[hh^T]. Eigenvalues descending,
    eigenvectors as columns.

    neg_tol : max allowed |most-negative eigval| / max-eigval before we
              treat it as a real problem rather than numerical dust.
    """
    H = H.to(torch.float64)                 # precision for the small-eigenvalue tail
    N, d = H.shape
    mean = H.mean(dim=0)                     # h_bar, (d,)

    M = (H.T @ H) / N                        # uncentered: E[hh^T]
    C = M - torch.outer(mean, mean)          # centered: Cov(h) = M - h_bar h_bar^T

    def eigh_desc(A, name):
        A = 0.5 * (A + A.T)                  # enforce exact symmetry
        w, V = torch.linalg.eigh(A)          # ascending eigenvalues, eigvecs = columns
        w = w.flip(0)                        # -> descending
        V = V.flip(1)

        # --- magnitude check BEFORE clamping ---
        w_min, w_max = w.min().item(), w.max().item()
        rel_neg = -w_min / w_max if w_max > 0 else float("inf")
        if rel_neg > neg_tol:               # too big to be rounding dust
            warnings.warn(
                f"[{name}] most-negative eigenvalue {w_min:.3e} is "
                f"{rel_neg:.3e} of the max ({w_max:.3e}); "
                f"exceeds neg_tol={neg_tol:.1e}. Not just numerical dust — "
                f"check the input matrix.",
                RuntimeWarning,
            )

        w = w.clamp_min(0.0)                 # map PSD dust to clean 0
        return w, V, {"min": w_min, "max": w_max, "rel_neg": rel_neg}

    eval_c, evec_c, diag_c = eigh_desc(C, "centered")
    eval_u, evec_u, diag_u = eigh_desc(M, "uncentered")

    return {
        "mean": mean,                                       # for the top-direction check
        "centered":   {"eigvals": eval_c, "eigvecs": evec_c, "neg_diag": diag_c},
        "uncentered": {"eigvals": eval_u, "eigvecs": evec_u, "neg_diag": diag_u},
    }


@torch.no_grad()
def concentration_curves(dhs, eigvecs, n_random: int = 1, seed: int = 0):
    """
    dhs      : (K, d) forget-set mean-shift vectors (K=10), or a list of (d,).
    eigvecs  : (d, d) eigenvectors as COLUMNS, ranked to match eigvals DESCENDING
               (rank 0 = highest-variance direction). Use the centered basis.
    Returns cumulative squared-mass-vs-rank curves: one per shift, plus random baseline(s).
    Each curve is ||dh||^2-normalized (ends at 1.0) so all share a scale.
    """
    if isinstance(dhs, (list, tuple)):
        dhs = torch.stack([torch.as_tensor(x) for x in dhs], dim=0)
    dhs = dhs.to(eigvecs.dtype)                 # match the float64 basis
    K, d = dhs.shape

    # project every shift onto every eigenvector at once:
    # coeff[k, r] = <dh_k, v_r>, since eigvecs[:, r] = v_r
    coeffs = dhs @ eigvecs                        # (K, d) @ (d, d) -> (K, d)
    cum = coeffs.pow(2).cumsum(dim=1)             # cumulative squared mass along rank
    cum = cum / cum[:, -1:].clamp_min(1e-30)      # ||dh||^2-normalize -> ends at 1.0

    # random unit-vector baseline(s) in the same d-dim space
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(n_random, d, generator=g, dtype=eigvecs.dtype)
    R = R / R.norm(dim=1, keepdim=True)
    rcum = (R @ eigvecs).pow(2).cumsum(dim=1)
    rcum = rcum / rcum[:, -1:].clamp_min(1e-30)

    return {"shift": cum, "random": rcum}         # (K, d) and (n_random, d)


def plot_concentration(curves, title="Shift-mass concentration vs. eigenvalue rank"):
    shift, rand = curves["shift"], curves["random"]
    K, d = shift.shape
    ranks = torch.arange(1, d + 1)                # x-axis: 1..d

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k in range(K):
        ax.plot(ranks, shift[k].cpu(), color="C0", alpha=0.35, lw=1)
    ax.plot(ranks, shift.mean(0).cpu(), color="C0", lw=2.5, label="shift Δh (mean of 10)")
    ax.plot(ranks, rand.mean(0).cpu(), color="C1", lw=2, ls="--", label="random unit vector")
    ax.plot(ranks, ranks / d, color="k", lw=1, ls=":", label="uniform (analytic diag.)")

    ax.set_xlabel("eigenvalue rank  (0 = highest variance  →  d = lowest variance)")
    ax.set_ylabel("cumulative squared mass")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(1, d); ax.set_ylim(0, 1.01)
    fig.tight_layout()
    return fig, ax