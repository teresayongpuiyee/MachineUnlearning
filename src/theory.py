import math
import warnings
import torch
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import pandas as pd


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
    dhs = dhs.to(device=eigvecs.device, dtype=eigvecs.dtype)   # match basis device+dtype
    K, d = dhs.shape

    # project every shift onto every eigenvector at once:
    # coeff[k, r] = <dh_k, v_r>, since eigvecs[:, r] = v_r
    coeffs = dhs @ eigvecs                        # (K, d) @ (d, d) -> (K, d)
    cum = coeffs.pow(2).cumsum(dim=1)             # cumulative squared mass along rank
    shift_sq_mass = cum[:, -1:].clone()
    cum = cum / cum[:, -1:].clamp_min(1e-30)      # ||dh||^2-normalize -> ends at 1.0

    # random unit-vector baseline(s) in the same d-dim space
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(n_random, d, generator=g, dtype=eigvecs.dtype).to(eigvecs.device)
    R = R / R.norm(dim=1, keepdim=True)
    rcum = (R @ eigvecs).pow(2).cumsum(dim=1)
    rand_sq_mass = rcum[:, -1:].clone()
    rcum = rcum / rcum[:, -1:].clamp_min(1e-30)

    return {"shift": cum, "random": rcum, "shift squared mass": shift_sq_mass, "random squared mass": rand_sq_mass}         # (K, d) and (n_random, d)


def plot_concentration(curves, title="Shift-mass concentration vs. eigenvalue rank"):
    shift, rand = curves["shift"], curves["random"]
    K, d = shift.shape
    ranks = torch.arange(1, d + 1)                # x-axis: 1..d
    rand_mean = rand.mean(0).cpu()

    ncols = min(K, 4)
    nrows = math.ceil(K / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.25 * ncols, 2.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    for k in range(K):
        ax = axes_flat[k]
        ax.plot(ranks, shift[k].cpu(), color="C0", lw=1.5, label=f"shift {k}")
        ax.plot(ranks, rand_mean, color="C1", lw=2, ls="--", label="random unit vector")
        ax.set_title(f"shift {k}", fontsize=9)
        ax.set_xlim(1, d); ax.set_ylim(0, 1.01)
        ax.legend(loc="upper left", fontsize=7)

    # hide any unused axes
    for ax in axes_flat[K:]:
        ax.axis("off")

    fig.supxlabel("eigenvalue rank  (0 = highest variance  →  d = lowest variance)")
    fig.supylabel("cumulative squared mass")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


@torch.no_grad()
def diagnose_concentration(dhs, eigvals, eigvecs):
    if isinstance(dhs, (list, tuple)):
        dhs = torch.stack([torch.as_tensor(x) for x in dhs], 0)
    dhs = dhs.to(device=eigvecs.device, dtype=eigvecs.dtype)
    K, d = dhs.shape

    # 0. is the basis actually sorted descending? (the thing you're right to question)
    desc = bool((eigvals[:-1] >= eigvals[1:] - 1e-9).all())
    print(f"eigvals descending? {desc}")
    print(f"  eigval[rank 0]   = {eigvals[0]:.4e}   (should be LARGEST)")
    print(f"  eigval[rank {d-1}] = {eigvals[-1]:.4e}   (should be SMALLEST)")

    # 1. per-rank mass (the DERIVATIVE of your cumulative curve) averaged over the 10 shifts
    coeffs = dhs @ eigvecs
    mass = coeffs.pow(2)
    mass = mass / mass.sum(1, keepdim=True)      # normalize each shift
    mass_mean = mass.mean(0)                       # (d,)

    # 2. where is the STEEPEST increase, and is that rank high- or low-variance?
    top = int(mass_mean.argmax())
    pct = 100 * (eigvals < eigvals[top]).float().mean()   # 100% = highest variance
    print(f"\nsteepest-increase rank (most mass): {top}")
    print(f"  eigenvalue there: {eigvals[top]:.4e}")
    print(f"  variance percentile: {pct:.1f}%  (100% = highest-variance end)")

    # 3. cumulative mass at a few high-variance cut points
    for k in [1, 5, 10, 25, 50, d // 2]:
        print(f"  mass in top-{k:>3} highest-var ranks: {mass_mean[:k].sum():.3f}")
    return mass_mean


@torch.no_grad()
def diagnose_by_variance(dhs, eigvals, eigvecs):
    if isinstance(dhs, (list, tuple)):
        dhs = torch.stack([torch.as_tensor(x) for x in dhs], 0)
    dhs = dhs.to(device=eigvecs.device, dtype=eigvecs.dtype)
    K, d = dhs.shape
    ev = eigvals.to(eigvecs.dtype)

    # how front-loaded is the spectrum? (this is the crux)
    var_cum = ev.cumsum(0) / ev.sum()             # cumulative VARIANCE fraction vs rank
    for f in [0.5, 0.9, 0.95, 0.99]:
        r = int((var_cum < f).sum())
        print(f"  {int(f*100)}% of retain variance is in the top {r} ranks "
              f"({100*r/d:.1f}% of the rank axis)")

    # shift mass per rank
    mass = (dhs @ eigvecs).pow(2)
    mass = mass / mass.sum(1, keepdim=True)
    mass_mean = mass.mean(0)

    # KEY comparison: shift-mass CDF vs variance CDF, both over rank
    mass_cum = mass_mean.cumsum(0)
    print("\n  rank | cum-variance | cum-shift-mass")
    for k in [1, 5, 10, 25, 50, 100, 256, 512]:
        k = min(k, d)
        print(f"  {k:>4} |   {var_cum[k-1]:.3f}      |   {mass_cum[k-1]:.3f}")

    # variance-weighted: does shift mass fall on HIGH-variance directions more than chance?
    # <mass, normalized-eigval> vs the random-vector expectation (= mean eigval fraction)
    ev_frac = ev / ev.sum()
    align = (mass_mean * ev_frac).sum() * d        # >1 = mass favors high-var dirs
    print(f"\n  variance-alignment ratio: {align:.3f}  (1.0 = like random; >1 high-var; <1 low-var)")
    return mass_mean, var_cum


@torch.no_grad()
def plot_concentration_by_variance(dhs, eigvals, eigvecs, n_random=200, seed=0):
    if isinstance(dhs, (list, tuple)):
        dhs = torch.stack([torch.as_tensor(x) for x in dhs], 0)
    dhs = dhs.to(device=eigvecs.device, dtype=eigvecs.dtype)
    K, d = dhs.shape
    ev = eigvals.to(eigvecs.dtype)

    x = (ev.cumsum(0) / ev.sum()).cpu()           # x-axis: cumulative retain-variance fraction

    def cum_mass(V):                               # V: (m, d) unit rows
        m = (V @ eigvecs).pow(2).cumsum(1)
        return (m / m[:, -1:].clamp_min(1e-30)).cpu()

    shift = cum_mass(dhs)

    g = torch.Generator().manual_seed(seed)
    R = torch.randn(n_random, d, generator=g, dtype=eigvecs.dtype).to(eigvecs.device)
    R = R / R.norm(dim=1, keepdim=True)
    rand = cum_mass(R)
    r_lo, r_med, r_hi = rand.quantile(torch.tensor([0.05, 0.5, 0.95], dtype=rand.dtype), dim=0)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k in range(K):
        ax.plot(x, shift[k], color="C0", alpha=0.35, lw=1)
    ax.plot(x, shift.mean(0), color="C0", lw=2.5, label="shift Δh (mean of 10)")
    ax.plot(x, r_med, color="C1", lw=2, ls="--", label="random (median)")
    ax.fill_between(x, r_lo, r_hi, color="C1", alpha=0.15, label="random 5–95%")
    ax.plot([0, 1], [0, 1], color="k", lw=1, ls=":", label="y=x (mass tracks variance)")

    ax.set_xlabel("cumulative retain-variance fraction  (0 → high-var directions used up → 1)")
    ax.set_ylabel("cumulative shift-mass fraction")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    fig.tight_layout()
    return fig, ax


@torch.no_grad()
def resolve_concentration(dhs, eigvals, eigvecs):
    if isinstance(dhs, (list, tuple)):
        dhs = torch.stack([torch.as_tensor(x) for x in dhs], 0)
    dhs = dhs.to(device=eigvecs.device, dtype=eigvecs.dtype)
    K, d = dhs.shape
    ev = eigvals.to(eigvecs.dtype)

    var_cum  = (ev.cumsum(0) / ev.sum())
    mass     = (dhs @ eigvecs).pow(2)
    mass     = mass / mass.sum(1, keepdim=True)
    mass_cum = mass.mean(0).cumsum(0)

    # (a) how front-loaded is the spectrum? — the missing variable
    print("spectrum shape:")
    for f in [0.5, 0.9, 0.95, 0.99]:
        r = int((var_cum < f).sum())
        print(f"  {int(f*100)}% variance by rank {r}  ({100*r/d:.1f}% of rank axis)")

    # (b) the tie-breaker: variance at the cum-mass=0.1 rank
    r1 = int((mass_cum >= 0.1).nonzero()[0])
    print(f"\ncum-mass=0.10 at rank {r1}; cum-variance there = {var_cum[r1]*100:.1f}%")
    print("  -> high cum-variance here = LOW-variance concentration (prediction holds)")
    print("  -> low  cum-variance here = HIGH-variance concentration")

    # (c) axis-free confirmation: variance-alignment ratio (immune to rank nonlinearity)
    ev_frac = ev / ev.sum()
    align = (mass.mean(0) * ev_frac).sum() * d
    print(f"\nvariance-alignment ratio: {align:.3f}  "
          f"(<1 low-var, =1 random, >1 high-var)")
    return align.item()


@torch.no_grad()
def settle_it(dhs, eigvals, eigvecs):
    if isinstance(dhs, (list, tuple)):
        dhs = torch.stack([torch.as_tensor(x) for x in dhs], 0)
    dhs = dhs.to(device=eigvecs.device, dtype=eigvecs.dtype)
    K, d = dhs.shape
    ev = eigvals.to(eigvecs.dtype)

    mass = (dhs @ eigvecs).pow(2)
    mass = (mass / mass.sum(1, keepdim=True)).mean(0)     # (d,) mean mass per rank
    ev_frac = ev / ev.sum()

    # 1. how much mass is literally on the top few ranks?
    for k in [1, 2, 3, 5, 8]:
        print(f"  mass in top-{k} ranks: {mass[:k].sum():.4f}   "
              f"(their variance frac: {ev_frac[:k].sum():.4f})")

    # 2. decompose the alignment ratio: per-rank contribution mass[r]*ev_frac[r]*d
    contrib = mass * ev_frac * d
    order = contrib.argsort(descending=True)
    print(f"\n  alignment ratio = {contrib.sum():.3f}")
    print("  top contributors to that ratio:")
    for r in order[:5].tolist():
        print(f"    rank {r:>3}: mass={mass[r]:.4f}, ev_frac={ev_frac[r]:.4f}, "
              f"contributes {contrib[r]:.3f}")

    # 3. the honest split: mass in top-k vs variance in top-k, several k
    var_cum, mass_cum = ev_frac.cumsum(0), mass.cumsum(0)
    print("\n  rank | cum-mass | cum-var")
    for k in [1, 8, 25, 50, 100, 256, 512]:
        k = min(k, d)
        print(f"  {k:>4} | {mass_cum[k-1]:.3f}    | {var_cum[k-1]:.3f}")
    return contrib


@torch.no_grad()
def full_concentration_report(dhs, eigvals, eigvecs,
                              out_path,
                              seed=0,
                              mass_levels=(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                           0.6, 0.7, 0.8, 0.9, 1.0)):
    """
    dhs      : (K, d) forget-set mean-shift vectors (K=10), or list of (d,).
    eigvals  : (d,) descending eigenvalues of retain covariance (centered).
    eigvecs  : (d, d) matching eigenvectors as COLUMNS, descending.
    Saves K+1 figures (each 4 subplots) and a CSV of threshold tables.
    """
    out_dir = f"{out_path}/concentration_figs"
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(dhs, (list, tuple)):
        dhs = torch.stack([torch.as_tensor(x) for x in dhs], 0)
    dhs = dhs.to(device=eigvecs.device, dtype=eigvecs.dtype)
    K, d = dhs.shape
    ev = eigvals.to(eigvecs.dtype)

    # --- shared retain-spectrum quantities (same for every vector) ---
    ev_frac = (ev / ev.sum())                 # per-rank variance fraction
    var_cum = ev_frac.cumsum(0)               # cumulative variance fraction
    ranks = torch.arange(d)                    # 0-indexed rank axis

    # --- build the list of vectors: 10 shifts + 1 random unit vector ---
    g = torch.Generator().manual_seed(seed)
    rvec = torch.randn(d, generator=g, dtype=eigvecs.dtype).to(eigvecs.device)
    rvec = rvec / rvec.norm()
    vectors = [(f"retrain{k:02d}", dhs[k]) for k in range(K)]
    vectors.append(("random", rvec))

    # numpy views for plotting
    var_cum_np, ranks_np = var_cum.cpu().numpy(), ranks.numpy()

    all_rows = []
    for name, v in vectors:
        coeff = v @ eigvecs                    # (d,) projection coefficients
        mass_frac = (coeff.pow(2) / coeff.pow(2).sum())   # per-rank mass fraction
        mass_cum = mass_frac.cumsum(0)
        mc_np = mass_cum.cpu().numpy()

        # ---------- threshold table ----------
        print(f"\n=== {name} : cum-mass thresholds ===")
        print(" level | rank | actual cum-mass | cum-variance")
        for lev in mass_levels:
            hit = (mass_cum >= lev).nonzero()
            idx = int(hit[0]) if len(hit) else d - 1
            print(f"  {lev:.2f}  | {idx:>4} |     {mass_cum[idx]:.4f}      |   {var_cum[idx]:.4f}")
            all_rows.append({"vector": name, "mass_level": lev, "rank": idx,
                             "cumulative_mass": float(mass_cum[idx]),
                             "cumulative_variance": float(var_cum[idx])})

        # ---------- 2-subplot figure ----------
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle(f"Concentration report — {name}", fontsize=13)

        # (1) cumulative mass vs rank (+ variance overlay for context)
        a = ax[0]
        a.plot(ranks_np, mc_np, color="C0", lw=2, label="cum. shift mass")
        a.plot(ranks_np, var_cum_np, color="C2", lw=1.5, ls="-.", label="cum. retain variance")
        a.plot(ranks_np, ranks_np / d, color="k", lw=1, ls=":", label="uniform")
        a.set_xlabel("rank (0 = highest variance)"); a.set_ylabel("cumulative fraction")
        a.set_title("cumulative mass vs rank"); a.legend(fontsize=8); a.set_ylim(0, 1.01)

        # (3) cumulative mass vs cumulative variance
        a = ax[1]
        a.plot(var_cum_np, mc_np, color="C0", lw=2)
        a.plot([0, 1], [0, 1], color="k", lw=1, ls=":", label="y=x (mass tracks variance)")
        a.set_xlabel("cumulative retain-variance fraction")
        a.set_ylabel("cumulative shift-mass fraction")
        a.set_title("cumulative mass vs cumulative variance"); a.legend(fontsize=8)
        a.set_xlim(0, 1); a.set_ylim(0, 1.01)

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(out_dir, f"concentration_{name}.png"), dpi=300)
        plt.close(fig)

    # ---------- save all threshold tables to one CSV ----------
    csv_path = os.path.join(out_dir, "threshold_tables.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["vector", "mass_level", "rank", "cumulative_mass", "cumulative_variance"])
        w.writeheader(); w.writerows(all_rows)

    print(f"\nSaved {K+1} figures + threshold_tables.csv to '{out_dir}/'")
    return all_rows


@torch.no_grad()
def feature_loss_curvature(H_feats, W, eigvecs, mass_ranks, n_random=20, seed=0):
    """
    H_feats : (N_r, d) retain features at theta_o (float64).
    W       : (C, d) final FC weight (logits = W @ h + b), from ORIGINAL model.
    eigvecs : (d, d) centered eigenvectors, columns, descending (from Task C).
    Returns c(u) scalars for the top eigenvectors and random directions.
    """
    dtype = eigvecs.dtype
    Hf = H_feats.to(dtype)
    W = W.to(dtype)                                    # (C, d)
    N, d = Hf.shape

    # build the feature-space GGN/Fisher: mean_x W^T (diag(p) - p p^T) W
    logits = Hf @ W.T                                  # (N, C)
    p = torch.softmax(logits, dim=1)                   # (N, C), original-model probs
    # accumulate G = (1/N) sum_x W^T (diag(p_x) - p_x p_x^T) W  as a (d,d) matrix
    # = W^T [ (1/N) sum_x (diag(p_x) - p_x p_x^T) ] W
    A = torch.diag_embed(p) - p.unsqueeze(2) * p.unsqueeze(1)   # (N, C, C)
    A_mean = A.mean(0)                                 # (C, C)
    G = W.T @ A_mean @ W                               # (d, d) feature-space curvature
    G = 0.5 * (G + G.T)

    def curv(U):                                       # U: (m, d) rows are directions
        U = U / U.norm(dim=1, keepdim=True)            # unit-normalize
        return torch.einsum("md,dk,mk->m", U, G, U)    # (m,) u^T G u per row

    # eigenvector directions (top block + explicit mass-jump ranks)
    top = min(20, d)
    ranks = sorted(set(list(range(top)) + list(mass_ranks)))
    Ve = eigvecs[:, ranks].T                           # (len(ranks), d)
    c_eig = curv(Ve)

    # random unit directions
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(n_random, d, generator=g, dtype=dtype)
    c_rand = curv(R)

    return {"ranks": ranks, "c_eig": c_eig, "c_rand": c_rand, "G": G}


# --------------------------------------------------------------------------- #
# 1. Parallel residual series (signed s_t and RMS Q_t) for ONE reference j
# --------------------------------------------------------------------------- #
def compute_parallel_residuals(h_o, h_rj, h_ft_list):
    """
    Parameters
    ----------
    h_o      : (N, D)  forget-set reps of the original model theta_o  (= epoch-0)
    h_rj     : (N, D)  forget-set reps of retrain reference j, theta_{rj}
    h_ft_list: list of (N, D), length T   fine-tune checkpoints epoch 1 .. T
 
    Returns dict:
        v          : (D,)   unit direction v_j^F
        delta_norm : float  ||Delta_j||
        epochs     : (T+1,) [0, 1, ..., T]
        s          : (T+1,) signed mean parallel residual  s_{t,j},  t = 0..T
        Q          : (T+1,) RMS parallel residual          Q_{t,j},  t = 0..T
        s_norm     : (T+1,) s_{t,j} / s_{0,j}   (== 1.0 at t=0 by construction)
        Q_norm     : (T+1,) Q_{t,j} / Q_{0,j}   (== 1.0 at t=0)
    """
    # float64 for a tight sanity check
    h_o  = torch.as_tensor(h_o).double()
    h_rj = torch.as_tensor(h_rj).double()
    checkpoints = [h_o] + [torch.as_tensor(h).double() for h in h_ft_list]  # t = 0..T
 
    N, D = h_o.shape
    for k, h in enumerate(checkpoints):
        assert h.shape == (N, D), f"checkpoint {k} shape {tuple(h.shape)} != {(N, D)}"
    assert h_rj.shape == (N, D), f"h_rj shape {tuple(h_rj.shape)} != {(N, D)}"
 
    # fixed direction from the original->retrain shift on the forget set
    delta = h_rj.mean(0) - h_o.mean(0)          # (D,)  destination - origin
    delta_norm = torch.linalg.norm(delta)
    v = delta / delta_norm                      # (D,)  unit v_j^F
 
    s_list, Q_list = [], []
    for h_ft in checkpoints:
        r = h_ft - h_rj                         # (N, D) per-sample residual
        s_t = r.mean(0) @ v                      # scalar: (mean residual) . v
        proj = r @ v                             # (N,)  per-sample parallel component
        Q_t = torch.sqrt((proj ** 2).mean())     # scalar: RMS of parallel component
        s_list.append(s_t)
        Q_list.append(Q_t)
 
    s = torch.stack(s_list)                     # (T+1,)
    Q = torch.stack(Q_list)
    T = len(h_ft_list)
    return {
        "v": v,
        "delta_norm": delta_norm.item(),
        "epochs": torch.arange(T + 1),
        "s": s,
        "Q": Q,
        "s_norm": s / s[0],
        "Q_norm": Q / Q[0],
    }


# --------------------------------------------------------------------------- #
# 2. Sanity check:  s_{0,j} == -||Delta_j||
# --------------------------------------------------------------------------- #
def sanity_check_s0(result, atol=1e-6, rtol=1e-5, raise_on_fail=True, label=""):
    """
    Verify the epoch-0 identity for one reference. `result` is the dict from
    compute_parallel_residuals. Returns (passed, s0, target, abs_err).
    Raises AssertionError (to STOP the run) if it fails and raise_on_fail=True.
    """
    s0 = result["s"][0].item()
    target = -result["delta_norm"]              # -||Delta_j||
    abs_err = abs(s0 - target)
    passed = abs_err <= atol + rtol * abs(target)
    if not passed and raise_on_fail:
        raise AssertionError(
            f"[sanity fail {label}] s_0 = {s0:.8e}  expected -||Delta|| = {target:.8e}  "
            f"(|err| = {abs_err:.2e}). STOP — check: reps/model mismatch, sign of "
            f"Delta (must be retrain-minus-original), row misalignment, or feature "
            f"extraction not in the same eval/BN regime."
        )
    return passed, s0, target, abs_err


# --------------------------------------------------------------------------- #
# 3. Two-view plot: signed on linear-y | absolute on log-y
#    Accepts a list of normalized series (one per reference). Works for either
#    the s_t/s_0 list or the Q_t/Q_0 list.
# --------------------------------------------------------------------------- #
def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
 
 
def plot_residual_decay(norm_series, epochs=None, quantity="s",
                        theory_per_epoch=None, title=None,
                        save_path=None, dpi=300, close=True):
    """
    Two-view decay plot for a SINGLE retrain reference.
 
    Left  : signed value,  LINEAR y-axis.
    Right : |value|,       LOG y-axis  (exponential-rate view / fitting).
 
    Parameters
    ----------
    norm_series      : (T+1,) array/tensor — ONE normalized series (== 1.0 at t=0),
                       e.g. results[j]["s_norm"] or results[j]["Q_norm"].
    epochs           : (T+1,) indices; defaults to 0..T.
    quantity         : "s" or "Q"  (labels only).
    theory_per_epoch : optional float. Overlays (rate)**t. For the bare
                       weight-decay law pass exp(-lr*wd*S), S = |D_r| / batch_size.
    title            : figure suptitle (e.g. "Run1  j=3").
    save_path        : if given, save the figure here (per-reference file).
    dpi              : save resolution.
    close            : close the figure after saving (avoids many open figures
                       when looping over the 10 references).
 
    Returns (fig, (axL, axR)).
    """
    s = _to_np(norm_series).astype(float)
    T1 = len(s)
    epochs = np.arange(T1, dtype=float) if epochs is None else _to_np(epochs).astype(float)
    qname = {"s": r"$s_t/s_0$", "Q": r"$Q_t/Q_0$"}.get(quantity, quantity)
 
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5))
    axL.plot(epochs, s, marker="o", ms=4, lw=1.5, color="C0")
    axR.plot(epochs, np.abs(s), marker="o", ms=4, lw=1.5, color="C0", label="data")
 
    if theory_per_epoch is not None:
        theory = theory_per_epoch ** epochs
        axL.plot(epochs, theory, "r--", lw=2, label="WD law")
        axR.plot(epochs, theory, "r--", lw=2,
                 label=rf"WD law $({theory_per_epoch:.5f})^t$")
        axL.legend(fontsize=9)
        axR.legend(fontsize=9, loc="lower left")
 
    axL.axhline(0.0, color="grey", lw=0.8, ls=":")
    axL.set(xlabel="fine-tune epoch t", ylabel=f"signed {qname}",
            title="signed, linear y")
    axL.grid(True, alpha=0.3)
 
    axR.set_yscale("log")
    axR.set(xlabel="fine-tune epoch t", ylabel=f"|{qname}|  (log)",
            title="absolute, log y  (exponential-rate view)")
    axR.grid(True, which="both", alpha=0.3)
 
    if title:
        fig.suptitle(title)
    fig.tight_layout()
 
    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if close:
            plt.close(fig)
    return fig, (axL, axR)


# --------------------------------------------------------------------------- #
# 4. Save per-epoch values to CSV for ONE retrain (long format, appends)
# --------------------------------------------------------------------------- #
_CSV_COLUMNS = [
    "fine_tune_epoch", "retrain_model",
    "s_t", "s_t/s_0", "log|s_t/s_0|",
    "Q_t", "Q_t/Q_0", "log(Q_t/Q_0)",
]
 

def save_residual_csv(result, retrain_name, csv_path):
    """
    Append the per-epoch values for a SINGLE retrain to a CSV (long format:
    one row per fine-tune epoch t = 0..T). Creates the file with a header if it
    does not exist, otherwise appends without a header.
 
    Columns
    -------
    fine_tune_epoch : t = 0..T   (0 = original model theta_o)
    retrain_model   : the retrain identifier you pass in (str)
    s_t             : signed mean parallel residual
    s_t/s_0         : signed normalized (== 1.0 at t=0)
    log|s_t/s_0|    : base 10 log of |s_t/s_0|  (0.0 at t=0; -inf if s_t == 0 exactly)
    Q_t             : RMS parallel residual  (>= 0)
    Q_t/Q_0         : normalized RMS (>= 0, == 1.0 at t=0)
    log(Q_t/Q_0)    : base 10 log of Q_t/Q_0  (0.0 at t=0)
 
    Note: log columns are base 10 log (np.log10).
    """
    epochs = _to_np(result["epochs"])
    s      = _to_np(result["s"])
    s_norm = _to_np(result["s_norm"])
    Q      = _to_np(result["Q"])
    Q_norm = _to_np(result["Q_norm"])
 
    with np.errstate(divide="ignore"):          # log(0) -> -inf, kept honestly
        log_abs_s_norm = np.log10(np.abs(s_norm))
        log_Q_norm     = np.log10(Q_norm)
 
    df = pd.DataFrame({
        "fine_tune_epoch": epochs.astype(int),
        "retrain_model":   str(retrain_name),
        "s_t":             s.astype(float),
        "s_t/s_0":         s_norm.astype(float),
        "log|s_t/s_0|":    log_abs_s_norm.astype(float),
        "Q_t":             Q.astype(float),
        "Q_t/Q_0":         Q_norm.astype(float),
        "log(Q_t/Q_0)":    log_Q_norm.astype(float),
    })[_CSV_COLUMNS]
 
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
 
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    if file_exists:
        # guard against appending under a mismatched schema
        with open(csv_path, "r") as f:
            existing_header = f.readline().strip().split(",")
        if existing_header != _CSV_COLUMNS:
            raise ValueError(
                f"Column mismatch appending to {csv_path}:\n"
                f"  existing: {existing_header}\n  new:      {_CSV_COLUMNS}"
            )
 
    df.to_csv(csv_path, mode="a" if file_exists else "w",
              header=not file_exists, index=False)
    return csv_path