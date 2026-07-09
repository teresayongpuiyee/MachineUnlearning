import math
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


import os
import csv
import numpy as np


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