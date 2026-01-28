import numpy as np
from scipy.linalg import eig, eigh

def rel_asym(A):
    nrm = np.linalg.norm(A)
    if nrm == 0.0:
        return 0.0
    return np.linalg.norm(A - A.T) / nrm

def assemble_HS(SH_layers, alpha, beta, debug = False, coords="s12mu"):
    S = np.zeros_like(next(iter(SH_layers["S"].values())), dtype=np.float64)
    H = np.zeros_like(S, dtype=np.float64)

    for (rAB0, s0), Sl in SH_layers["S"].items():
        if coords.endswith("_morse"):
            exps = np.exp(-2 * alpha * s0 - 2 * beta * (rAB0-1.4011)**2)
        else:
            exps = np.exp(-2 * alpha * s0 - 2 * beta * rAB0)

        S += Sl * exps
        Hl = np.zeros_like(H)
        Hl += SH_layers["H_1"][rAB0, s0]
        Hl += SH_layers["H_alpha"][rAB0, s0] * alpha
        Hl += SH_layers["H_alpha2"][rAB0, s0] * alpha**2
        Hl += SH_layers["H_alphabeta"][rAB0, s0] * alpha*beta
        Hl += SH_layers["H_beta"][rAB0, s0] * beta
        Hl += SH_layers["H_beta2"][rAB0, s0] * beta**2
        H += Hl * exps

    if debug:
        print("H asym rel:", rel_asym(H))
        print("S asym rel:", rel_asym(S))
    return H, S


def diag_rescale_generalized(H, S, eps=1e-300):
    """
    Diagonal re-weighting (NOT whitening):
        W = diag(1/sqrt(diag(S)))
        S' = W S W
        H' = W H W

    Returns: Hs, Ss, W (as 1D vector of diagonal entries), d (diag(S))
    """

    d = np.diag(S).copy()

    # guard against zeros/negatives on the diagonal (shouldn't happen, but can numerically)
    if np.any(d <= 0):
        bad = np.where(d <= 0)[0][:10]
        raise ValueError(f"Non-positive diagonal entries in S at indices {bad}. "
                         f"Min diag(S)={d.min():.3e}. Fix basis / integration / symmetrize S first.")

    w = 1.0 / np.sqrt(np.maximum(d, eps))   # vector of W diagonal entries

    # Diagonal scaling without forming W explicitly  ( (W S W)_{ij} = w_i * S_{ij} * w_j )
    Ss = (S * w[None, :]) * w[:, None]
    Hs = (H * w[None, :]) * w[:, None]

    return Hs, Ss, w

def print_near_deps_from_S(S, basis_idx, eig_abs_cut=1e-15, topk=20, max_sets=20):
    """
    Print near-linear dependencies inferred from eigenvectors of S
    with |eigenvalue| < eig_abs_cut.

    For each such eigenvector, print the top-k largest absolute coefficients
    and the corresponding basis_idx entries.
    """
    Ssym = 0.5 * (S + S.T)
    w, U = np.linalg.eigh(Ssym)   # ascending

    tiny_ids = np.where(np.abs(w) < eig_abs_cut)[0]
    print(f"[deps] dim={S.shape[0]}  |eig|<{eig_abs_cut:g}: {len(tiny_ids)}")

    if len(tiny_ids) == 0:
        return

    for rank, k in enumerate(tiny_ids[:max_sets]):
        uk = U[:, k]
        ak = np.abs(uk)

        idx = np.argsort(-ak)[:topk]

        print(f"\n[deps #{rank}] eig={w[k]: .3e}")
        for i in idx:
            print(
                f"  i={i:4d}  coef={uk[i]: .3e}  |coef|={ak[i]:.3e}  "
                f"basis_idx={basis_idx[i]}"
            )


def reduce_by_overlap(S, rcond=1e-12):
    S = 0.5*(S + S.T)
    w, U = eigh(S)                      # ascending
    wmax = w.max()
    keep = w > rcond*wmax               # threshold

    Ur = U[:, keep]
    wr = w[keep]
    X = Ur / np.sqrt(wr)                # whitening map: c = X y, X^T S X = I
    return X, keep, w

def gen_residual_norm(H,S,E,c):
    r = H@c - E*(S@c)
    return np.linalg.norm(r) / (np.linalg.norm(H@c) + abs(E)*np.linalg.norm(S@c))

def sensitivity_test(H,S,eps=1e-14, trials=20, seed=0):
    rng = np.random.default_rng(seed)
    e, c = eig(H, S)
    e0 = np.min(np.real(e))
    shifts = []
    for _ in range(trials):
        dS = rng.standard_normal(S.shape)
        dS = 0.5*(dS + dS.T)
        dH = rng.standard_normal(H.shape)
        Hp = H + eps*np.linalg.norm(H)*dH/np.linalg.norm(dH)
        Sp = S + eps*np.linalg.norm(S)*dS/np.linalg.norm(dS)
        e, c = eig(Hp, Sp)
        ep = np.min(np.real(e))
        shifts.append(ep - e0)
    return e0, np.std(shifts), np.max(np.abs(shifts))

def cond(S):
    eigS = np.linalg.eigvalsh(S)
    cond = float(eigS.max() / np.abs(eigS).min())
    return cond

def solve_HS(layers, alpha, beta, basis_idx, rcond = 1e-15, coords="s12mu"):   # alpha = 0.98

    H, S = assemble_HS(layers, alpha, beta, coords=coords) # todo: how come the min-eigS changes so much with alpha? Tiny function?
    H, S, q = diag_rescale_generalized(H, S)
    # print_near_deps_from_S(S, basis_idx)
    # print(S)

    X, keep, w = reduce_by_overlap(S, rcond)
    print(f"{int(np.count_nonzero(keep) / len(keep) * 100)}% of dimensions ({np.count_nonzero(keep)} functions) kept")

    Hp = X.T @ H @ X
    E, Y = eig(Hp)
    C = q[:, None] * (X @ Y)

    return np.real(E), np.real(C), cond(S)

    # idx = np.argmin(np.real(E))
    # E0 = E[idx]
    # y0 = Y[:, idx]
    # print("residual: ", gen_residual_norm(H,S,E0,X @ y0).max())
    # print("E =", np.real(E0))


# Todo: 1): Make sure numerical cancellations in integration loop don't destroy significant digits. Sample mu1 from smallest-pos, smallest-neg, next-pos etc. Ensure integrals over mu1^odd cancel. (Especially inv_mu1_2 could lead to catastropic cancellations - test.)