import numpy as np
from scipy.linalg import eig, eigh
from layers import assemble_HS_multi_alpha, print_H_asymmetry_ranked


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

def sensitivity_test(H,eps=1e-11, trials=20, seed=0):
    rng = np.random.default_rng(seed)
    e, c = eig(H)
    e0 = np.min(np.real(e))
    shifts = []
    for _ in range(trials):
        # dS = rng.standard_normal(S.shape)
        # dS = 0.5*(dS + dS.T)
        dH = rng.standard_normal(H.shape)
        Hp = H + eps*np.linalg.norm(H)*dH/np.linalg.norm(dH)
        # Sp = S + eps*np.linalg.norm(S)*dS/np.linalg.norm(dS)
        e, c = eig(Hp)
        ep = np.min(np.real(e))
        shifts.append(ep - e0)
    return e0, np.std(shifts), np.max(np.abs(shifts))

def cond(S):
    eigS = np.linalg.eigvalsh(S)
    cond = float(eigS.max() / np.abs(eigS).min())
    return cond

def solve_HS_from_layers(layers, alpha, beta, basis_idx, rcond = 1e-15, coords="s12mu"):   # alpha = 0.98

    # H, S = assemble_HS(layers, alpha, beta, coords=coords)
    # H, S, frankenBasis = assemble_HS_multi_alpha(layers, alphas=[0.3, 1.0, 1.5, 2.5], betas = [8.5, 9.0, 9.5], Rms=[1.3, 1.4, 1.5], coords=coords)
    H, S, frankenBasis = assemble_HS_multi_alpha(layers, alphas=[0.5, 1.0, 1.5, 2.5], betas = [8.0], Rms=[1.4], coords=coords)
    return solve_HS(H, S, rcond=rcond), frankenBasis

def solve_HS(H, S, rcond = 1e-15, basis_idx = None):   # alpha = 0.98

    print(np.linalg.norm(H.T@S - S@H)/np.linalg.norm(H)/np.linalg.norm(S))
    if basis_idx is not None:
        scale = 1.0 / (np.maximum(np.abs(H.T@S) + np.abs(S@H), 1e-300))
        print_H_asymmetry_ranked((H.T@S - S@H)*scale, basis_idx)
    H, S, q = diag_rescale_generalized(H, S)
    print("Removing linearly dependent functions...")
    X, keep, w = reduce_by_overlap(S, rcond)
    print(f"    {int(np.count_nonzero(keep) / len(keep) * 100)}% of dimensions ({np.count_nonzero(keep)} functions) kept")
    print(f"Solving...")
    Hp = X.T @ H @ X
    E, Y = eig(Hp)
    C = q[:, None] * (X @ Y)

    print(sensitivity_test(Hp))
    # print(gen_residual_norm(H,S,E[0],C[:, 0]))

    idx = np.argmin(np.real(E))
    E0 = E[idx]
    y0 = Y[:, idx]
    print("residual: ", gen_residual_norm(H,S,E0,X @ y0).max())
    print("E =", np.real(E0))

    return np.real(E), np.real(C), cond(S)


# Todo: 1): Make sure numerical cancellations in integration loop don't destroy significant digits. Sample mu1 from smallest-pos, smallest-neg, next-pos etc. Ensure integrals over mu1^odd cancel. (Especially inv_mu1_2 could lead to catastropic cancellations - test.)