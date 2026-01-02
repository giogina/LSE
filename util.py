import numpy as np
from scipy.linalg import eig
from scipy.linalg import eigh
import scipy.linalg as la

def _rayleigh_quotient(H, S, x):
    num = x.conj().T @ (H @ x)
    den = x.conj().T @ (S @ x)
    return (num / den).item()

def _snorm(S, x):
    v = x.conj().T @ (S @ x)
    # If S is indefinite/noisy, v might go tiny/negative; fall back safely.
    if np.iscomplexobj(v):
        v = v.real
    if v <= 0:
        return np.linalg.norm(x)
    return float(np.sqrt(v))

def make_start_vector_from_topleft(H, S, k=100, sigma=None, which="closest"):
    """
    Build x0 by solving the generalized eigenproblem on the top-left k×k block.
    Returns x0 (length n), lam0 (estimated eigenvalue from the block).
    which:
      - "closest": pick eigenvalue closest to sigma (requires sigma)
      - "min": pick smallest real eigenvalue
      - "max": pick largest real eigenvalue
    """
    import scipy.linalg as la

    n = H.shape[0]
    Hk = np.array(H[:k, :k], copy=False)
    Sk = np.array(S[:k, :k], copy=False)

    # Dense generalized eig on small block
    Ek, Ck = la.eig(Hk, Sk)

    # Decide which eigenpair to use
    if which == "closest":
        if sigma is None:
            raise ValueError("sigma must be provided when which='closest'")
        idx = np.argmin(np.abs(Ek - sigma))
    elif which == "min":
        idx = np.argmin(np.real(Ek))
    elif which == "max":
        idx = np.argmax(np.real(Ek))
    else:
        raise ValueError("which must be 'closest', 'min', or 'max'")

    lam0 = Ek[idx]
    ck = Ck[:, idx]

    # Normalize the small vector in S-norm (if possible)
    nk = _snorm(Sk, ck)
    if nk == 0 or not np.isfinite(nk):
        ck = ck / (np.linalg.norm(ck) + 1e-300)
    else:
        ck = ck / nk

    # Embed into full vector
    x0 = np.zeros(n, dtype=complex if (np.iscomplexobj(H) or np.iscomplexobj(S) or np.iscomplexobj(ck)) else float)
    x0[:k] = ck
    return x0, lam0

def shift_invert_target_eigpair(
    H, S, sigma,
    x0,
    max_iter=50,
    tol=1e-10,
    update_sigma=False,
    regularize_mu=0.0,
    normalize="S",   # "S" or "2"
    verbose=True,
):
    """
    Target eigenpair near sigma for H x = lambda S x by shift-invert inverse iteration.

    Iteration:
      Solve (H - sigma S + mu I) y = S x
      x <- y / ||y||
      lambda <- (x^T H x) / (x^T S x)

    If update_sigma=True, sets sigma <- lambda each iteration (Rayleigh quotient iteration style).
    That can converge very fast near the solution, but can also destabilize if sigma is not close enough.
    """
    import scipy.linalg as la

    H = np.asarray(H)
    S = np.asarray(S)
    n = H.shape[0]
    assert H.shape == (n, n) and S.shape == (n, n)

    x = np.asarray(x0).astype(complex if (np.iscomplexobj(H) or np.iscomplexobj(S) or np.iscomplexobj(x0)) else float)
    # Basic normalization
    if normalize == "S":
        nx = _snorm(S, x)
    else:
        nx = np.linalg.norm(x)
    x = x / (nx + 1e-300)

    lam = _rayleigh_quotient(H, S, x)
    if verbose:
        r = H @ x - lam * (S @ x)
        print(f"[init]  sigma={sigma}  lam={lam}  ||r||2={np.linalg.norm(r):.3e}")

    for it in range(1, max_iter + 1):
        # Build/factor shifted matrix
        sig = lam if update_sigma else sigma
        K = H - sig * S
        if regularize_mu != 0.0:
            K = K + regularize_mu * np.eye(n, dtype=K.dtype)

        # LU factorization (dense)
        lu, piv = la.lu_factor(K)

        # Solve (H - sig S) y = S x
        rhs = S @ x
        y = la.lu_solve((lu, piv), rhs)

        # Normalize
        if normalize == "S":
            ny = _snorm(S, y)
        else:
            ny = np.linalg.norm(y)
        x = y / (ny + 1e-300)

        # Update eigenvalue estimate
        lam_new = _rayleigh_quotient(H, S, x)

        # Residual
        r = H @ x - lam_new * (S @ x)
        rnorm = np.linalg.norm(r)

        if verbose:
            print(f"[{it:02d}] sig={sig}  lam={lam_new}  ||r||2={rnorm:.3e}")

        # Convergence test (residual-based)
        if rnorm <= tol:
            return lam_new, x, {"iters": it, "residual_norm": rnorm}

        # Stagnation / nan guard
        if not np.isfinite(rnorm) or not np.isfinite(lam_new.real) or not np.isfinite(lam_new.imag):
            return lam_new, x, {"iters": it, "residual_norm": rnorm, "failed": True}

        lam = lam_new

    return lam, x, {"iters": max_iter, "residual_norm": rnorm, "converged": False}


def s_deflate(H, S, rcond=1e-12, symmetrize=True):
    if symmetrize:
        H = 0.5*(H + H.T)
        S = 0.5*(S + S.T)

    lamS, U = la.eigh(S)
    keep = lamS >= rcond * lamS.max()
    Uk = U[:, keep]
    Sk = Uk.T @ S @ Uk          # ~diag(lamS[keep])
    Hk = Uk.T @ H @ Uk
    return Hk, Sk, Uk, lamS, keep

def s_norm(v, S):
    return np.sqrt(np.real(v.T @ (S @ v)))

def rayleigh(H, S, x):
    num = np.real(x.T @ (H @ x))
    den = np.real(x.T @ (S @ x))
    return num / den

def residual_minimize_generalized(
    H, S, lam, x0,
    *,
    iters=10,
    rcond=1e-12,
    symmetrize=True,
    damping=0.7,
    ridge=0.0,
):
    """
    Residual minimization / JD-like correction in deflated subspace.
    Returns improved (lam, x) in original space.

    damping: step size <1 helps if updates overshoot.
    ridge: optional Tikhonov regularization added to (H-lam S) solve: (A + ridge*I).
    """
    Hk, Sk, Uk, lamS, keep = s_deflate(H, S, rcond=rcond, symmetrize=symmetrize)

    # project initial guess to reduced space
    x = Uk.T @ x0
    x = x.astype(np.float64, copy=False)
    x /= (s_norm(x, Sk) + 1e-300)

    lam = float(lam)

    I = np.eye(Hk.shape[0])

    for _ in range(iters):
        # update lambda using Rayleigh quotient (usually stabilizes)
        lam = rayleigh(Hk, Sk, x)

        Hx = Hk @ x
        Sx = Sk @ x
        r  = Hx - lam * Sx

        # Convergence check: relative residual
        relr = la.norm(r) / (la.norm(Hx) + 1e-300)
        # print("lam", lam, "relr", relr)

        # Build projectors implicitly:
        # We need δ satisfying x^T S δ = 0.
        # Use projector P(v) = v - x*(x^T S v)
        def P(v):
            return v - x * (np.real(x.T @ (Sk @ v)))

        # Define linear operator for the projected correction equation:
        # Aδ = P( (H - lam S) P(δ) )
        A = Hk - lam * Sk
        if ridge != 0.0:
            A = A + ridge * I

        # Solve Atilde δ = -r in projected space by forming and solving dense system:
        # We can explicitly build Atilde = P(A(P(e_j))) basis-wise,
        # but that's O(N^3) extra. Since N is only ~200-1000, it's still okay.
        # Faster approach: solve unprojected then project and re-solve a couple times.

        # Practical dense JD-ish trick:
        # solve A y = -r, then δ = P(y), then enforce again by one refinement solve.
        y = la.solve(A, -r, assume_a='gen', overwrite_a=False, overwrite_b=False, check_finite=False)
        delta = P(y)

        # optional refinement to respect constraint better:
        # Solve A y2 = -P(A delta) + (-r)  (i.e., correct for projection error)
        corr_rhs = -r - P(A @ delta)
        y2 = la.solve(A, corr_rhs, assume_a='gen', check_finite=False)
        delta = delta + P(y2)

        # damped update
        x = x + damping * delta
        x /= (s_norm(x, Sk) + 1e-300)

    # backtransform to full space
    x_full = Uk @ x
    return lam, x_full

def inverse_iteration_generalized(H, S, lam, x0=None, iters=10, rcond=1e-12):
    H = np.asarray(H); S = np.asarray(S)
    N = H.shape[0]

    # Deflate using S-eigendecomposition
    evalS, U = la.eigh(S)
    keep = evalS >= rcond * evalS.max()
    Uk = U[:, keep]
    dk = evalS[keep]

    # Reduced matrices
    Hr = Uk.T @ H @ Uk
    Sr = Uk.T @ S @ Uk   # should be diag(dk), up to roundoff

    # Work in reduced space
    if x0 is None:
        xr = np.random.default_rng(0).standard_normal(Hr.shape[0])
    else:
        xr = Uk.T @ x0

    # S-normalize helper
    def snorm(v):
        return np.sqrt(np.real(v.T @ (Sr @ v)))

    # Pre-factor (Hr - lam Sr)
    A = Hr - lam*Sr
    lu, piv = la.lu_factor(A)

    for _ in range(iters):
        rhs = Sr @ xr
        xr = la.lu_solve((lu, piv), rhs)
        xr /= (snorm(xr) + 1e-300)

        # optional Rayleigh update (usually improves)
        lam = np.real((xr.T @ (Hr @ xr)) / (xr.T @ (Sr @ xr)))

    # Backtransform
    x = Uk @ xr
    return lam, x


def solve_gen_eig_deflated(
    H: np.ndarray,
    S: np.ndarray,
    *,
    rcond: float = 1e-12,
    assume_S_spd: bool = True,
    assume_H_symmetric: bool = False,
    sort: bool = True,
):
    """
    Solve H c = E S c with S-deflation + S^{-1/2} transform.

    Parameters
    ----------
    H, S : (N,N) arrays
        Generalized eigenproblem matrices. S should be symmetric (SPD-ish).
    rcond : float
        Relative cutoff on eigenvalues of S. Keep modes with lam/lam_max >= rcond.
        If your S spectrum is smooth, try scanning rcond: 1e-10 ... 1e-16.
    assume_S_spd : bool
        If True, enforce symmetry in S and use eigh.
    assume_H_symmetric : bool
        If True, solve in transformed space with eigh (fast, stable).
        If False, use eig (works for non-symmetric transformed matrix).
    sort : bool
        Sort eigenpairs by real(E).

    Returns
    -------
    E : (K,) complex
    C : (N,K) complex
        Coefficients in the ORIGINAL basis (padded to full length N, but living
        in the kept subspace).
    info : dict
        Diagnostic info: kept mask, S eigenvalues, projected dimension, residuals.
    """
    H = np.asarray(H)
    S = np.asarray(S)
    N = H.shape[0]
    assert H.shape == (N, N) and S.shape == (N, N)

    # Symmetrize S defensively (tiny asymmetry is common from numerics)
    if assume_S_spd:
        S_work = 0.5 * (S + S.T)
    else:
        S_work = S

    # Eigendecompose S
    # S = U diag(lam) U^T
    lam, U = np.linalg.eigh(S_work)

    lam_max = np.max(lam)
    if lam_max <= 0:
        raise ValueError("S does not look SPD (max eigenvalue <= 0).")

    keep = lam >= (rcond * lam_max)
    K = int(np.sum(keep))
    if K == 0:
        raise ValueError(f"All S-eigenmodes dropped at rcond={rcond:g}. Try smaller rcond.")

    Uk = U[:, keep]
    lamk = lam[keep]

    # Build S^{-1/2} on kept subspace: X = Uk * diag(lamk^{-1/2}) * Uk^T
    inv_sqrt_lamk = 1.0 / np.sqrt(lamk)

    # We do the transform implicitly to avoid forming full X:
    # A = (Uk^T H Uk) with scaling on both sides by inv_sqrt_lamk
    Hr = Uk.T @ H @ Uk                       # (K,K)
    # A = D^{-1/2} Hr D^{-1/2} where D = diag(lamk)
    A = (inv_sqrt_lamk[:, None] * Hr) * inv_sqrt_lamk[None, :]

    # Solve standard eigenproblem
    if assume_H_symmetric:
        # Force symmetry if it's supposed to be symmetric (numerical noise)
        A_work = 0.5 * (A + A.T)
        E, Y = np.linalg.eigh(A_work)         # Y: (K,K), orthonormal
    else:
        E, Y = np.linalg.eig(A)               # for non-symmetric

    # Backtransform: c = Uk * D^{-1/2} * y
    # where y is eigenvector in transformed space
    Z = inv_sqrt_lamk[:, None] * Y            # (K,K)
    C = Uk @ Z                                 # (N,K)

    if sort:
        idx = np.argsort(np.real(E))
        E = E[idx]
        C = C[:, idx]

    # Diagnostics: relative residual norms
    # r = Hc - E S c
    HC = H @ C
    SC = S @ C
    R = HC - SC * E[None, :]
    # normalize by ||H||*||c|| (rough)
    Hn = np.linalg.norm(H, ord=2)
    Cn = np.linalg.norm(C, axis=0)
    rn = np.linalg.norm(R, axis=0) / (Hn * Cn + 1e-300)

    info = {
        "rcond": rcond,
        "N": N,
        "K": K,
        "S_eigs": lam,
        "S_lam_max": lam_max,
        "keep_mask": keep,
        "residual_rel": rn,
    }
    return E, C, info


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

    return Hs, Ss



def inspect_small_overlap_eigenvectors(S, thresh=1e-7, top_thr=1e7, top_k=10):
    """
    Print eigenvectors of overlap matrix S whose eigenvalues < thresh,
    and list indices of largest |components| in each eigenvector.

    Parameters
    ----------
    S : (N,N) ndarray
        Overlap matrix (assumed symmetric).
    thresh : float
        Eigenvalue cutoff.
    top_k : int
        Number of largest components (by abs value) to report.
    """

    # Always symmetrize first
    S = 0.5 * (S + S.T)

    w, U = np.linalg.eigh(S)

    small_idx = np.where(w < thresh)[0]

    print(f"Found {len(small_idx)} eigenvalues < {thresh:g}\n")

    for n in small_idx:
        eigval = w[n]
        vec = U[:, n]

        print("=" * 80)
        print(f"Eigenvalue {n}: {eigval:.6e}")

        # full eigenvector (optional but you asked for it)
        print("\nEigenvector:")
        np.set_printoptions(precision=6, suppress=False)
        print(vec)

        # largest components by magnitude
        absvec = np.abs(vec)
        idx = np.argsort(absvec)[::-1][:top_k]

        print(f"\nTop {top_k} components (by |value|):")
        for i in idx:
            print(f"  index {i:4d} :  {vec[i]: .6e}")

    small_idx = np.where(w > top_thr)[0]

    print(f"Found {len(small_idx)} eigenvalues > {top_thr:g}\n")

    for n in small_idx:
        eigval = w[n]
        vec = U[:, n]

        print("=" * 80)
        print(f"Eigenvalue {n}: {eigval:.6e}")

        # full eigenvector (optional but you asked for it)
        print("\nEigenvector:")
        np.set_printoptions(precision=6, suppress=False)
        print(vec)

        # largest components by magnitude
        absvec = np.abs(vec)
        idx = np.argsort(absvec)[::-1][:top_k]

        print(f"\nTop {top_k} components (by |value|):")
        for i in idx:
            print(f"  index {i:4d} :  {vec[i]: .6e}")

    return w, U, small_idx



def solve_gen_eig_project_only(H, S, rcond=1e-12, sort=True):
    """
    Deflate near-linear dependencies by projecting onto the well-conditioned
    eigen-subspace of S. No whitening.

    Returns
    -------
    E : (k,) eigenvalues
    C_full : (n,k) eigenvectors in original basis (columns)
    info : diagnostics dict
    """
    H = np.array(H, dtype=float, copy=True)
    S = np.array(S, dtype=float, copy=True)

    S = 0.5 * (S + S.T)

    w, V = eigh(S)  # ascending
    w_max = float(np.max(w))
    if w_max <= 0:
        raise ValueError("S has non-positive max eigenvalue; not an overlap-like matrix.")

    keep = w >= (rcond * w_max)
    if not np.any(keep):
        raise ValueError("All directions cut; decrease rcond.")

    Vk = V[:, keep]                 # n x k
    wk = w[keep]

    Hk = Vk.T @ H @ Vk              # k x k
    Sk = Vk.T @ S @ Vk              # k x k  (≈ diag(wk))

    eigvals = np.linalg.eigvalsh(Sk)
    #
    # # optional diagnostics
    print((eigvals))
    print("min eig:", eigvals.min())
    print("max eig:", eigvals.max())
    print("cond:", eigvals.max() / eigvals.min())

    E, Ck = eig(Hk, Sk)             # columns of Ck
    E = np.real_if_close(E)

    # Recover full-length eigenvectors (columns)
    C_full = Vk @ Ck

    if sort:
        idx = np.argsort(np.real(E))
        E = E[idx]
        C_full = C_full[:, idx]

    info = {
        "n": S.shape[0],
        "k": int(np.sum(keep)),
        "num_deflated": int(np.sum(~keep)),
        "cutoff": float(rcond * w_max),
        "w_max": w_max,
        "w_min_kept": float(np.min(wk)),
        "cond_est_kept": float(np.max(wk) / np.min(wk)),
        "min_w": float(np.min(w)),
        "num_negative_w": int(np.sum(w < 0)),
    }
    print(info)
    return E, C_full

