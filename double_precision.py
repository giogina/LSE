import numpy as np

SPLIT = np.float64(134217729.0)  # 2^27 + 1 for float64
EPS   = np.finfo(np.float64).eps

def dd_from(x):
    x = np.asarray(x, dtype=np.float64)
    return x, np.zeros_like(x)

def dd_value(hi, lo):
    return hi + lo

# ---------- error-free primitives (vectorized) ----------

def two_sum(a, b):
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e

def split(a):
    c = SPLIT * a
    ah = c - (c - a)
    al = a - ah
    return ah, al

def two_prod(a, b):
    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e

def renorm(hi, lo):
    # normalize so lo is the residual
    s, e = two_sum(hi, lo)
    return s, e

# ---------- DD ops (vectorized) ----------

def dd_add(a, b):
    s, e = two_sum(a[0], b[0])
    e = e + a[1] + b[1]
    return renorm(s, e)

def dd_subs(a, b):
    s, e = two_sum(a[0], -b[0])
    e = e + a[1] - b[1]
    return renorm(s, e)

def dd_minus(a):
    return -a[0], -a[1]

def dd_mul(a, b):
    p, e = two_prod(a[0], b[0])
    e = e + a[0] * b[1] + a[1] * b[0]
    e = e + a[1] * b[1]
    return renorm(p, e)

def dd_mul_exact_scalar(a, b):
    p, e = two_prod(a[0], b)
    e = e + a[1] * b
    return renorm(p, e)

def dd_square(x):
    """float64 -> DD for x*x (captures product rounding error)"""
    hi, lo = dd_mul(x, x)
    return renorm(hi, lo)

def dd_inverse(a, iters=2):
    """
    Double-double reciprocal: returns (x_hi, x_lo) ~ 1/(a_hi+a_lo).
    Vectorized (arrays/scalars). iters=2 is usually enough for ~DD precision.
    """

    x = (1.0 / a[0], np.zeros_like(a[0]))  # Initial float64 guess
    one = (np.float64(1.0), np.float64(0.0))

    for _ in range(iters):
        ax = dd_mul(a, x)  # ax = a * x
        r = dd_subs(one, ax)   # r = 1 - ax
        xr = dd_mul(x, r)   # x = x + x*r
        x = dd_add(x, xr)

    return renorm(x[0], x[1])

def dd_sqrt(a, iters=2):
    """
    Double-double square root for nonnegative a = (hi, lo).
    Vectorized (arrays/scalars). iters=2 is usually enough for ~DD precision.
    Returns (x_hi, x_lo) ~ sqrt(a).
    """
    a_hi = np.asarray(a[0], dtype=np.float64)
    a_lo = np.asarray(a[1], dtype=np.float64)

    # Initial float64 guess
    y0 = np.sqrt(a_hi + a_lo)
    x = (y0, np.zeros_like(y0))

    # Handle a == 0 (avoid 0 inverse)
    zero = (a_hi == 0.0) & (a_lo == 0.0)
    if np.any(zero):
        # ensure x starts at 0 there
        x = (np.where(zero, 0.0, x[0]), np.where(zero, 0.0, x[1]))

    half = (np.float64(0.5), np.float64(0.0))

    # Newton: x_{n+1} = 0.5 * (x_n + a / x_n)
    for _ in range(iters):
        invx = dd_inverse(x)        # 1/x
        a_over_x = dd_mul(a, invx)  # a/x
        x = dd_mul(dd_add(x, a_over_x), half)

        if np.any(zero):
            x = (np.where(zero, 0.0, x[0]), np.where(zero, 0.0, x[1]))

    return renorm(x[0], x[1])

def dd_pow_int(x, n: int):
    """
    Double-double integer power: returns x**n for integer n (can be negative).
    Vectorized over x (scalars or arrays).

    x: (hi, lo) DD tuple
    n: int

    Returns: (y_hi, y_lo) DD tuple
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError("dd_pow_int: n must be an integer")
    if n == 0:
        return dd_from(1.0)

    x = (np.asarray(x[0], dtype=np.float64), np.asarray(x[1], dtype=np.float64))

    if n == 0:
        return (np.ones_like(x[0], dtype=np.float64), np.zeros_like(x[0], dtype=np.float64))

    if n < 0:
        x = dd_inverse(x)
        n = -int(n)

    # exponentiation by squaring
    y = (np.ones_like(x[0], dtype=np.float64), np.zeros_like(x[0], dtype=np.float64))
    b = x
    e = int(n)

    while e:
        if e & 1:
            y = dd_mul(y, b)
        e >>= 1
        if e:
            b = dd_mul(b, b)

    return renorm(y[0], y[1])

def dd_pow_table(x, n_max: int):
    """
    Build DD power table for a DD base x:
        out[e] = x**e  for e = 0..n_max
    using a fast recurrence (sequential multiply), vectorized over x.

    x: (hi, lo) DD tuple (scalar or array)
    n_max: nonnegative int

    Returns: (pow_hi, pow_lo) with shape (n_max+1, *x.shape)
    """
    if not isinstance(n_max, (int, np.integer)) or n_max < 0:
        raise ValueError("dd_pow_table: n_max must be a nonnegative integer")

    x = (np.asarray(x[0], dtype=np.float64), np.asarray(x[1], dtype=np.float64))
    shp = x[0].shape
    E = int(n_max) + 1

    pow_hi = np.empty((E,) + shp, dtype=np.float64)
    pow_lo = np.empty_like(pow_hi)

    # x^0 = 1
    pow_hi[0] = 1.0
    pow_lo[0] = 0.0
    if n_max == 0:
        return pow_hi, pow_lo

    # x^1 = x
    pow_hi[1] = x[0]
    pow_lo[1] = x[1]

    cur = (x[0], x[1])
    for e in range(2, E):
        cur = dd_mul(cur, x)
        pow_hi[e] = cur[0]
        pow_lo[e] = cur[1]

    return pow_hi, pow_lo

def dd_log(a, terms=5):
    """
    Double-double natural log for positive a = (hi, lo).

    Uses:
        log(hi+lo) = log(hi) + log1p(lo/hi)
    with log1p(x) evaluated by a short alternating Taylor series in DD:
        log1p(x) = x - x^2/2 + x^3/3 - x^4/4 + ...

    This is appropriate for DD numbers where |lo/hi| ~ 1e-16.
    Vectorized over arrays/scalars.

    terms: number of Taylor terms (>=1). 5 is plenty.
    """
    hi = np.asarray(a[0], dtype=np.float64)
    lo = np.asarray(a[1], dtype=np.float64)

    # base result arrays
    out_hi = np.empty_like(hi, dtype=np.float64)
    out_lo = np.empty_like(hi, dtype=np.float64)

    # masks
    pos = (hi > 0.0)  # we require positive hi; with DD, hi carries the sign
    zero = (hi == 0.0) & (lo == 0.0)

    # log(0) = -inf
    if np.any(zero):
        out_hi[zero] = -np.inf
        out_lo[zero] = 0.0

    if not np.any(pos):
        return (out_hi, out_lo)

    # Work only on positive entries
    hip = hi[pos]
    lop = lo[pos]

    # log(hi) in float64, promoted to DD
    log_hi_dd = dd_from(np.log(hip))

    # x = lo/hi in DD
    inv_hi_dd = dd_inverse(dd_from(hip))
    x = dd_mul(dd_from(lop), inv_hi_dd)  # DD

    # Taylor series for log1p(x) in DD
    # s = x
    s = x
    xpow = x  # x^k

    # add terms k=2..terms
    # term_k = (-1)^(k+1) * x^k / k
    for k in range(2, int(terms) + 1):
        xpow = dd_mul(xpow, x)  # x^k
        term = dd_mul_exact_scalar(xpow, 1.0 / float(k))
        if (k % 2) == 0:  # even k -> subtract
            s = dd_subs(s, term)
        else:             # odd k -> add
            s = dd_add(s, term)

    # log(a) = log(hi) + log1p(lo/hi)
    y = dd_add(log_hi_dd, s)
    y = renorm(y[0], y[1])

    out_hi[pos] = y[0]
    out_lo[pos] = y[1]
    return (out_hi, out_lo)

def dd_exp(a):
    """
    Double-double exponential.
    Works well when |lo| << |hi|.
    """

    hi = a[0]
    lo = a[1]

    # exp(hi) in float64
    exp_hi = np.exp(hi)
    exp_hi_dd = dd_from(exp_hi)

    # exp(lo) via 1 + lo + lo^2/2  (lo ~ 1e-16)
    lo_dd = (np.zeros_like(lo), lo)
    lo2   = dd_square(lo_dd)
    corr  = dd_add(dd_from(1.0), lo_dd)
    corr  = dd_add(corr, dd_mul_exact_scalar(lo2, 0.5))

    return dd_mul(exp_hi_dd, corr)


def dd_tensordot_a0(exps, admat):
    """
    ad_S = sum_a exps[a] * admat[a,...] in DD.
    exps: (Na,) DD
    admat: (Na, ...) DD
    returns (...) DD
    """
    ex_hi, ex_lo = exps
    A_hi, A_lo   = admat

    Na = ex_hi.shape[0]
    out_hi = np.zeros(A_hi.shape[1:], dtype=np.float64)
    out_lo = np.zeros_like(out_hi)

    for a in range(Na):
        term = dd_mul((ex_hi[a], ex_lo[a]), (A_hi[a], A_lo[a]))
        out_hi, out_lo = dd_add((out_hi, out_lo), term)

    return (out_hi, out_lo)

def dd_kron(A, B):
    """
    Kronecker product in DD.
    A, B are DD tuples.
    Returns DD tuple.
    """
    A_hi, A_lo = A
    B_hi, B_lo = B

    K_hi = np.kron(A_hi, B_hi)

    # full expansion of (hi+lo)⊗(hi+lo) - hi⊗hi
    K_lo = (np.kron(A_hi, B_lo) +
            np.kron(A_lo, B_hi) +
            np.kron(A_lo, B_lo))

    return renorm(K_hi, K_lo)


import numpy as np

def dd_kron_reduced(R_dd, A_dd, c=0.5):
    """
    Build reduced K_dd for basis:
        [ spec ] + [ all (rhl>=2) ⊗ (all ad) ]

    where:
        spec = (rhl0 ⊗ ad0) + c * (rhl1 ⊗ ad1)
        rhl0 index 0 (h=0,l=0)
        rhl1 index 1 (h=1,l=0)
        ad0 = 0, ad1 = 1

    R_dd: DD tuple (R_hi, R_lo) shape (nrhl,nrhl)
    A_dd: DD tuple (A_hi, A_lo) shape (NK,NK)

    Returns: DD tuple (K_hi, K_lo) shape (1 + (nrhl-2)*NK, 1 + (nrhl-2)*NK)
    """
    R_hi, R_lo = R_dd
    A_hi, A_lo = A_dd

    nrhl = int(R_hi.shape[0])
    NK = int(A_hi.shape[0])
    n2 = nrhl - 2
    Nred = 1 + n2 * NK

    out_hi = np.zeros((Nred, Nred), dtype=np.float64)
    out_lo = np.zeros_like(out_hi)

    # --- (1) big block: rhl>=2 vs rhl>=2 ---
    if n2 > 0:
        R22 = (R_hi[2:, 2:], R_lo[2:, 2:])
        K22_hi, K22_lo = dd_kron(R22, A_dd)  # uses your fast vectorized kron
        out_hi[1:, 1:] = K22_hi
        out_lo[1:, 1:] = K22_lo

    # Convenience: A rows/cols for ad0/ad1 (as float64 vectors)
    a0_row_hi, a0_row_lo = A_hi[0, :], A_lo[0, :]
    a1_row_hi, a1_row_lo = A_hi[1, :], A_lo[1, :]
    a0_col_hi, a0_col_lo = A_hi[:, 0], A_lo[:, 0]
    a1_col_hi, a1_col_lo = A_hi[:, 1], A_lo[:, 1]

    # --- (2) special-special corner (scalar) ---
    # Kss = R00*A00 + c*R01*A01 + c*R10*A10 + c^2*R11*A11  (DD accurate scalar mult)
    # Use dd_mul_exact_scalar on 1x1 DD pieces (fast enough).
    A00 = (A_hi[0:1, 0:1], A_lo[0:1, 0:1])
    A01 = (A_hi[0:1, 1:1+1], A_lo[0:1, 1:1+1])
    A10 = (A_hi[1:1+1, 0:1], A_lo[1:1+1, 0:1])
    A11 = (A_hi[1:1+1, 1:1+1], A_lo[1:1+1, 1:1+1])

    Kss = dd_mul_exact_scalar(A00, float(R_hi[0, 0]))
    Kss = dd_add(Kss, dd_mul_exact_scalar(A01, float(c * R_hi[0, 1])))
    Kss = dd_add(Kss, dd_mul_exact_scalar(A10, float(c * R_hi[1, 0])))
    Kss = dd_add(Kss, dd_mul_exact_scalar(A11, float((c * c) * R_hi[1, 1])))

    out_hi[0, 0] = Kss[0][0, 0]
    out_lo[0, 0] = Kss[1][0, 0]

    if n2 == 0:
        return renorm(out_hi, out_lo)

    # --- helper: DD multiply (n2,NK) outer products fast using two_prod ---
    def dd_outer_scalar_vec(rvec, a_hi, a_lo):
        """
        rvec: (n2,) float64
        a_hi/a_lo: (NK,) float64
        returns (hi,lo) shape (n2,NK) for rvec[:,None] * (a_hi+a_lo)[None,:]
        """
        # hi product + rounding error
        hi, err = two_prod(rvec[:, None], a_hi[None, :])  # (n2,NK)
        lo = err + rvec[:, None] * a_lo[None, :]
        return hi, lo

    # --- (3) special row vs big block: length n2*NK ---
    r0 = R_hi[0, 2:]  # (n2,)
    r1 = R_hi[1, 2:]  # (n2,)

    top0_hi, top0_lo = dd_outer_scalar_vec(r0, a0_row_hi, a0_row_lo)
    top1_hi, top1_lo = dd_outer_scalar_vec(r1, a1_row_hi, a1_row_lo)

    top_hi = top0_hi + c * top1_hi
    top_lo = top0_lo + c * top1_lo

    out_hi[0, 1:] = top_hi.reshape(-1)
    out_lo[0, 1:] = top_lo.reshape(-1)

    # --- (4) big block vs special column: length n2*NK ---
    c0v = R_hi[2:, 0]  # (n2,)
    c1v = R_hi[2:, 1]  # (n2,)

    left0_hi, left0_lo = dd_outer_scalar_vec(c0v, a0_col_hi, a0_col_lo)
    left1_hi, left1_lo = dd_outer_scalar_vec(c1v, a1_col_hi, a1_col_lo)

    left_hi = left0_hi + c * left1_hi
    left_lo = left0_lo + c * left1_lo

    out_hi[1:, 0] = left_hi.reshape(-1)
    out_lo[1:, 0] = left_lo.reshape(-1)

    return renorm(out_hi, out_lo)


def dd_add_inplace(X, Y):
    # X and Y are DD tuples; returns new tuple (don’t mutate views)
    return dd_add(X, Y)

def dd_scale(x, s):
    # x is DD, s float64 scalar
    return dd_mul_exact_scalar(x, float(s))



# ---------- “upgrade only where needed” helpers ----------

def dd_cos(a2, b2, c2, ia, ib):
    """
    Compute (a2 + b2 - c2) / a / b / 2 as DD, efficiently.
    a,b,c float64 arrays/scalars.
    """
    # a = np.asarray(a, dtype=np.float64)
    # b = np.asarray(b, dtype=np.float64)
    # c = np.asarray(c, dtype=np.float64)

    ab = dd_add(a2, b2)
    abc = dd_add(ab, (-c2[0], -c2[1]))
    abca = dd_mul(abc, ia)
    abcab = dd_mul(abca, ib) # todo: skip renorm on intermediate steps?
    return dd_mul(abcab, (0.5, 0.0))


def dd_matmul(coef, Fij):
    """
    DD-accurate Y = (coef_hi+coef_lo) @ Fij

    coef: ((N, t), (N, t)) float64
    Fij: (t, b) int or float (ints <= 2^53 exactly representable)
    Returns: (Y_hi, Y_lo) both (N, b) float64
    """

    p, t = coef[0].shape
    t2, b = Fij.shape
    if t2 != t:
        raise ValueError(f"Shape mismatch: coef has t={t} but F has t={t2}")
    #
    # acc = dd_from(np.zeros((p, b), dtype=np.float64))
    #
    # for k in range(t):
    #     coef_vector_times_f_hi = np.zeros((p, b), dtype = np.float64)
    #     coef_vector_times_f_lo = np.zeros((p, b), dtype = np.float64)
    #     for f in np.unique(Fij[k, :]):   # unique values that coef_vector_1[:, k] has to be multiplied by
    #         f_mask = Fij[k, :] == f
    #         ff = dd_mul_exact_scalar((coef[0][:,k], coef[1][:,k]), f)
    #         coef_vector_times_f_hi[:, f_mask] = ff[0][:, None]
    #         coef_vector_times_f_lo[:, f_mask] = ff[1][:, None]
    #     acc = dd_add(acc, (coef_vector_times_f_hi, coef_vector_times_f_lo))

    acc = dd_from(np.zeros((p, b), np.float64))

    for k in range(t):
        row = Fij[k, :]  # (b,) ints

        fvals, inv = np.unique(row, return_inverse=True)  # inv in [0..u-1]
        u = fvals.size
        fvals_f = fvals.astype(np.float64)  # exact for small ints

        # Build grouping matrix G: (u,b) one-hot rows
        G = np.zeros((u, b), dtype=np.float64)
        G[inv, np.arange(b)] = 1.0

        # Build scaled DD vectors for each unique factor: (p,u)
        a_hi = coef[0][:, k][:, None]  # (p,1)
        a_lo = coef[1][:, k][:, None]

        # multiply a_hi by each f in DD: p = a_hi*f, err = rounding error
        prod_hi, prod_err = two_prod(a_hi, fvals_f[None, :])  # (p,u)
        prod_lo = prod_err + a_lo * fvals_f[None, :]  # (p,u)

        # scatter to columns by tiny GEMM
        term_hi = prod_hi @ G  # (p,b)
        term_lo = prod_lo @ G  # (p,b)

        acc = dd_add(acc, (term_hi, term_lo))

    return acc

_EPS64 = np.finfo(np.float64).eps  # ~2.220446049250313e-16

def digits_ok_for_addsub(a, b, op="+", min_digits=6, name=None):
    """
    Estimate decimal digits of accuracy for s = a (+/-) b in float64 due to cancellation.
    Warns if estimated digits < min_digits.
    Works for scalars or arrays (float64).
    Returns D (same shape as broadcast(a,b)).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if op == "+":
        s = a + b
    elif op == "-":
        s = a - b
    else:
        raise ValueError("op must be '+' or '-'")

    denom = np.abs(a) + np.abs(b)
    abs_s = np.abs(s)

    # kappa ≈ (|a|+|b|)/|s|
    zero = abs_s == 0.0
    kappa = np.empty_like(abs_s)
    kappa[~zero] = denom[~zero] / abs_s[~zero]
    kappa[zero] = np.where(denom[zero] == 0.0, 1.0, np.inf)

    # D ≈ -log10(kappa*eps)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        badness = kappa * _EPS64
        D = -np.log10(badness)

    D = np.where(np.isfinite(D), D, -np.inf)
    D = np.maximum(D, 0.0)

    minD = float(np.min(D))
    if minD < min_digits:
        shape = D.shape
        if shape == ():
            idx = ()
        else:
            idx = np.unravel_index(int(np.argmin(D)), shape)

        # broadcast so scalar inputs work cleanly
        aB = np.broadcast_to(a, shape)
        bB = np.broadcast_to(b, shape)
        sB = np.broadcast_to(s, shape)
        kB = np.broadcast_to(kappa, shape)

        print(
            f"[DIGITS WARNING] {name or f'a {op} b'}  "
            f"a={aB[idx]:.16e}  b={bB[idx]:.16e}  s={sB[idx]:.16e}  Digits≈{D[idx]:.2f}"
        )

    return D
