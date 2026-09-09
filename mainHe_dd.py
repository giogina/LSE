import time

from numpy.polynomial.legendre import leggauss
from double_precision import *
from double_precision import dd_inverse
from ratplot import ratplot, save_psi_ad_bundle_pickle


# todo: currently testing gamma=4 - is it better?
# todo: Only 24% of functions are being kept - remove most ln-functions (except for known ones); what else?
# todo: d=0 is just a single point (also a=0). Does this have to propagate upwards into an a-dependent function?
#  Can I take the a^0 coefficients as "optimization seeds" and propagate from there; and then only solve for those seed coeffs?
# todo: *Is* it almost a product? Of what?
# Todo: Psi(r,d) at a=0 (fig 3) has straight lines from certain angles. What are they?

BO = True
M = 1836.1526738  #Previously used: 1836.153
Z = 2.0  # just for the potential

# Basis set maximum powers (rAB^h * r12^k * s^n * t^m * (mu1^i*mu2^j + mu1^j*mu2^i) * exp( - alpha*s - beta*rAB - gamma*r12 )
p_max = 10
h_max = 3
k_max = 2
n_max = 2

alpha = Z  # Necessary for reduced kron version (psi0+psi1 are hard-coded)

nrR = 100  # At least 60
nrA = 100
r_max = 60
# −2.903 724 377 034 119 598 311 159 245 194 404 446 696 9 05 37

def build_r_grid(nrR=60, r_max=60., gamma=4.0):
    """
    Build a clustered Gauss–Legendre quadrature grid for r ∈ [0, r_max],
    with clustering towards r=0 via r = r_max * u**gamma.

    Parameters
    ----------
    nrR : int
        Number of Gauss–Legendre nodes (recommend ~48–60).
    r_max : float
        Cutoff for r. (recommended: 60/alpha).
    gamma : float
        Clustering strength (>=1). (recommended: gamma=3 clusters strongly near 0; good for functions involving ln(r).)

    Returns
    -------
    r : ndarray
        Quadrature nodes in [0, r_max].
    wr : ndarray
        Corresponding quadrature weights for dr.
    """

    x, w = leggauss(nrR)          # nodes/weights on [-1,1]
    u = 0.5 * (x + 1.0)          # map to [0,1]
    wu = 0.5 * w                 # du weights

    # cluster near 0
    r = r_max * (u ** gamma)
    dr_du = r_max * gamma * (u ** (gamma - 1.0))
    wr = 2.0 * np.pi**2 * r**5 * wu * dr_du

    return np.array(r), np.array(wr)

def build_a_grid(Ka=48):
    """
    Gauss–Legendre quadrature on a ∈ [-1,1].
    """
    a, wa = leggauss(Ka)
    a = 0.5 * (a + 1.0)
    return np.array(a), np.array(wa)

def frange(start, stop, step):
    x = start
    if step > 0:
        while x <= stop + 1e-12:
            yield x
            x += step
    else:
        while x >= stop - 1e-12:
            yield x
            x += step

def d_bounds_from_a(a):
    one = (np.float64(1.0), np.float64(0.0))

    s = dd_sqrt(dd_subs(one, dd_square(a)))      # s = sqrt(1 - a^2)
    dmin = dd_sqrt(dd_subs(one, s))              # sqrt(1 - s)
    dmax = dd_sqrt(dd_add(one, s))               # sqrt(1 + s)
    return dmin, dmax

def D_matrix(dmin, dmax, k_max, Delta=0, k_factor=None):

    k = np.arange(k_max + 1, dtype=np.int64)
    p = (k[:, None] + k[None, :] + int(Delta) + 2)  # (K,K) integer
    K = k_max + 1
    Na = dmin[0].shape[0]

    # k_factor: enforce integer-valued on integer k
    fac = None
    if k_factor is not None:
        fac_raw = np.asarray(k_factor(k), dtype=np.float64)
        if (not np.all(np.isfinite(fac_raw))) or (not np.all(fac_raw == np.round(fac_raw))):
            raise ValueError(f"k_factor produced non-integers: {fac_raw}")
        fac = fac_raw.astype(np.int64).astype(np.float64)  # exact if small

    # power tables (DD) for exponents 0..pmax
    pmax = int(np.max(p))
    pow_min = dd_pow_table(dmin, pmax)
    pow_max = dd_pow_table(dmax, pmax)

    idx = p.astype(np.intp)  # (K,K)
    dmin_p = (pow_min[0][idx, :], pow_min[1][idx, :])  # (K,K,Na)
    dmax_p = (pow_max[0][idx, :], pow_max[1][idx, :])

    num = dd_subs(dmax_p, dmin_p)  # DD

    # DD divide by integer p using a tiny reciprocal table (true DD everywhere)
    G = dd_from(np.zeros((K, K, Na), np.float64))
    mask_p0 = (p == 0)
    p_vals = np.unique(p.astype(np.int64))
    p_vals = p_vals[p_vals != 0]

    invp = {int(pp): dd_inverse(dd_from(np.float64(pp))) for pp in p_vals}

    for pp in p_vals:
        m = (p == pp)
        inv = invp[int(pp)]
        out = dd_mul((num[0][m, :], num[1][m, :]), inv)
        G[0][m, :] = out[0]
        G[1][m, :] = out[1]

    if np.any(mask_p0):
        G0 = G[0].copy(); G1 = G[1].copy()
        G0[mask_p0, :] = 0.0; G1[mask_p0, :] = 0.0
        G = (G0, G1)

    # scale columns (j) by exact integer polynomial factor
    if fac is not None:
        G = dd_mul(G, dd_from(fac[None, :, None]))

    return G

def A_matrix(a, a_factor, n_max, n_factor=None):
    """
    Build A[ni, nj, a] with:
        A = a_factor(a) * nj_factor(nj) * a^(2*(ni+nj))
    """

    n = np.arange(n_max + 1, dtype=np.int64)           # (N,)
    nn = n[:, None] + n[None, :]                       # (N,N)
    p = (2 * nn).astype(np.int64)                      # exponents (N,N)

    # power table for a^(0..pmax) in DD
    pmax = int(np.max(p))
    pow_a = dd_pow_table(a, pmax)                   # (pmax+1,Na)

    idx = p.astype(np.intp)
    A = (pow_a[0][idx, :], pow_a[1][idx, :])           # (N,N,Na) DD

    # multiply by a-dependent factor in DD
    A = dd_mul(A, (a_factor[0][None, None, :], a_factor[1][None, None, :]))

    # multiply columns (nj) by n_factor if given
    if n_factor is not None:
        pj = np.asarray(n_factor(n), dtype=np.float64)  # (N,)
        A = dd_mul(A, dd_from(pj[None, :, None]))

    return A

def R_matrix(r, h_max, delta_h = 0, delta_l = 0, factor_hl=None):

    h_list = []
    l_list = []
    for h in np.arange(h_max + 1):
        for l in np.arange(h // 2 + 1):
            h_list.append(h)
            l_list.append(l)

    h = np.array(h_list, dtype=np.int64)
    l = np.array(l_list, dtype=np.int64)

    # exponent matrices
    hh = h[:, None] + h[None, :] + int(delta_h)  # (Nb,Nb)
    ll = l[:, None] + l[None, :] + int(delta_l)  # (Nb,Nb)
    Nb = len(h_list)

    if factor_hl is not None:
        pj = np.asarray(factor_hl(h, l), dtype=np.float64)  # (Nb,)
    else:
        pj = np.ones_like(h, dtype=np.float64)

    r_dd = dd_from(float(r))
    log_dd = dd_from(np.log(float(r))) # in tests, perfectly accurate w.r.t. dd version of the function.

    R_hi = np.zeros((Nb, Nb), dtype=np.float64)
    R_lo = np.zeros((Nb, Nb), dtype=np.float64)

    for i in range(Nb):
        for j in range(Nb):
            rpow = dd_pow_int(r_dd, int(hh[i, j]))  # r^hh
            lpow = dd_pow_int(log_dd, int(ll[i, j]))  # (log r)^ll
            val = dd_mul(rpow, lpow)

            if pj[j] != 1.0:   # multiply by pj[j]
                val = dd_mul_exact_scalar(val, pj[j])

            R_hi[i, j] = val[0]
            R_lo[i, j] = val[1]

    return (R_hi, R_lo)


def kron_3d(wa, amat, dmat):
    Na = wa[0].size

    Nu = amat[0].shape[0]
    Kd = dmat[0].shape[0]

    # wa as DD, broadcast to (1,1,1,1,a)
    Wa = (wa[0][None, None, None, None, :],
          wa[1][None, None, None, None, :])

    # broadcast amat to (u,1,v,1,a)
    A = (amat[0][:, None, :, None, :],
         amat[1][:, None, :, None, :])

    # broadcast dmat to (1,p,1,q,a)
    D = (dmat[0][None, :, None, :, :],
         dmat[1][None, :, None, :, :])

    K4 = dd_mul(dd_mul(Wa, A), D)  # (u,p,v,q,a) as DD

    # (u,p,v,q,a) -> (a,u,p,v,q) -> (a, Nu*Kd, Nu*Kd)
    K_hi = K4[0].transpose(4, 0, 1, 2, 3).reshape(Na, Nu * Kd, Nu * Kd)
    K_lo = K4[1].transpose(4, 0, 1, 2, 3).reshape(Na, Nu * Kd, Nu * Kd)

    return (K_hi, K_lo)


if BO:
    Minv = dd_from(0.0)
    M1M = dd_from(1.0)
else:
    Minv = dd_inverse(dd_from(M))
    M1M = dd_mul(dd_from(M+1.0), Minv)

r, wr = build_r_grid(nrR, r_max)
a, wa = build_a_grid(nrA)

zero = dd_from(0.0)
one = dd_from(1.0)
two = dd_from(2.0)
a = dd_from(a)

# a-dependent quantities (odd powers of r1, r2)
q1 = dd_sqrt(dd_mul_exact_scalar(dd_add(one, a), 2.0))  # sqrt(2+2a)
q2 = dd_sqrt(dd_mul_exact_scalar(dd_subs(one, a), 2.0))  # sqrt(2-2a)
qs = dd_add(q1, q2)
qt = dd_subs(q1, q2)
inv_q1 = dd_inverse(q1)
inv_q2 = dd_inverse(q2)
inv_q1_q2 = dd_mul(inv_q1, inv_q2)
vs = dd_add(inv_q1, inv_q2)
vt = dd_subs(inv_q1, inv_q2)

# int int int R[hi,li,hj,lj](r) * A[ni,nj](a) * D[ki,kj](d) * r^5*d dr da dd
dmin, dmax = d_bounds_from_a(a)

dmat_S = D_matrix(dmin, dmax, k_max)
amat_S = A_matrix(a, dd_from(np.ones_like(a[0])), n_max)
admat_S = kron_3d(dd_from(wa), amat_S, dmat_S)

terms = []

def build_term(
        const = one,
        a_dep = dd_from(np.ones_like(a[0])),
        delta_h = -2,
        delta_l = 0,
        delta_k = 0,
        factor_n = lambda n: 0*n + 1,  # (need the 0*n to make f(vector) stay a vector)
        factor_k = lambda k: 0*k + 1,
        factor_hl = lambda h, l: 0*h + 1
):
    return {
        "const": const,
        "delta_h": delta_h,
        "delta_l": delta_l,
        "delta_k": delta_k,
        "a_dep": a_dep,
        "factor_n": factor_n,
        "factor_k": factor_k,
        "factor_hl": factor_hl
    }


# alpha*a*vt/r*k/d^2
terms.append(build_term(
    delta_h=-1,
    delta_k=-2,
    a_dep=dd_mul_exact_scalar(dd_mul(a, vt), alpha),
    factor_k= lambda k: k
))

# -(4*n+1)/r^2*k/d^2
terms.append(build_term(
    delta_k=-2,
    factor_k= lambda k: k,
    factor_n= lambda n: -(4*n+1)
))

# -k^2/r^2/d^2
terms.append(build_term(
    delta_k=-2,
    factor_k= lambda k: -k**2
))


# (-(1/2)*M1M-(1/2)*Minv+1)*k^2/r^2
terms.append(build_term(
    const = dd_subs(dd_from(0.5), Minv),
    factor_k=lambda k: k**2
))

# (M1M+Minv-1)*h*k/r^2
terms.append(build_term(
    const = dd_mul_exact_scalar(Minv, 2.0),
    factor_k=lambda k:  k,
    factor_hl=lambda h, l: h
))

# (M1M+Minv-1)*l*k/(ln(r)*r^2)
terms.append(build_term(
    const = dd_mul_exact_scalar(Minv, 2.0),
    delta_l=-1,
    factor_k=lambda k: k,
    factor_hl=lambda h, l: l
))

# -(1/2)*alpha*(M1M*qs+2*Minv*vs-2*vs)*k/r
terms.append(build_term(
    delta_h=-1,
    a_dep=dd_mul_exact_scalar(dd_add(dd_mul(dd_mul_exact_scalar(M1M, 0.5), qs), dd_mul(dd_subs(Minv, one),vs)), -alpha),
    factor_k=lambda k: k
))

# (2*M1M-Minv)*k/r^2   with M1M=Minv+1 => (Minv+2)*k/r^2
terms.append(build_term(
    const = dd_add(Minv, dd_from(2.0)),
    factor_k=lambda k:  k
))

# (-4*Minv+4)*n*k/r^2
terms.append(build_term(
    const = dd_mul_exact_scalar(dd_subs(one, Minv), 4.0),  # 4*(1-Minv)
    factor_n=lambda n:  n,
    factor_k=lambda k: k
))

# (-(1/2)*M1M-(1/2)*Minv)*h^2/r^2   with M1M=Minv+1 => -(Minv+1/2)*h^2/r^2
terms.append(build_term(
    const = dd_mul_exact_scalar(dd_add(Minv, dd_from(0.5)), -1.0),
    factor_hl=lambda h, l: h**2
))

# (-M1M-Minv)*h*l/(ln(r)*r^2)   with M1M=Minv+1 => -(2*Minv+1)*h*l/(ln(r)*r^2)
terms.append(build_term(
    const = dd_mul_exact_scalar(dd_add(dd_mul_exact_scalar(Minv, 2.0), one), -1.0),
    delta_l = -1,
    factor_hl=lambda h, l: h * l
))
# (M+1)/M + 1/M = 1+2/M

# (1/2)*alpha*(M1M*qs+2*Minv*vs)*h/r   with M1M=Minv+1
terms.append(build_term(
    delta_h=-1,
    a_dep=dd_mul_exact_scalar(
        dd_mul_exact_scalar(
            dd_add(dd_mul(M1M, qs), dd_mul_exact_scalar(dd_mul(Minv, vs), 2.0)),
            alpha
        ),
        0.5
    ),
    factor_hl=lambda h, l: h
))

# (-2*M1M+Minv)*h/r^2   with M1M=Minv+1 => (-Minv-2)*h/r^2
terms.append(build_term(
    const = dd_mul_exact_scalar(dd_add(Minv, dd_from(2.0)), -1.0),
    factor_hl=lambda h, l:  h
))

# 4*Minv*h*n/r^2
terms.append(build_term(
    const = dd_mul_exact_scalar(Minv, 4.0),
    factor_n=lambda n: n,
    factor_hl=lambda h, l: h
))

# (1/2)*l*alpha*(M1M*qs+2*Minv*vs)/(ln(r)*r)   with M1M=Minv+1
terms.append(build_term(
    delta_h=-1,
    delta_l=-1,
    a_dep=dd_mul_exact_scalar(
        dd_mul_exact_scalar(
            dd_add(dd_mul(M1M, qs), dd_mul_exact_scalar(dd_mul(Minv, vs), 2.0)),
            alpha
        ),
        0.5
    ),
    factor_hl=lambda h, l: l
))

# 4*Minv*l*n/(r^2*ln(r))
terms.append(build_term(
    const = dd_mul_exact_scalar(Minv, 4.0),
    delta_l=-1,
    factor_n=lambda n: n,
    factor_hl=lambda h, l: l
))

# (-2*M1M+Minv)*l/(r^2*ln(r))   with M1M=Minv+1 => (-Minv-2)*l/(r^2*ln(r))
terms.append(build_term(
    const = dd_mul_exact_scalar(dd_add(Minv, dd_from(2.0)), -1.0),
    delta_l=-1,
    factor_hl=lambda h, l: l
))

# -alpha^2*(M1M*q1*q2+2*Minv)/(q1*q2)  (no r dependence)
# => -alpha^2*( M1M + 2*Minv/(q1*q2) )
terms.append(build_term(
    delta_h=0,
    a_dep=dd_mul_exact_scalar(
        dd_add(
            M1M,
            dd_mul_exact_scalar(dd_mul(Minv, inv_q1_q2), 2.0)
        ),
        -alpha**2
    )
))

# (-2*qs*alpha*M1M)*n/r   with M1M=Minv+1
terms.append(build_term(
    delta_h=-1,
    a_dep=dd_mul_exact_scalar(dd_mul(qs, M1M), -2.0*alpha),
    factor_n=lambda n: n
))

# (2*qt*alpha*(M1M-Minv)/a)*n/r  with M1M-Minv=1 => (2*qt*alpha/a)*n/r
terms.append(build_term(
    delta_h=-1,
    a_dep=dd_mul_exact_scalar(dd_mul(qt, dd_inverse(a)), 2.0*alpha),
    factor_n=lambda n: n
))

# 2*alpha*vs*M1M/r   with M1M=Minv+1
terms.append(build_term(
    delta_h=-1,
    a_dep=dd_mul_exact_scalar(dd_mul(vs, M1M), 2.0*alpha)
))

# ((-8*M1M+8*Minv)/a^2)*n^2/r^2   with M1M=Minv+1 => (-8/a^2)*n^2/r^2
terms.append(build_term(
    a_dep=dd_mul_exact_scalar(dd_inverse(dd_mul(a, a)), -8.0),
    factor_n=lambda n: n**2
))

# (8*M1M-8*Minv)*n^2/r^2   with M1M=Minv+1 => (8)*n^2/r^2
terms.append(build_term(
    const = dd_from(8.0),
    factor_n=lambda n: n**2
))

# (8*M1M-4*Minv)*n/r^2  with M1M=Minv+1 => (4*Minv+8)*n/r^2
terms.append(build_term(
    const = dd_add(dd_mul_exact_scalar(Minv, 4.0), dd_from(8.0)),
    factor_n=lambda n: n
))

# ((4*M1M-4*Minv)/a^2)*n/r^2  with M1M=Minv+1 => (4/a^2)*n/r^2
terms.append(build_term(
    a_dep=dd_mul_exact_scalar(dd_inverse(dd_mul(a, a)), 4.0),
    factor_n=lambda n: n
))

# (-(1/2)*M1M-(1/2)*Minv)*l*(l-1)/(r^2*ln(r)^2)  with M1M=Minv+1 => -(Minv+1/2)
terms.append(build_term(
    const = dd_mul_exact_scalar(dd_add(Minv, dd_from(0.5)), -1.0),
    delta_l=-2,
    factor_hl=lambda h, l: l * (l - 1)
))

# (1/2)*d^2*k^2*Minv/r^2
terms.append(build_term(
    const = dd_mul_exact_scalar(Minv, 0.5),
    delta_k=2,
    factor_k=lambda k: k**2
))

# -d^2*h*k*Minv/r^2
terms.append(build_term(
    const = dd_mul_exact_scalar(Minv, -1.0),
    delta_k=2,
    factor_k=lambda k: k,
    factor_hl=lambda h, l: h
))

# -d^2*l*k*Minv/(r^2*ln(r))
terms.append(build_term(
    const = dd_mul_exact_scalar(Minv, -1.0),
    delta_k=2,
    delta_l=-1,
    factor_k=lambda k: k,
    factor_hl=lambda h, l: l
))

# alpha*d^2*k*Minv*vs/r
terms.append(build_term(
    delta_k=2,
    delta_h=-1,
    a_dep=dd_mul_exact_scalar(dd_mul(Minv, vs), alpha),
    factor_k=lambda k: k
))

# 4*d^2*n*k*Minv/r^2
terms.append(build_term(
    delta_k=2,
    const = dd_mul_exact_scalar(Minv, 4.0),
    factor_n=lambda n: n,
    factor_k=lambda k: k
))

# d^2*k*Minv/r^2
terms.append(build_term(
    delta_k=2,
    const = Minv,
    factor_k=lambda k: k
))

# (1/2)*Minv*d^2*h^2/r^2
terms.append(build_term(
    delta_k=2,
    const = dd_mul_exact_scalar(Minv, 0.5),
    factor_hl=lambda h, l: h**2
))

# Minv*d^2*h*l/(r^2*ln(r))
terms.append(build_term(
    delta_k=2,
    delta_l=-1,
    const = Minv,
    factor_hl=lambda h, l: h * l
))

# -alpha*vs*d^2*Minv*h/r
terms.append(build_term(
    delta_k=2,
    delta_h=-1,
    a_dep=dd_mul_exact_scalar(dd_mul(Minv, vs), -alpha),
    factor_hl=lambda h, l: h
))

# -4*Minv*h*n*d^2/r^2
terms.append(build_term(
    delta_k=2,
    const = dd_mul_exact_scalar(Minv, -4.0),
    factor_n=lambda n: n,
    factor_hl=lambda h, l: h
))

# -Minv*d^2*h/r^2
terms.append(build_term(
    delta_k=2,
    const = dd_mul_exact_scalar(Minv, -1.0),
    factor_hl=lambda h, l: h
))

# (1/2)*Minv*d^2*l^2/(r^2*ln(r)^2)
terms.append(build_term(
    delta_k=2,
    delta_l=-2,
    const = dd_mul_exact_scalar(Minv, 0.5),
    factor_hl=lambda h, l: l**2
))

# -alpha*vs*d^2*Minv*l/(r*ln(r))
terms.append(build_term(
    delta_k=2,
    delta_h=-1,
    delta_l=-1,
    a_dep=dd_mul_exact_scalar(dd_mul(Minv, vs), -alpha),
    factor_hl=lambda h, l: l
))

# -4*Minv*l*n*d^2/(r^2*ln(r))
terms.append(build_term(
    delta_k=2,
    delta_l=-1,
    const = dd_mul_exact_scalar(Minv, -4.0),
    factor_n=lambda n: n,
    factor_hl=lambda h, l: l
))

# -Minv*d^2*l/(r^2*ln(r))
terms.append(build_term(
    delta_k=2,
    delta_l=-1,
    const = dd_mul_exact_scalar(Minv, -1.0),
    factor_hl=lambda h, l: l
))

# -(1/2)*Minv*d^2*l/(r^2*ln(r)^2)
terms.append(build_term(
    delta_k=2,
    delta_l=-2,
    const = dd_mul_exact_scalar(Minv, -0.5),
    factor_hl=lambda h, l: l
))

# 2*alpha^2*Minv*d^2/(q1*q2)   (no r dependence)
terms.append(build_term(
    delta_k=2,
    delta_h=0,
    a_dep=dd_mul_exact_scalar(dd_mul(Minv, inv_q1_q2), 2.0*alpha**2)
))

# 2*alpha*n*Minv*d^2*qt/(r*a)
terms.append(build_term(
    delta_k=2,
    delta_h=-1,
    a_dep=dd_mul_exact_scalar(dd_mul(dd_mul(Minv, qt), dd_inverse(a)), 2.0*alpha),
    factor_n=lambda n: n
))

# 8*Minv*(-1+a)*(1+a)*d^2*n^2/(a^2*r^2)
# note: (-1+a)*(1+a) = a^2 - 1  ->  8*Minv*(a^2-1)/a^2
terms.append(build_term(
    delta_k=2,
    a_dep=dd_mul_exact_scalar(
        dd_mul(
            dd_mul(Minv, dd_subs(dd_mul(a, a), one)),
            dd_inverse(dd_mul(a, a))
        ),
        8.0
    ),
    factor_n=lambda n: n**2
))

# 4*Minv*(a^2+1)*d^2*n/(a^2*r^2)  -> 4*Minv*(1 + 1/a^2)
terms.append(build_term(
    delta_k=2,
    a_dep=dd_mul_exact_scalar(
        dd_mul(
            Minv,
            dd_add(one, dd_inverse(dd_mul(a, a)))
        ),
        4.0
    ),
    factor_n=lambda n: n
))

# Potential!
# 1/r/d
terms.append(build_term(
    delta_k=-1,
    delta_h=-1
))

# -2*Z*vs/r
terms.append(build_term(
    delta_h=-1,
    a_dep = dd_mul_exact_scalar(vs, -2*Z)
))

for term in terms:
    term["admat"] = kron_3d(dd_mul_exact_scalar(term["const"], wa),
         A_matrix(a, term["a_dep"], n_max, n_factor=term["factor_n"]),   # (absorb constant factors into the n_factor)
         D_matrix(dmin, dmax, k_max, Delta=term["delta_k"], k_factor=term["factor_k"]))

nrhl = 0
h_list = []
for hh in np.arange(h_max+1):
    for ll in np.arange(np.floor(hh/2)+1):
        nrhl += 1
        h_list.append(hh)
h_vals = np.array(h_list, dtype=np.int64)

matsize = (nrhl-2) * (n_max+1) * (k_max+1) + 1
# matsize = (nrhl) * (n_max+1) * (k_max+1)
S = dd_from(np.zeros((matsize, matsize), dtype=np.float64))
H = dd_from(np.zeros((matsize, matsize), dtype=np.float64))

start = time.time()
for ri in np.arange(len(r)):
    print(ri, r[ri], f"{time.time() - start:.2f}s")
    exps = dd_exp(dd_mul_exact_scalar(qs, -alpha*r[ri]))
    ad_S = dd_tensordot_a0(exps, admat_S)

    rmat_S = R_matrix(r[ri], h_max)
    K = dd_kron_reduced(rmat_S, ad_S)
    # K = dd_kron(rmat_S, ad_S)
    S = dd_add(S, dd_scale(K, wr[ri]))

    for term in terms:
        ad_H = dd_tensordot_a0(exps, term["admat"])
        rmat_H = R_matrix(r[ri], h_max, term["delta_h"], term["delta_l"], term["factor_hl"])
        K = dd_kron_reduced(rmat_H, ad_H)
        # K = dd_kron(rmat_H, ad_H)
        H = dd_add(H, dd_mul_exact_scalar(K, wr[ri]))

from solver import solve_HS
E, C, cond = solve_HS(H[0], S[0], 1e-17)

idx = np.argmin(np.real(E))
E0 = E[idx]
C0 = C[:, idx]
C0 = C0/C0[0]

C_full = np.zeros(nrhl * (n_max+1) * (k_max+1), dtype=np.float64) # Pad solution vector to full size (with all h=0, 1 coeffs)
C_full[0] = 1.0 # Psi0 = 1
C_full[(n_max+1) * (k_max+1) + 1] = 0.5  # Psi1 = d/2*r
C_full[2 * (n_max+1) * (k_max+1):] = C0[1:]

save_psi_ad_bundle_pickle(
    "he_solution",
    C=C_full,
    energy=E0,
    n_max=n_max,
    k_max=k_max,
    h_max=h_max,
    alpha=alpha,
    label=f"he_p{p_max}_r{nrR}_a{nrA}_h{h_max}_k{k_max}_n{n_max}_gamma4",
    extra={
        "notes": "",
    },
)










