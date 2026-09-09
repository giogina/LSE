import numpy as np
from numpy.polynomial.legendre import leggauss


def build_r_grid(nrR=60, r_max=60., gamma=3.0):
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

    return np.array(r), np.array(wr) # todo: why tf is it clustering near rmax, don't need that. Same for s sampling?

def build_a_grid(Ka=48):  # Todo: only need positive half
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
    s = np.sqrt(1.0 - a*a)
    dmin = np.sqrt(1.0 - s)
    dmax = np.sqrt(1.0 + s)
    return dmin, dmax

def D_matrix(dmin, dmax, k_max, Delta=0, k_factor=None):
    """
    Returns KxK matrix G[p,q](a) = P_on_ki(k_p) * P_on_kj(k_q) * I_{k_p+k_q+Delta}(a)
    where I_m(a) = ∫ d^{m+1} dd over allowed d-range.
    """
    k = np.arange(k_max + 1, dtype=np.int64)              # (K,)
    kk = k[:, None] + k[None, :] + int(Delta)             # (K,K)  -> m
    p = kk + 2

    p3 = p[:, :, None].astype(np.float64)
    mask_p0 = (p == 0)
    if np.any(mask_p0):
        p3[mask_p0, :] = 1.0  # temporary nonzero to avoid division by 0

    G = (dmax[None, None, :] ** p3 - dmin[None, None, :] ** p3) / p3

    if np.any(mask_p0):
        G = G.copy()
        G[mask_p0, :] = 0.0

    if k_factor is not None:
        G *= k_factor(k)[None, :, None]
    return G


def A_matrix(a, a_factor, n_max, n_factor=None):
    """
    Build A[ni, nj, a] with:
        A = a_factor(a) * nj_factor(nj) * a^(2*(ni+nj))
    """

    n = np.arange(n_max + 1, dtype=np.int64)           # (N,)
    nn = n[:, None] + n[None, :]                       # (N,N)

    A = a[None, None, :] ** (2 * nn[:, :, None])       # a^(2*(ni+nj)) for all a (N,N,Na)
    A *= a_factor[None, None, :]    # multiply scalar a-dependent factor

    if n_factor is not None:    # multiply nj-polynomial factor (broadcast over ni and a)
        pj = np.asarray(n_factor(n), dtype=np.float64)    # (N,)
        A *= pj[None, :, None]
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

    R = (r ** hh) * (np.log(r) ** ll)

    if factor_hl is not None:
        pj = np.asarray(factor_hl(h, l), dtype=np.float64)  # (N,)
    else:
        pj = np.ones_like(hh, dtype = np.float64)

        R *= pj
    return R

BO = True
M = 1836.1526738  #Previously used: 1836.153
Z = 2.0  # just for the potential

# Basis set maximum powers (rAB^h * r12^k * s^n * t^m * (mu1^i*mu2^j + mu1^j*mu2^i) * exp( - alpha*s - beta*rAB - gamma*r12 )
h_max = 6
k_max = 6
n_max = 6

alpha = 2.0

nrR = 60  # todo: 60
nrA = 60
r_max = 60
# −2.903 724 377 034 119 598 311 159 245 194 404 446 696 9 05 37
label = "he-test"

if BO:
    Minv = 0.0
    M1M = 1.0
else:
    Minv = 1.0 / M
    M1M = (M+1.0)/M

r, wr = build_r_grid(nrR, r_max)
a, wa = build_a_grid(nrA)

# a-dependent quantities (odd powers of r1, r2)
q1 = np.sqrt(2.0 * (1.0 + a))  # sqrt(2+2a)
q2 = np.sqrt(2.0 * (1.0 - a))  # sqrt(2-2a)
qs = q1 + q2
qt = q1 - q2
inv_q1 = 1.0 / q1
inv_q2 = 1.0 / q2
inv_q1_q2 = inv_q1 * inv_q2
vs = inv_q1 + inv_q2
vt = inv_q1 - inv_q2

# int int int R[hi,li,hj,lj](r) * A[ni,nj](a) * D[ki,kj](d) * r^5*d dr da dd
dmin, dmax = d_bounds_from_a(a)

def kron_3d(wa, amat, dmat):
    Na = wa.size
    Nu = amat.shape[0]
    Kd = dmat.shape[0]
    # Build K in 4D form: (u,p,v,q,a)
    K4 = (wa[None,None,None,None,:] *
          amat[:, None, :, None, :] *     # (u,1,v,1,a)
          dmat[None, :, None, :, :])      # (1,p,1,q,a)
    return K4.transpose(4,0,1,2,3).reshape(Na, Nu*Kd, Nu*Kd)

dmat_S = D_matrix(dmin, dmax, k_max)
amat_S = A_matrix(a, np.ones_like(a), n_max)
admat_S = kron_3d(wa, amat_S, dmat_S)

terms = []

def build_term(
        a_dep = np.ones_like(a),
        delta_h = -2,
        delta_l = 0,
        delta_k = 0,
        factor_n = lambda n: 0*n + 1.0,  # (need the 0*n to make f(vector) stay a vector)
        factor_k = lambda k: 0*k + 1.0,
        factor_hl = lambda h, l: 1.0
):
    return {
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
    a_dep=alpha * a * vt,
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
    factor_k=lambda k: (-(0.5)*M1M-(0.5)*Minv+1.0) * k**2
))

# (M1M+Minv-1)*h*k/r^2
terms.append(build_term(
    factor_k=lambda k: (M1M+Minv-1.0) * k,
    factor_hl=lambda h, l: h
))

# (M1M+Minv-1)*l*k/(ln(r)*r^2)
terms.append(build_term(
    delta_l=-1,
    factor_k=lambda k: (M1M+Minv-1.0) * k,
    factor_hl=lambda h, l: l
))

# -(1/2)*alpha*(M1M*qs+2*Minv*vs-2*vs)*k/r
terms.append(build_term(
    delta_h=-1,
    a_dep=-(0.5)*alpha*(M1M*qs + 2*Minv*vs - 2*vs),
    factor_k=lambda k: k
))

# (2*M1M-Minv)*k/r^2
terms.append(build_term(
    factor_k=lambda k: (M1M + 1.0) * k
))

# (-4*Minv+4)*n*k/r^2
terms.append(build_term(
    factor_n=lambda n: (-4*Minv + 4.0) * n,
    factor_k=lambda k: k
))

# (-(1/2)*M1M-(1/2)*Minv)*h^2/r^2
terms.append(build_term(
    factor_hl=lambda h, l: -(0.5)*(M1M+Minv) * h**2
))

# (-M1M-Minv)*h*l/(ln(r)*r^2)
terms.append(build_term(
    delta_l=-1,
    factor_hl=lambda h, l: -(1.0 + 2.0*Minv) * h * l
))
# (M+1)/M + 1/M = 1+2/M

# (1/2)*alpha*(M1M*qs+2*Minv*vs)*h/r
terms.append(build_term(
    delta_h=-1,
    a_dep=(0.5)*alpha*(M1M*qs + 2*Minv*vs),
    factor_hl=lambda h, l: h
))

# (-2*M1M+Minv)*h/r^2
terms.append(build_term(
    factor_hl=lambda h, l: (-2*M1M + Minv) * h
))

# 4*Minv*h*n/r^2
terms.append(build_term(
    factor_n=lambda n: n,
    factor_hl=lambda h, l: 4*Minv * h
))

# (1/2)*l*alpha*(M1M*qs+2*Minv*vs)/(ln(r)*r)
terms.append(build_term(
    delta_h=-1,
    delta_l=-1,
    a_dep=(0.5)*alpha*(M1M*qs + 2*Minv*vs),
    factor_hl=lambda h, l: l
))

# 4*Minv*l*n/(r^2*ln(r))
terms.append(build_term(
    delta_l=-1,
    factor_n=lambda n: n,
    factor_hl=lambda h, l: 4*Minv * l
))

# (-2*M1M+Minv)*l/(r^2*ln(r))
terms.append(build_term(
    delta_l=-1,
    factor_hl=lambda h, l: (-2*M1M + Minv) * l
))

# -alpha^2*(M1M*q1*q2+2*Minv)/(q1*q2)     (no r dependence)
terms.append(build_term(
    delta_h=0,
    a_dep=-alpha**2 * (M1M*q1*q2 + 2*Minv) / (q1*q2)
))

# (-2*qs*alpha*M1M)*n/r
terms.append(build_term(
    delta_h=-1,
    a_dep=(-2.0*qs*alpha*M1M),
    factor_n=lambda n: n
))

# (2*qt*alpha*(M1M-Minv)/a)*n/r
terms.append(build_term(
    delta_h=-1,
    a_dep=(2*qt*alpha*(M1M-Minv)/a),
    factor_n=lambda n: n
))
print(M1M-Minv)
# 2*alpha*vs*M1M/r
terms.append(build_term(
    delta_h=-1,
    a_dep=2*alpha*vs*M1M*np.ones_like(a)
))

# ((-8*M1M+8*Minv)/a^2)*n^2/r^2
terms.append(build_term(
    a_dep=((-8*M1M + 8*Minv)/a**2),
    factor_n=lambda n: n**2
))

# (8*M1M-8*Minv)*n^2/r^2
terms.append(build_term(
    a_dep=(8*M1M - 8*Minv)*np.ones_like(a),
    factor_n=lambda n: n**2
))

# (8*M1M-4*Minv)*n/r^2
terms.append(build_term(
    a_dep=(8*M1M - 4*Minv)*np.ones_like(a),
    factor_n=lambda n: n
))

# ((4*M1M-4*Minv)/a^2)*n/r^2
terms.append(build_term(
    a_dep=((4*M1M - 4*Minv)/a**2),
    factor_n=lambda n: n
))

# (-(1/2)*M1M-(1/2)*Minv)*l*(l-1)/(r^2*ln(r)^2)
terms.append(build_term(
    delta_l=-2,
    factor_hl=lambda h, l: (-(0.5)*M1M-(0.5)*Minv) * l * (l - 1)
))

# (1/2)*d^2*k^2*Minv/r^2
terms.append(build_term(
    delta_k=2,
    factor_k=lambda k: 0.5*Minv * k**2
))

# -d^2*h*k*Minv/r^2
terms.append(build_term(
    delta_k=2,
    factor_k=lambda k: -Minv * k,
    factor_hl=lambda h, l: h
))

# -d^2*l*k*Minv/(r^2*ln(r))
terms.append(build_term(
    delta_k=2,
    delta_l=-1,
    factor_k=lambda k: -Minv * k,
    factor_hl=lambda h, l: l
))

# alpha*d^2*k*Minv*vs/r
terms.append(build_term(
    delta_k=2,
    delta_h=-1,
    a_dep=alpha * Minv * vs,
    factor_k=lambda k: k
))

# 4*d^2*n*k*Minv/r^2
terms.append(build_term(
    delta_k=2,
    factor_n=lambda n: 4*Minv * n,
    factor_k=lambda k: k
))

# d^2*k*Minv/r^2
terms.append(build_term(
    delta_k=2,
    factor_k=lambda k: Minv * k
))

# (1/2)*Minv*d^2*h^2/r^2
terms.append(build_term(
    delta_k=2,
    factor_hl=lambda h, l: 0.5*Minv * h**2
))

# Minv*d^2*h*l/(r^2*ln(r))
terms.append(build_term(
    delta_k=2,
    delta_l=-1,
    factor_hl=lambda h, l: Minv * h * l
))

# -alpha*vs*d^2*Minv*h/r
terms.append(build_term(
    delta_k=2,
    delta_h=-1,
    a_dep=-alpha * Minv * vs,
    factor_hl=lambda h, l: h
))

# -4*Minv*h*n*d^2/r^2
terms.append(build_term(
    delta_k=2,
    factor_n=lambda n: n,
    factor_hl=lambda h, l: -4*Minv * h
))

# -Minv*d^2*h/r^2
terms.append(build_term(
    delta_k=2,
    factor_hl=lambda h, l: -Minv * h
))

# (1/2)*Minv*d^2*l^2/(r^2*ln(r)^2)
terms.append(build_term(
    delta_k=2,
    delta_l=-2,
    factor_hl=lambda h, l: 0.5*Minv * l**2
))

# -alpha*vs*d^2*Minv*l/(r*ln(r))
terms.append(build_term(
    delta_k=2,
    delta_h=-1,
    delta_l=-1,
    a_dep=-alpha * Minv * vs,
    factor_hl=lambda h, l: l
))

# -4*Minv*l*n*d^2/(r^2*ln(r))
terms.append(build_term(
    delta_k=2,
    delta_l=-1,
    factor_n=lambda n: n,
    factor_hl=lambda h, l: -4*Minv * l
))

# -Minv*d^2*l/(r^2*ln(r))
terms.append(build_term(
    delta_k=2,
    delta_l=-1,
    factor_hl=lambda h, l: -Minv * l
))

# -(1/2)*Minv*d^2*l/(r^2*ln(r)^2)
terms.append(build_term(
    delta_k=2,
    delta_l=-2,
    factor_hl=lambda h, l: -(0.5)*Minv * l
))

# 2*alpha^2*Minv*d^2/(q1*q2)   (no r dependence)
terms.append(build_term(
    delta_k=2,
    delta_h=0,
    a_dep=2*alpha**2 * Minv / (q1*q2)
))

# 2*alpha*n*Minv*d^2*qt/(r*a)
terms.append(build_term(
    delta_k=2,
    delta_h=-1,
    a_dep=2*alpha*Minv * qt / a,
    factor_n=lambda n: n
))

# 8*Minv*(-1+a)*(1+a)*d^2*n^2/(a^2*r^2)
# note: (-1+a)*(1+a) = a^2 - 1
terms.append(build_term(
    delta_k=2,
    a_dep=8*Minv * ((a**2 - 1.0) / (a**2)),
    factor_n=lambda n: n**2
))

# 4*Minv*(a^2+1)*d^2*n/(a^2*r^2)
terms.append(build_term(
    delta_k=2,
    a_dep=4*Minv * ((a**2 + 1.0) / (a**2)),
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
    a_dep = -2*Z*vs
))

for term in terms:
    term["admat"] = kron_3d(wa,
         A_matrix(a, term["a_dep"], n_max, n_factor=term["factor_n"]),   # (absorb constant factors into the n_factor)
         D_matrix(dmin, dmax, k_max, Delta=term["delta_k"], k_factor=term["factor_k"]))

nrhl = 0
for hh in np.arange(h_max+1):
    for ll in np.arange(np.floor(hh/2)+1):
        nrhl += 1

matsize = nrhl * (n_max+1) * (k_max+1)
S = np.zeros((matsize, matsize), dtype=np.float64)
H = np.zeros((matsize, matsize), dtype=np.float64)

for ri in np.arange(len(r)):
    print(ri, r[ri])
    exps = np.exp(-alpha*r[ri] * qs)
    ad_S = np.tensordot(exps, admat_S, axes=(0,0))

    rmat_S = R_matrix(r[ri], h_max)
    S += wr[ri] * np.kron(rmat_S, ad_S)

    for term in terms:
        ad_H = np.tensordot(exps, term["admat"], axes=(0,0))
        rmat_H = R_matrix(r[ri], h_max, term["delta_h"], term["delta_l"], term["factor_hl"])
        H += wr[ri] * np.kron(rmat_H, ad_H)

from solver import solve_HS
E, C, cond = solve_HS(H, S, 1e-16)










