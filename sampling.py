import time
import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln
from scipy.linalg import eig

from plot import mask_closest_phi, plot_phi_and_local_energy_mu2, plot_mu2_with_alpha_beta
from util import diag_rescale_generalized


# from main import frange
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

def power_table(x, p_max, p_min=0):
    x = np.asarray(x, dtype=np.float64)

    n_pos = p_max + 1              # includes 0
    n_neg = -p_min if p_min < 0 else 0
    n_cols = n_pos + n_neg

    out = np.empty((x.shape[0], n_cols) if x.ndim else (n_cols,), dtype=np.float64)
    out[..., 0] = 1.0  # s^0

    for e in range(1, p_max + 1):
        out[..., e] = out[..., e - 1] * x  # s^1 ... s^p_max

    if p_min < 0:
        col = p_max + 1
        out[..., col] = x ** p_min
        for e in range(p_min + 1, 0):
            out[..., col + 1] = out[..., col] * x   # s^p_min ... s^-1
            col += 1

    return out

def potential_ri(rAB, rA1, rB1, rA2, rB2, r12):
    return 1/rAB + 1/r12 - 1/rA1 - 1/rB1 - 1/rA2 - 1/rB2

def build_s_shells(rAB, Ks, s_max, gamma=3.0):
    s_min = 2.0 * rAB

    x, w = leggauss(Ks)         # [-1,1]
    u = 0.5*(x + 1.0)           # [0,1]
    wu = 0.5*w

    s = s_min + (s_max - s_min) * (u**gamma)
    ds_du = (s_max - s_min) * gamma * (u**(gamma - 1.0))
    ws = wu * ds_du

    return s, ws

def split_s(s, rAB, Ku):
    """
    Deterministic split of total s into s1,s2.
    s1 ∈ [rAB, s-rAB], s2 = s - s1.
    rAB is the full internuclear distance R.
    """
    x, w = leggauss(Ku)       # nodes in [-1,1]
    u = 0.5 * (x + 1.0)       # map to [0,1]
    wu = 0.5 * w              # du weights on [0,1]

    # Map u -> s1 in [rAB, s-rAB]
    s1 = rAB + u * (s - 2.0*rAB)
    s2 = s - s1
    w_split = (s - 2.0*rAB) * wu

    return s1, s2, w_split


def shell_area_weight(s1, s2, rAB):
    shell_area_1 = 4*np.pi*((s1 / rAB)**2 - 1/3)
    shell_area_2 = 4*np.pi*((s2 / rAB)**2 - 1/3)
    return shell_area_1 * shell_area_2

def sample_s_shell(rAB, s, nMu=12, Nphi=16, octant=False):
    """
    Deterministic quadrature on the prolate spheroidal shell rA+rB = s.
    Returns (x,y,z, w) arrays of length nMu*Nphi.

    R = full internuclear distance (same R used in mu=s/R)
    """
    mu = s / rAB

    nu0, w0 = leggauss(nMu)
    a, b = (0, 1) if octant else (-1, 1)
    nu = 0.5*(b-a)*nu0 + 0.5*(a+b)
    wnu = 0.5*(b-a)*w0 * (2 if octant else 1)  # (Immediately undo half-weighting, since octant will be mirrored back)

    # phi trapezoid grid on [0,2π)
    pa, pb = (0, np.pi/2) if octant else (0, 2*np.pi)
    phi = pa + (pb-pa)/Nphi * np.arange(Nphi)
    wphi = (pb-pa)/Nphi * (4 if octant else 1)

    # tensor product grid
    nu_grid = np.repeat(nu, Nphi)
    phi_grid = np.tile(phi, nMu)

    # shell Jacobian factor
    jac = (mu*mu - nu_grid*nu_grid)

    # total quadrature weight for each (nu,phi)
    w = np.repeat(wnu, Nphi) * wphi * jac

    # map to Cartesian (nuclei on x-axis at ±R/2)
    x = 0.5 * rAB * mu * nu_grid
    rho = 0.5 * rAB * np.sqrt((mu*mu - 1.0) * (1.0 - nu_grid*nu_grid))
    y = rho * np.cos(phi_grid)
    z = rho * np.sin(phi_grid) if Nphi > 2 else np.zeros_like(x)

    return x, y, z, nu_grid, phi_grid, w


BO = True
M = 1836.153

# Basis set maximum powers (rAB^h * r12^k * s^n * t^m * (mu1^i*mu2^j + mu1^j*mu2^i) * exp( - alpha*s - beta*rAB - gamma*r12 )
h_max = 5
k_max = 8
n_max = 8
m_max = 8
ij_max = 8
total_max = 3
delta = 0.1

# Todo: allow cusp-enabling functions - maybe s^-1 would help? Go back to s1, s2? (^-1, or more ideally ln)?

# delta = 0.1: E[0] := -1.174474883468479:
# E[0] := -1.1744788198234721 at delta=0.1, alpha=0.695

plot_phi_target = np.pi/2*0.1
plot_rAB_target = 1.4
plot_s2_target = 8

nMu = 24
nS = 50
sMax = 50

use_delta = delta != 0

if BO:
    M1M = 1
    M_inv = 0
else:
    M1M = (M + 1) / M
    M_inv = 1 / M

if BO:  # Reset if not used
    h_max = 0  # avoid rAB dependence
    beta = 0

rows = []
for h in frange(0, h_max, 1):
    for k in frange(0, k_max, 1):
        for n in frange(0, n_max, 1):
            for m in range(m_max + 1):  # careful: Whenever using negative indices, adjust power_table call accordingly.
                for i in range(ij_max + 1):
                    for j in range(i + 1):

                        # constraints
                        if (i + j) % 2 != 0: continue
                        if m % 2 != 0: continue
                        t = h + k + n + m + i + j
                        if t > total_max: continue
                        rows.append((h, k, n, m, i, j))  #-k-m

basis_idx = np.array(rows, dtype=np.int16)

# Shorthands for later use (no copies)
h_idx = basis_idx[:, 0]
k_idx = basis_idx[:, 1]
n_idx = basis_idx[:, 2]
m_idx = basis_idx[:, 3]
i_idx = basis_idx[:, 4]
j_idx = basis_idx[:, 5]

h = h_idx.astype(np.float64)
k = k_idx.astype(np.float64)
n = n_idx.astype(np.float64)
m = m_idx.astype(np.float64)
i = i_idx.astype(np.float64)
j = j_idx.astype(np.float64)

Fij = np.stack([n * n - n, n, k * n, n * m, n * (m - h), n * i, n * j,
                 m * m, m, m * k, m * i, m * j, m * (i + j - h),
                 i * i - i, i, i * j, i * k, i * h,
                 j * j - j, j, j * k, j * h,
                 h * h + h, h, k * k + k, k,
                 np.ones_like(n)], axis=0)
Fji = np.stack([n * n - n, n, k * n, n * m, n * (m - h), n * j, n * i,
                m * m, m, m * k, m * j, m * i, m * (j + i - h),
                j * j - j, j, j * i, j * k, j * h,
                i * i - i, i, i * k, i * h,
                h * h + h, h, k * k + k, k,
                np.ones_like(n)], axis=0)

matSize = h_idx.size
print(f"MatSize = {matSize}")

start = time.time()

nrP = 0
abRange = frange(1.4, 1.4, 0.2)

S_layers = {}
H_1_layers = {}
H_alpha_layers = {}
H_alpha_2_layers = {}
H_beta_layers = {}
H_beta_2_layers = {}
H_alpha_beta_layers = {}

plot_mu2_chunks = []
plot_x1_chunks = []
plot_y1_chunks = []
plot_B_chunks  = []
plot_chunks_A_1 = []
plot_chunks_A_alpha = []
plot_chunks_A_beta = []
plot_chunks_A_alpha2 = []
plot_chunks_A_alphabeta = []
plot_chunks_A_beta2 = []

for rAB in abRange:
    s_shells, sW = build_s_shells(rAB, Ks=nS, s_max=sMax, gamma = 3.0)
    idx = np.abs(s_shells - plot_s2_target - plot_rAB_target).argmin()
    plot_s2_target = s_shells[idx] - plot_rAB_target - 0.0001  # Ensure small s1 values are included in the plot-sampling
    print(f"s2 = {plot_s2_target}")

    for ks, s in enumerate(s_shells):

        # rAB, s - only primitives
        inv_rAB = 1.0 / rAB
        rAB_2 = rAB * rAB
        inv_s = 1.0 / s
        inv_s_2 = inv_s * inv_s
        inv_rAB_2 = inv_rAB * inv_rAB

        s1_vals, s2_vals, splitW = split_s(s, rAB, Ku=10)

        Sl = np.zeros((matSize, matSize), dtype=np.float64)
        Hl_1 = np.zeros((matSize, matSize), dtype=np.float64)
        Hl_alpha = np.zeros((matSize, matSize), dtype=np.float64)
        Hl_alpha2 = np.zeros((matSize, matSize), dtype=np.float64)
        Hl_alphabeta = np.zeros((matSize, matSize), dtype=np.float64)
        Hl_beta = np.zeros((matSize, matSize), dtype=np.float64)
        Hl_beta2 = np.zeros((matSize, matSize), dtype=np.float64)

        if s-rAB > plot_s2_target:
            s2_vals = np.append(s2_vals, plot_s2_target)
            s1_vals = np.append(s1_vals, s - plot_s2_target)
            splitW = np.append(splitW, 0.0)

        for j, (s1, s2) in enumerate(zip(s1_vals, s2_vals)):

            # sample electron 1 on its s1-shell
            x1, y1, _, mu1, _, w1 = sample_s_shell(rAB, s1, Nphi=2, nMu=nMu)  # x-y plane only
            rA1 = np.sqrt((x1 + rAB/2) ** 2 + y1 ** 2)
            rB1 = np.sqrt((x1 - rAB/2) ** 2 + y1 ** 2)  # vectorized distances

            # sample electron 2 on its s2-shell
            x2, y2, z2, mu2, phi2, w2 = sample_s_shell(rAB, s2, octant=True, nMu=nMu)
            rA2 = np.sqrt((x2 + rAB/2)**2 + y2**2 + z2**2)
            rB2 = np.sqrt((x2 - rAB/2)**2 + y2**2 + z2**2)

            # e1-only primitives
            inv_rA1 = 1.0 / rA1
            inv_rB1 = 1.0 / rB1
            rA1_2 = rA1 * rA1
            rB1_2 = rB1 * rB1
            # mu1 = (rA1 - rB1) * inv_rAB
            inv_mu1 = 1 / mu1
            inv_mu1_2 = inv_mu1 * inv_mu1
            v1 = inv_rA1 + inv_rB1
            cos1AB = (rA1_2 + rAB_2 - rB1_2) * inv_rA1 * inv_rAB * 0.5
            cos1BA = (rAB_2 + rB1_2 - rA1_2) * inv_rAB * inv_rB1 * 0.5
            cosA1B = (rA1_2 - rAB_2 + rB1_2) * inv_rA1 * inv_rB1 * 0.5
            c11 = M_inv * (cos1AB - cos1BA)
            mu1_p = power_table(mu1, ij_max)

            # e2-only primitives
            inv_rA2 = 1.0 / rA2
            inv_rB2 = 1.0 / rB2
            rA2_2 = rA2 * rA2
            rB2_2 = rB2 * rB2
            # mu2 = (rA2 - rB2) * inv_rAB
            inv_mu2 = 1.0 / mu2
            inv_mu2_2 = inv_mu2 * inv_mu2
            v2 = inv_rA2 + inv_rB2
            cos2AB = (rA2_2 + rAB_2 - rB2_2) * inv_rA2 * inv_rAB * 0.5
            cos2BA = (rAB_2 + rB2_2 - rA2_2) * inv_rAB * inv_rB2 * 0.5
            cosA2B = (rA2_2 - rAB_2 + rB2_2) * inv_rA2 * inv_rB2 * 0.5
            c12 = M_inv * (cos2AB - cos2BA)
            mu2_p = power_table(mu2, ij_max)

            # Mixed but scalar
            t = (s1 - s2) * inv_rAB * 0.5
            inv_t = 1 / t
            inv_t_2 = inv_t * inv_t
            inv_rAB_s = inv_rAB * inv_s
            inv_rAB_t = inv_rAB * inv_t

            P1 = rA1.size
            P2 = rA2.size
            P = P1 * P2

            # expand e1
            rA1 = np.repeat(rA1, P2)
            rB1 = np.repeat(rB1, P2)
            rA1_2 = np.repeat(rA1_2, P2)
            rB1_2 = np.repeat(rB1_2, P2)
            inv_rA1 = np.repeat(inv_rA1, P2)
            inv_rB1 = np.repeat(inv_rB1, P2)
            mu1 = np.repeat(mu1, P2)
            inv_mu1 = np.repeat(inv_mu1, P2)
            inv_mu1_2 = np.repeat(inv_mu1_2, P2)
            v1 = np.repeat(v1, P2)
            cos1AB = np.repeat(cos1AB, P2)
            cos1BA = np.repeat(cos1BA, P2)
            cosA1B = np.repeat(cosA1B, P2)
            c11 = np.repeat(c11, P2)
            mu1_p = np.repeat(mu1_p, P2, axis=0)       # shape (P, Npow)

            # expand e2
            rA2 = np.tile(rA2, P1)
            rB2 = np.tile(rB2, P1)
            rA2_2 = np.tile(rA2_2, P1)
            rB2_2 = np.tile(rB2_2, P1)
            inv_rA2 = np.tile(inv_rA2, P1)
            inv_rB2 = np.tile(inv_rB2, P1)
            mu2 = np.tile(mu2, P1)
            inv_mu2 = np.tile(inv_mu2, P1)
            inv_mu2_2 = np.tile(inv_mu2_2, P1)
            v2 = np.tile(v2, P1)
            cos2AB = np.tile(cos2AB, P1)
            cos2BA = np.tile(cos2BA, P1)
            cosA2B = np.tile(cosA2B, P1)
            c12 = np.tile(c12, P1)
            mu2_p = np.tile(mu2_p, (P1, 1))

            # r12-dependent quantities
            dx = x1[:, None] - x2[None, :]
            dy = y1[:, None] - y2[None, :]
            dz = 0.0         - z2[None, :]
            r12   = np.sqrt(dx*dx + dy*dy + dz*dz).ravel()  # vector of r12 values for all e1, e2 positions
            r12 = np.maximum(r12, 10**(-8))
            r12_p = power_table(r12, k_max)

            inv_r12 = 1.0 / r12
            r12_2 = r12 * r12
            inv_r12_2 = inv_r12 * inv_r12

            v12 = v1 + v2

            cos12A = (r12_2 - rA1_2 + rA2_2) * inv_r12 * inv_rA2 * 0.5
            cos12B = (r12_2 - rB1_2 + rB2_2) * inv_r12 * inv_rB2 * 0.5
            cos1A2 = (rA1_2 + rA2_2 - r12_2) * inv_rA1 * inv_rA2 * 0.5
            cos1B2 = (rB1_2 + rB2_2 - r12_2) * inv_rB1 * inv_rB2 * 0.5
            cos21A = (r12_2 + rA1_2 - rA2_2) * inv_r12 * inv_rA1 * 0.5
            cos21B = (r12_2 + rB1_2 - rB2_2) * inv_r12 * inv_rB1 * 0.5

            # Recurring combinations
            c1 = cos12A + cos12B + cos21A + cos21B
            c2 = cosA1B + cosA2B
            c3 = (cos12A - cos21B + cos12B - cos21A)
            c4 = cos21A - cos21B
            c5 = cos12A - cos12B
            c6 = cosA2B - cosA1B
            c7 = M_inv * (cos1A2 + cos1B2)
            c8 = M_inv * (cos1AB + cos1BA + cos2AB + cos2BA)
            c9 = M_inv * (cos1AB + cos1BA - cos2AB - cos2BA)
            c10 = M_inv * (cos1A2 - cos1B2)

            # Coefficients of: [n^2, n, k*n, n*m, m^2, m, m*k, i^2-i, i, i*k, j^2-j, j, j*k, k^2 + k, k, 1]
            c_n2 = -( 2.0*M1M + c2 + c7 ) * inv_s_2  # n^2 - n
            c_n  = - M1M*v12 * inv_s  # n
            c_n_alpha = ( 2.0*(M1M*2.0 + c2 + c7) ) * inv_s  # n
            c_kn = -c1 * inv_r12 * inv_s  # k*n
            c_nm = c6 * inv_t  * inv_rAB_s  # n*m
            c_nmh= c8 * inv_rAB_s  # n*(m-h)
            c_ni = (c8 - c10 * inv_mu1) * inv_rAB_s  # n*i
            c_nj = (c8 - c10 * inv_mu2) * inv_rAB_s # n*j
            c_m2 = -0.25*( 2*M1M + c2 - c7)*inv_t_2*inv_rAB_2 - M_inv*inv_rAB_2 + 0.5*c9*inv_rAB_2*inv_t  # m^2
            c_m  = 0.5* M1M*(v2-v1) * inv_rAB_t + 0.25*( M1M*2.0 + c2 - c7 )*inv_t_2*inv_rAB_2 + M_inv*inv_rAB_2  # m
            c_m_alpha = - c6 * inv_rAB_t - c8 * inv_rAB
            c_mk = 0.5 * c3 *inv_r12*inv_rAB_t  # m*k
            c_mi = (c11 + 0.5*c10*inv_t) * inv_mu1 * inv_rAB_2  # m*i
            c_mj = (c12 - 0.5*c10*inv_t) * inv_mu2 * inv_rAB_2  # m*j
            c_mijh = ( 0.5 * c9 * inv_t - 2.0*M_inv ) * inv_rAB_2  # m*(i+j-h)
            c_i2 =  (( -M1M  + cosA1B )*inv_mu1_2*inv_rAB_2  + c11*inv_mu1*inv_rAB_2- M_inv*inv_rAB_2) * np.ones_like(c_n)  # i^2-i
            c_i  = ( -M1M*(inv_rA1-inv_rB1)  + c11*inv_rAB ) *inv_mu1*inv_rAB  * np.ones_like(c_n) #i
            c_i_alpha  = ( c10*inv_mu1 - c8 ) *inv_rAB * np.ones_like(c_n)  #i
            c_ij = ( -c7*inv_mu1*inv_mu2 + c11*inv_mu1 + c12*inv_mu2 - 2.0*M_inv ) * inv_rAB_2  # i*j
            c_ik = -c4 * inv_r12 * inv_mu1 * inv_rAB  # i*k
            c_ih = ( -c11*inv_mu1 + 2.0*M_inv ) * inv_rAB_2 * np.ones_like(c_n)  # i*h
            c_j2 = ( -M1M + cosA2B )*inv_mu2_2*inv_rAB_2  + c12*inv_mu2*inv_rAB_2 - M_inv*inv_rAB_2  # j^2-j
            c_j  = ( -M1M*(inv_rA2-inv_rB2) + c12*inv_rAB)*inv_mu2*inv_rAB # j
            c_j_alpha  = ( c10*inv_mu2 - c8)*inv_rAB  # j
            c_jk = -c5*inv_r12*inv_mu2*inv_rAB  # j*k
            c_jh = ( -c12*inv_mu2 + 2.0*M_inv ) * inv_rAB_2  # j*h
            c_h2 = -M_inv*inv_rAB_2 * np.ones_like(c_n)  # h^2 + h
            c_h_alpha = c8 * inv_rAB  * np.ones_like(c_n) # h
            c_k2 = - inv_r12_2  # k^2 + k
            c_k_alpha  = c1 * inv_r12  # k
            c_1_alpha  = M1M*v12
            c_1_alpha2  = -( 2*M1M + c2 + c7 )

            c_n_beta = c8 * inv_s  # n
            c_m_beta = 0.5*c9*inv_rAB_t - 2.*M_inv*inv_rAB
            c_i_beta  = (c11*inv_mu1 - 2.*M_inv) * inv_rAB * np.ones_like(c_n)  #i
            c_j_beta  = (c12*inv_mu2 - 2.*M_inv) * inv_rAB  # j
            c_h_beta  = 2.0 * M_inv * inv_rAB  * np.ones_like(c_n) # h
            c_1_alphabeta = -c8
            c_1_beta =   M_inv*2.*inv_rAB * np.ones_like(c_n)
            c_1_beta2 =  -M_inv * np.ones_like(c_n)

            if use_delta:
                c_m_delta = - 0.5*c3*inv_rAB_t
                c_n_delta = c1 * inv_s  # n
                c_i_delta = c4 * inv_mu1*inv_rAB * np.ones_like(c_n) # i
                c_j_delta = c5 * inv_mu2*inv_rAB  # j
                c_1_alphadelta = -c1
                c_1_delta =  2.0*inv_r12
                c_1_delta2 =  -1.0 * np.ones_like(c_n)
                c_k_delta  = 2.0 * inv_r12  # k

            zero = np.zeros_like(r12)
            coef_vector_1 = np.stack([c_n2, c_n, c_kn, c_nm, c_nmh, c_ni, c_nj, c_m2, c_m, c_mk, c_mi, c_mj, c_mijh, c_i2, c_i, c_ij, c_ik, c_ih, c_j2, c_j, c_jk, c_jh, c_h2, zero, c_k2, zero, zero], axis=1)
            coef_vector_alpha = np.stack([zero, c_n_alpha, zero, zero, zero, zero, zero, zero, c_m_alpha, zero, zero, zero, zero, zero, c_i_alpha, zero, zero, zero, zero, c_j_alpha, zero, zero, zero, c_h_alpha, zero, c_k_alpha, c_1_alpha], axis=1)
            coef_vector_alpha2 = np.stack([zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, c_1_alpha2], axis=1)

            coef_vector_beta = np.stack([zero, c_n_beta, zero, zero, zero, zero, zero, zero, c_m_beta, zero, zero, zero, zero, zero, c_i_beta, zero, zero, zero, zero, c_j_beta, zero, zero, zero, c_h_beta, zero, zero, c_1_beta], axis=1)
            coef_vector_alphabeta = np.stack([zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, c_1_alphabeta], axis=1)
            coef_vector_beta2 = np.stack([zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, c_1_beta2], axis=1)

            if use_delta:
                coef_vector_delta = np.stack([zero, c_n_delta, zero, zero, zero, zero, zero, zero, c_m_delta, zero, zero, zero, zero, zero, c_i_delta, zero, zero, zero, zero, c_j_delta, zero, zero, zero, zero, zero, c_k_delta, c_1_delta], axis=1)
                coef_vector_alphadelta = np.stack([zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, c_1_alphadelta], axis=1)
                coef_vector_delta2 = np.stack([zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, c_1_delta2], axis=1)

            potential = potential_ri(rAB, rA1, rB1, rA2, rB2, r12)
            H_1_ij = coef_vector_1 @ Fij + potential[:, None]
            H_1_ji = coef_vector_1 @ Fji + potential[:, None]
            if use_delta:
                H_1_ij += (coef_vector_delta * delta + coef_vector_delta2 * delta ** 2) @ Fij
                H_1_ji += (coef_vector_delta * delta + coef_vector_delta2 * delta ** 2) @ Fji

            H_alpha_ij = coef_vector_alpha @ Fij
            H_alpha_ji = coef_vector_alpha @ Fji
            if use_delta:
                tmp = coef_vector_alphadelta*delta @ Fij
                H_alpha_ij += tmp
                H_alpha_ji += tmp
            H_alpha2 = coef_vector_alpha2 @ Fij
            H_alphabeta = coef_vector_alphabeta @ Fij
            H_beta_ij = coef_vector_beta @ Fij
            H_beta_ji = coef_vector_beta @ Fji
            H_beta2 = coef_vector_beta2 @ Fij

            # weights
            W0 = sW[ks] * splitW[j]  # scalar
            Wpair = W0 * (w1[:, None] * w2[None, :]).ravel()  # (P,)
            sqrtW = np.sqrt(Wpair)  # (P,)

            scale = np.exp(n_idx * np.log(2.0 * 0.75) - 0.5 * gammaln(2 * n_idx + 1))[None, :]

            # Assemble basis functions from power matrices
            B = np.ones((r12_p.shape[0], matSize), dtype=np.float64) * (rAB**h_idx*s**n_idx*t**m_idx)[None, :] * scale
            B *= r12_p[:, k_idx]
            B *= np.exp(-delta*r12)[:, None]

            mu_part_ij = mu1_p[:, i_idx] * mu2_p[:, j_idx]
            mu_part_ji = mu1_p[:, j_idx] * mu2_p[:, i_idx]

            if abs(s2-plot_s2_target) < 1e-8:
                mask_e2, phi_used = mask_closest_phi(phi2, plot_phi_target)
                mask_pair = np.tile(mask_e2, P1)
                x1_sel = np.repeat(x1, P2)[mask_pair]
                y1_sel = np.repeat(y1, P2)[mask_pair]
                mu2_sel = mu2[mask_pair]
                B_ij = (B * mu_part_ij)
                B_ji = (B * mu_part_ji)
                B_plot = (B_ij + B_ji)[mask_pair, :]
                plot_mu2_chunks.append(mu2_sel)
                plot_x1_chunks.append(x1_sel)
                plot_y1_chunks.append(y1_sel)
                plot_B_chunks.append(B_plot)

                A_1_full = H_1_ij * B_ij + H_1_ji * B_ji
                A_alpha_full = H_alpha_ij * B_ij + H_alpha_ji * B_ji
                A_beta_full = H_beta_ij * B_ij + H_beta_ji * B_ji
                Bs = (B_ij + B_ji)
                A_alpha2_full = H_alpha2 * Bs
                A_beta2_full = H_beta2 * Bs
                A_ab_full = H_alphabeta * Bs
                A_1_sel       = A_1_full[mask_pair, :]
                A_alpha_sel   = A_alpha_full[mask_pair, :]
                A_beta_sel    = A_beta_full[mask_pair, :]
                A_alpha2_sel  = A_alpha2_full[mask_pair, :]
                A_beta2_sel   = A_beta2_full[mask_pair, :]
                A_ab_sel      = A_ab_full[mask_pair, :]
                assert B_plot.shape == A_alpha_sel.shape
                plot_chunks_A_1.append(A_1_sel)
                plot_chunks_A_alpha.append(A_alpha_sel)
                plot_chunks_A_beta.append(A_beta_sel)
                plot_chunks_A_alpha2.append(A_alpha2_sel)
                plot_chunks_A_alphabeta.append(A_ab_sel)
                plot_chunks_A_beta2.append(A_beta2_sel)

            B *= sqrtW[:, None]
            B_ij = B * mu_part_ij
            B_ji = B * mu_part_ji

            B = B_ij + B_ji

            Sl += B.T @ B
            Hl_1 += B.T @ (H_1_ij * B_ij + H_1_ji * B_ji)
            Hl_alpha += B.T @ (H_alpha_ij * B_ij + H_alpha_ji * B_ji)
            Hl_beta += B.T @ (H_beta_ij * B_ij + H_beta_ji * B_ji)
            Hl_alpha2 += B.T @ (H_alpha2 * (B_ij + B_ji))
            Hl_alphabeta += B.T @ (H_alphabeta * (B_ij + B_ji))
            Hl_beta2 += B.T @ (H_beta2 * (B_ij + B_ji))

            # H += (B.T @ A) * np.exp(-2*alpha*s - 2*beta*rAB)

            # if abs(rA1 - 2) < 0.6 and abs(theta - 1 * np.pi / 4) < 0.2 and (BO or abs(rAB - 1.4) < 0.0001):  # change to 1/4*Pi to see electron
            #     print(theta, x1, y1, rA1, rB1)
            #     test_A = A
            #     test_B = B  # Vectors of rows HP and P for later comparison of local energy deviation  # todo

            nrP += P

        S_layers[rAB, s] = Sl
        H_1_layers[rAB, s] = Hl_1
        H_alpha_layers[rAB, s] = Hl_alpha
        H_alpha_2_layers[rAB, s] = Hl_alpha2
        H_alpha_beta_layers[rAB, s] = Hl_alphabeta
        H_beta_layers[rAB, s] = Hl_beta
        H_beta_2_layers[rAB, s] = Hl_beta2

        print(ks, rAB, s, nrP, time.time() - start)

mu2_all = np.concatenate(plot_mu2_chunks)
x1_all = np.concatenate(plot_x1_chunks)
y1_all = np.concatenate(plot_y1_chunks)
B_plot_all = np.vstack(plot_B_chunks)
A_1_all = np.vstack(plot_chunks_A_1)
A_alpha_all = np.vstack(plot_chunks_A_alpha)
A_beta_all = np.vstack(plot_chunks_A_beta)
A_alpha2_all = np.vstack(plot_chunks_A_alpha2)
A_alphabeta_all = np.vstack(plot_chunks_A_alphabeta)
A_beta2_all = np.vstack(plot_chunks_A_beta2)

plot_mu2_with_alpha_beta(
    x1_all=x1_all, y1_all=y1_all, mu2_all=mu2_all,
    B_plot_all=B_plot_all,
    A_1_all=A_1_all, A_alpha_all=A_alpha_all, A_beta_all=A_beta_all,
    A_alpha2_all=A_alpha2_all, A_alphabeta_all=A_alphabeta_all, A_beta2_all=A_beta2_all,

    S_layers=S_layers,
    H_1_layers=H_1_layers,
    H_alpha_layers=H_alpha_layers,
    H_beta_layers=H_beta_layers,
    H_alpha_2_layers=H_alpha_2_layers,
    H_beta_2_layers=H_beta_2_layers,
    H_alpha_beta_layers=H_alpha_beta_layers,

    matSize=matSize,
    diag_rescale_generalized=diag_rescale_generalized,

    plot_rAB_target=plot_rAB_target,
    plot_s2_target=plot_s2_target,
    plot_phi_target=plot_phi_target,

    alpha_values=np.arange(0.6, 1.3, 0.005),
    beta_values=np.array([0.0]),

    zlim_eloc=(-3, 0),
    subsample=4000,
)

# for alpha in frange(0.75, 0.75, 0.01):
#     S = np.zeros((matSize, matSize), dtype=np.float64)
#     H = np.zeros((matSize, matSize), dtype=np.float64)
#
#     # alpha = 0.74
#     beta = 0
#
#     for (rAB, s), Sl in S_layers.items(): S += Sl * np.exp(-2*alpha*s - 2*beta*rAB)
#     for (rAB, s), Hl in H_1_layers.items(): H += Hl * np.exp(-2*alpha*s - 2*beta*rAB)
#     for (rAB, s), Hl in H_alpha_layers.items(): H += Hl * alpha * np.exp(-2*alpha*s - 2*beta*rAB)
#     for (rAB, s), Hl in H_alpha_2_layers.items(): H += Hl * alpha**2 * np.exp(-2*alpha*s - 2*beta*rAB)
#     for (rAB, s), Hl in H_alpha_beta_layers.items(): H += Hl * alpha*beta * np.exp(-2*alpha*s - 2*beta*rAB)
#     for (rAB, s), Hl in H_beta_layers.items(): H += Hl * beta * np.exp(-2*alpha*s - 2*beta*rAB)
#     for (rAB, s), Hl in H_beta_2_layers.items(): H += Hl * beta**2 * np.exp(-2*alpha*s - 2*beta*rAB)
#
#     H, S = diag_rescale_generalized(H, S)
#
#     eigvals = np.linalg.eigvalsh(S)
#     # print("After Diag rescaling:")
#     # print(eigvals)
#     # print("min eig:", np.abs(eigvals).min())
#     # print("max eig:", np.abs(eigvals).max())
#     print(f"\nalpha = {alpha}, cond: {eigvals.max() / eigvals.min():.3e}")
#
#     # inspect_small_overlap_eigenvectors(S)
#
#     h_idx = basis_idx[:, 0]
#     k_idx = basis_idx[:, 1]
#     n_idx = basis_idx[:, 2]
#     m_idx = basis_idx[:, 3]
#     i_idx = basis_idx[:, 4]
#     j_idx = basis_idx[:, 5]
#
#     E, C = eig(H, S)
#     idx = np.argsort(E)
#     E = np.real(E[idx])
#     C = np.real(C[:, idx])
#
#     i = 0
#     while i < len(E) and E[i] < 0:
#         ci = C[:, i]
#         ci = ci/ci[0]
#         # hp = test_A @ ci
#         # p = test_B @ ci
#
#         eps = np.linalg.norm(H @ ci - E[i] * (S @ ci)) / (np.linalg.norm(H @ ci) + 1e-30)
#         print(f"E[{i}] := {E[i]}: epsilon[{i}] := {eps}: "
#               # f"C[{i}] := {[f' + ({float(x)}) * rAB^{h_idx[ii]}*r12^{k_idx[ii]}*s^{n_idx[ii]}*t^{m_idx[ii]}*mu1^{i_idx[ii]}*mu2^{j_idx[ii]}' for ii, x in enumerate(ci)]}")
#               f"C[{i}] := " + "".join( f" + ({float(x)})*rAB^{h_idx[ii]}*r12^{k_idx[ii]}*s^{n_idx[ii]}*t^{m_idx[ii]}*mu1^{i_idx[ii]}*mu2^{j_idx[ii]}" for ii, x in enumerate(ci)) )
#         i += 1
#
#     x = x1_all
#     y = y1_all
#     rAB = plot_rAB_target
#     s = plot_s2_target + np.sqrt((x + 0.5 * rAB) ** 2 + y ** 2) + np.sqrt((x - 0.5 * rAB) ** 2 + y ** 2)
#
#     plot_phi_and_local_energy_mu2(
#         x1_all, y1_all, mu2_all,
#         B_plot_all,
#         A_1_all,
#         A_alpha_all,
#         A_beta_all,
#         A_alpha2_all,
#         A_alphabeta_all,
#         A_beta2_all,
#         rAB = plot_rAB_target,
#         phi = plot_phi_target,
#         s2 = plot_s2_target,
#         exps = np.exp(-alpha * s - beta * rAB)[:, None],
#         C = C, E = E,
#         alpha=alpha, beta=beta, zlim_eloc=(-3, 0)
#     )
