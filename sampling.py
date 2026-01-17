
import numpy as np
from numpy.polynomial.legendre import leggauss

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


def build_s_shells(rAB, Ks, s_max, gamma=1.0):
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

def sample_s_shell_old(rAB, s, nMu=12, Nphi=16, octant=False):
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


def sample_s_shell(rAB, s, nMu=12, Nphi=16, p_phi=3.0):
    """
    Like sample_s_shell(..., octant=True) but with phi clustered near 0
    using phi = (pi/2) * u^p, u uniform in [0,1).
    Weights include dphi/du Jacobian and the usual (mu^2-nu^2).
    """
    mu = s / rAB

    # nu in [0,1] for octant, with mirrored weight factor handled outside (same as your code)
    nu0, w0 = leggauss(nMu)
    a, b = 0.0, 1.0
    nu = 0.5*(b-a)*nu0 + 0.5*(a+b)
    wnu = 0.5*(b-a)*w0 * 2.0  # keep your "undo half" logic for octant

    # biased phi on [0, pi/2)
    u = (np.arange(Nphi) + 0.5) / Nphi          # midpoint rule in u
    phi = (0.5*np.pi) * (u ** p_phi)
    # dphi = (pi/2) * p * u^(p-1) du, du = 1/Nphi
    wphi = (0.5*np.pi) * p_phi * (u ** (p_phi - 1.0)) * (1.0 / Nphi)
    wphi = wphi * 4.0  # your octant mirroring factor

    nu_grid = np.repeat(nu, Nphi)
    phi_grid = np.tile(phi, nMu)

    jac = (mu*mu - nu_grid*nu_grid)
    w = np.repeat(wnu, Nphi) * np.tile(wphi, nMu) * jac

    x = 0.5 * rAB * mu * nu_grid
    rho = 0.5 * rAB * np.sqrt((mu*mu - 1.0) * (1.0 - nu_grid*nu_grid))
    y = rho * np.cos(phi_grid)
    z = rho * np.sin(phi_grid)

    return x,y,z,nu_grid,phi_grid,w




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

import numpy as np
from math import factorial

def _mu_int(n: int, p: float) -> float:
    # ∫_1^∞ μ^n e^{-pμ} dμ = e^{-p} Σ_{j=0..n} n!/(n-j)! * 1/p^{j+1}
    nfac = factorial(n)
    s = 0.0
    for j in range(n + 1):
        s += nfac / factorial(n - j) / (p ** (j + 1))
    return np.exp(-p) * s

def I3_exp(rAB: float, k: float) -> float:
    a = 0.5 * rAB
    p = k * rAB
    I2 = _mu_int(2, p)
    I0 = _mu_int(0, p)
    return 2*np.pi * a**3 * (2*I2 - (2/3)*I0)

def I3_s_exp(rAB: float, k: float) -> float:
    # ∫ s e^{-k s} dV = - d/dk I3
    # Since p=k rAB, d/dk = rAB d/dp
    a = 0.5 * rAB
    p = k * rAB

    # d/dp ∫ μ^n e^{-pμ} dμ = - ∫ μ^{n+1} e^{-pμ} dμ
    dI2_dp = -_mu_int(3, p)
    dI0_dp = -_mu_int(1, p)

    dI3_dp = 2*np.pi * a**3 * (2*dI2_dp - (2/3)*dI0_dp)
    return -(rAB * dI3_dp)
import numpy as np

def integrate_6d_via_split(rAB, s_max, Ks=48, Ku=32, Nnu=32, Nphi=64, f=None, gamma=2.0):
    """
    Numerically approximates ∫ f(...) d^3r1 d^3r2 over the domain s1>=R, s2>=R
    using your:
      build_s_shells(rAB, Ks, s_max, gamma)
      split_s(s, rAB, Ku)
      sample_s_shell(rAB, s1), sample_s_shell(rAB, s2)
    """
    s_grid, ws = build_s_shells(rAB, Ks, s_max, gamma=gamma)  # :contentReference[oaicite:5]{index=5}
    a = 0.5 * rAB
    pref = (a**3 / rAB)**2   # (a^3 dμ)^2 with dμ=ds/rAB  => overall rAB factors handled here

    total = 0.0
    for s, w_s in zip(s_grid, ws):
        s1, s2, w_split = split_s(s, rAB, Ku)  # :contentReference[oaicite:6]{index=6}
        for si, sj, w_ij in zip(s1, s2, w_split):
            x1,y1,z1,nu1,phi1,wsh1 = sample_s_shell(rAB, si, nMu=Nnu, Nphi=Nphi, octant=False)  # :contentReference[oaicite:7]{index=7}
            x2,y2,z2,nu2,phi2,wsh2 = sample_s_shell(rAB, sj, nMu=Nnu, Nphi=Nphi, octant=False)  # :contentReference[oaicite:8]{index=8}

            # evaluate on tensor product of shell grids
            if f is None:
                vals = 1.0
                block = np.sum(wsh1) * np.sum(wsh2)
            else:
                # build r12 if needed
                dx = x1[:,None]-x2[None,:]
                dy = y1[:,None]-y2[None,:]
                dz = z1[:,None]-z2[None,:]
                r12 = np.sqrt(dx*dx+dy*dy+dz*dz)

                vals = f(si, sj, s, x1,y1,z1, x2,y2,z2, r12)
                block = np.sum((wsh1[:,None]*wsh2[None,:]) * vals)

            total += w_s * w_ij * block

    return pref * total
def test_split_swap_zero(rAB=1.4, s_max=60.0, k=0.8):
    num = integrate_6d_via_split(
        rAB, s_max,
        f=lambda s1,s2,st, *args: (s1 - s2) * np.exp(-k*st)
    )
    print("swap-antisym (should be 0):", num)
def test_split_s1_moment(rAB=1.4, s_max=60.0, k=0.8):
    num_s1 = integrate_6d_via_split(
        rAB, s_max,
        f=lambda s1,s2,st, *args: (s1) * np.exp(-k*st)
    )
    num_s2 = integrate_6d_via_split(
        rAB, s_max,
        f=lambda s1,s2,st, *args: (s2) * np.exp(-k*st)
    )

    # exact over infinite range (your finite s_max should be large enough with exp decay)
    ex = I3_exp(rAB,k) * I3_s_exp(rAB,k)

    print("∫ s1 e^{-k(s1+s2)}:", num_s1)
    print("∫ s2 e^{-k(s1+s2)}:", num_s2)
    print("exact:", ex)
    print("relerr s1:", abs(num_s1-ex)/abs(ex))
    print("relerr s2:", abs(num_s2-ex)/abs(ex))
    print("swap diff (should be 0):", num_s1-num_s2)
def integrate_3d_generic(rAB, k, Kmu=80, Nnu=24, Nphi=48, g=None):
    t, wt = np.polynomial.laguerre.laggauss(Kmu)
    p = k * rAB
    mu = 1.0 + t / p
    a = 0.5 * rAB
    pref = (np.exp(-p)/p) * a**3

    total = 0.0
    for mui, wti in zip(mu, wt):
        s = mui * rAB
        x,y,z,nu,phi,wsh = sample_s_shell(rAB, s, nMu=Nnu, Nphi=Nphi, octant=False)  # :contentReference[oaicite:10]{index=10}
        vals = g(x,y,z,mui,nu,phi,s)
        total += wti * np.sum(wsh * vals)
    return pref * total

def test_parity(rAB=1.4, k=0.8):
    Ix = integrate_3d_generic(rAB,k, g=lambda x,y,z,mu,nu,phi,s: x*np.exp(-k*s))
    Iy = integrate_3d_generic(rAB,k, g=lambda x,y,z,mu,nu,phi,s: y*np.exp(-k*s))
    print("∫ x e^{-ks} dV (should be 0):", Ix)
    print("∫ y e^{-ks} dV (should be 0):", Iy)
def integrate_6d_gaussian_r12(rAB, alpha, beta, Kmu=22, Nnu=12, Nphi=24, p_map=1.2):
    """
    Generic μ quadrature: μ = 1 + t/p_map, t∈[0,∞).
    Uses Laguerre weights w (for ∫ e^{-t} g(t) dt), so to integrate ∫ f(t) dt we use g(t)=e^{t} f(t).
    """
    t, w = np.polynomial.laguerre.laggauss(Kmu)
    mu = 1.0 + t / p_map
    dmu = 1.0 / p_map

    a = 0.5 * rAB
    pref = (a**3 * dmu)**2

    total = 0.0
    for i, mui in enumerate(mu):
        s1 = mui * rAB
        x1,y1,z1,nu1,phi1,w1 = sample_s_shell(rAB, s1, nMu=Nnu, Nphi=Nphi, octant=False)  # :contentReference[oaicite:11]{index=11}
        r1_2 = x1*x1 + y1*y1 + z1*z1

        for j, muj in enumerate(mu):
            s2 = muj * rAB
            x2,y2,z2,nu2,phi2,w2 = sample_s_shell(rAB, s2, nMu=Nnu, Nphi=Nphi, octant=False)  # :contentReference[oaicite:12]{index=12}
            r2_2 = x2*x2 + y2*y2 + z2*z2

            dx = x1[:,None]-x2[None,:]
            dy = y1[:,None]-y2[None,:]
            dz = z1[:,None]-z2[None,:]
            r12_2 = dx*dx+dy*dy+dz*dz

            integrand = np.exp(-alpha*(r1_2[:,None] + r2_2[None,:]) - beta*r12_2)
            block = np.sum((w1[:,None]*w2[None,:]) * integrand)

            # convert Laguerre to plain dt integral: multiply by exp(t_i+t_j)
            total += (w[i]*np.exp(t[i])) * (w[j]*np.exp(t[j])) * block

    return pref * total

def exact_6d_gaussian(alpha, beta):
    return (np.pi**3) / ((alpha*(alpha+2.0*beta))**1.5)

def test_r12_gaussian(rAB=1.4, alpha=0.7, beta=0.4):
    num = integrate_6d_gaussian_r12(rAB, alpha, beta)
    ex  = exact_6d_gaussian(alpha, beta)
    print("6D Gaussian(r12) num:", num)
    print("6D Gaussian(r12) exact:", ex)
    print("relerr:", abs(num-ex)/abs(ex))
def exact_6d_gaussian_r12sq(alpha, beta):
    I = exact_6d_gaussian(alpha, beta)
    return (3.0/(alpha+2.0*beta)) * I

def test_r12_moment(rAB=1.4, alpha=0.7, beta=0.4):
    def integrate_r12sq():
        t, w = np.polynomial.laguerre.laggauss(22)
        p_map=1.2
        mu = 1.0 + t / p_map
        dmu = 1.0 / p_map
        a = 0.5*rAB
        pref = (a**3*dmu)**2

        total = 0.0
        for i, mui in enumerate(mu):
            s1 = mui*rAB
            x1,y1,z1,nu1,phi1,w1 = sample_s_shell(rAB, s1, nMu=12, Nphi=24, octant=False)  # :contentReference[oaicite:13]{index=13}
            r1_2 = x1*x1 + y1*y1 + z1*z1
            for j, muj in enumerate(mu):
                s2 = muj*rAB
                x2,y2,z2,nu2,phi2,w2 = sample_s_shell(rAB, s2, nMu=12, Nphi=24, octant=False)  # :contentReference[oaicite:14]{index=14}
                r2_2 = x2*x2 + y2*y2 + z2*z2

                dx = x1[:,None]-x2[None,:]
                dy = y1[:,None]-y2[None,:]
                dz = z1[:,None]-z2[None,:]
                r12_2 = dx*dx+dy*dy+dz*dz

                base = np.exp(-alpha*(r1_2[:,None]+r2_2[None,:]) - beta*r12_2)
                block = np.sum((w1[:,None]*w2[None,:]) * (r12_2 * base))

                total += (w[i]*np.exp(t[i]))*(w[j]*np.exp(t[j])) * block

        return pref * total

    num = integrate_r12sq()
    ex  = exact_6d_gaussian_r12sq(alpha, beta)
    print("6D r12^2 moment num:", num)
    print("6D r12^2 moment exact:", ex)
    print("relerr:", abs(num-ex)/abs(ex))

test_parity()
test_r12_gaussian()
test_r12_moment()
test_split_swap_zero()
test_split_s1_moment()