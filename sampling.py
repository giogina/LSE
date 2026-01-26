
import numpy as np
from numpy.polynomial.laguerre import laggauss
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

import numpy as np
from numpy.polynomial.legendre import leggauss

def build_rAB_grid(
    KR: int,
    R_min: float,
    R_max: float,
    gamma: float = 2.0,
):
    """
    Quadrature nodes/weights for integrating over R = rAB.

    Returns R, wR such that:
      measure="plain":  ∫_{R_min}^{R_max} f(R) dR  ≈ Σ wR[i] f(R[i])
      measure="R2":     ∫_{R_min}^{R_max} f(R) 4πR^2 dR ≈ Σ wR[i] f(R[i])

    gamma>1 clusters points toward R_min (useful if small-R is stiff).
    If you want clustering toward BOTH ends, see the note below.
    """
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1.0")
    if R_max <= R_min:
        raise ValueError("R_max must be > R_min")

    # Gauss-Legendre on [-1,1] -> u in [0,1]
    x, w = leggauss(KR)
    u = 0.5 * (x + 1.0)
    wu = 0.5 * w

    # Power map u -> t in [0,1] with clustering near 0
    t = u ** gamma
    dt_du = gamma * (u ** (gamma - 1.0))

    # Map to R
    R = R_min + (R_max - R_min) * t
    dR_du = (R_max - R_min) * dt_du

    # Base weights for dR
    wR = wu * dR_du
    wR = wR * (4.0 * np.pi * R * R)

    return R, wR


def build_s_shells(rAB, Ks, s_max, gamma=3.0):  # (Yes, leggauss is used on purpose. It works better than laggauss here, somehow.)
    s_min = 2.0 * rAB

    x, w = leggauss(Ks)         # [-1,1]
    u = 0.5*(x + 1.0)           # [0,1]
    wu = 0.5*w

    s = s_min + (s_max - s_min) * (u**gamma)
    ds_du = (s_max - s_min) * gamma * (u**(gamma - 1.0))
    ws = wu * ds_du
    return s, ws

import numpy as np
from numpy.polynomial.legendre import leggauss

def split_s(s, rAB, Ku, gamma=6.0, eta_end=0.5, end_mode="both"):
    """
    Deterministic split of total s into s1,s2 with two simultaneous clusterings:
      1) around s1=s2 (center of interval) via x -> sign(x)|x|^gamma
      2) around s1≈rAB (and optionally s2≈rAB) via an "end clustering" map

    Parameters
    ----------
    gamma : float >= 1
        Controls strength of both clusterings.
    eta_end : float in [0,1]
        Mix between center-clustering (0) and end-clustering (1).
    end_mode : {"both", "left"}
        "both": cluster near both ends (s1≈rAB and s2≈rAB)
        "left": cluster only near left end (s1≈rAB)

    Returns
    -------
    s1, s2, w_split
    """
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1.0")
    if not (0.0 <= eta_end <= 1.0):
        raise ValueError("eta_end must be in [0,1]")
    if end_mode not in ("both", "left"):
        raise ValueError("end_mode must be 'both' or 'left'")

    # Gauss-Legendre on [-1,1]
    x, w = leggauss(Ku)

    # --- (A) Center clustering on [-1,1]: densify near x=0 -> u=0.5 (=> s1=s2)
    xg = np.sign(x) * np.abs(x) ** gamma

    if gamma == 1.0:
        dxg_dx = np.ones_like(x)
    else:
        dxg_dx = gamma * np.abs(x) ** (gamma - 1.0)

    u_center = 0.5 * (xg + 1.0)
    du_center_dx = 0.5 * dxg_dx

    # --- (B) End clustering on [0,1], built from the SAME xg (so "same gamma")
    # u0 is already center-clustered; we now remap it to densify near ends.
    u0 = u_center
    du0_dx = du_center_dx

    if end_mode == "both":
        # cosine map clusters near u=0 and u=1 (endpoints) with smooth Jacobian
        u_end = 0.5 * (1.0 - np.cos(np.pi * u0))
        du_end_du0 = 0.5 * np.pi * np.sin(np.pi * u0)
        du_end_dx = du_end_du0 * du0_dx
    else:  # "left"
        # power map clusters near u=0 only
        u_end = u0 ** gamma
        if gamma == 1.0:
            du_end_du0 = np.ones_like(u0)
        else:
            du_end_du0 = gamma * np.maximum(u0, 0.0) ** (gamma - 1.0)
        du_end_dx = du_end_du0 * du0_dx

    # --- Combine monotonically (convex combo of monotone maps stays monotone)
    u = (1.0 - eta_end) * u_center + eta_end * u_end
    du_dx = (1.0 - eta_end) * du_center_dx + eta_end * du_end_dx

    # Map u -> s1 in [rAB, s-rAB]
    span = s - 2.0 * rAB
    s1 = rAB + u * span
    s2 = s - s1

    # Weights: dx -> u -> s1
    w_split = w * du_dx * span

    return s1, s2, w_split

def shell_area_weight(s1, s2, rAB):
    shell_area_1 = 4*np.pi*((s1 / rAB)**2 - 1/3)
    shell_area_2 = 4*np.pi*((s2 / rAB)**2 - 1/3)
    return shell_area_1 * shell_area_2

def sample_s_shell(rAB, s, nMu=12, Nphi=32, octant=False, s1 = 1.0):
    """
    Deterministic quadrature on the prolate spheroidal shell rA+rB = s.
    Returns (x,y,z, w) arrays of length nMu*Nphi.

    R = full internuclear distance (same R used in mu=s/R)
    """
    if octant: # todo: test further
        ds = np.abs(s-s1)  # difference between s1, s2 (small ds -> check more small phi values for small r12)
        gamma_phi = gamma_phi_from_ds(ds, gamma_max=8.)
        return sample_s_shell_phi_bias_octant(rAB, s, nMu+1, Nphi, gamma_phi) # * int(np.sqrt(gamma_phi))
        # return sample_s_shell_phi_bias_octant(rAB, s, nMu, Nphi, 3.0) # todo: the 1.3 factor helped a lot too

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


def sample_s_shell_octant_mc(
    rAB, s, ds,
    N=4096,
    gamma_max=6.0,
    d0=None,
    stratified=True,
    seed=0,
):
    """
    Monte-Carlo / stratified importance sampling on the octant for a fixed prolate shell rA+rB = s.

    Samples:
      nu in [0,1], phi in [0,pi/2)

    Proposal:
      nu ~ Uniform(0,1)
      phi generated by phi = (pi/2) * u^gamma, u ~ Uniform(0,1)
      where gamma = gamma_phi_from_ds(ds)

    Weights implement:
      ∫_0^1 ∫_0^{pi/2} f(nu,phi) (mu^2-nu^2) dphi dnu * 4
    so that downstream you can just do sum_i w_i * f_i.

    Returns: x,y,z, nu,phi, w   (arrays length N)
    """
    mu = s / rAB
    if d0 is None:
        d0 = 0.2 * rAB  # sensible default scale

    gamma_phi = gamma_phi_from_ds(ds, d0=d0, gamma_max=gamma_max)

    rng = np.random.default_rng(seed)

    # ---- sample nu, u with optional stratification ----
    if stratified:
        # 1D stratification for nu and u (phi), cheap and effective
        i = np.arange(N)
        nu = (i + rng.random(N)) / N
        u  = (rng.permutation(N) + rng.random(N)) / N  # permute to decorrelate
    else:
        nu = rng.random(N)
        u  = rng.random(N)

    # ---- phi importance map: phi = (pi/2) * u^gamma ----
    phi = 0.5*np.pi * (u ** gamma_phi)

    # proposal pdfs
    q_nu = np.ones_like(nu)  # uniform on [0,1]
    # u uniform -> phi pdf = du/dphi
    # u = (phi/(pi/2))^(1/gamma)
    # du/dphi = (1/gamma) * (2/pi)^(1/gamma) * phi^(1/gamma - 1)
    # safe formula using u: du/dphi = 1 / (dphi/du)
    dphi_du = 0.5*np.pi * gamma_phi * (u ** (gamma_phi - 1.0))
    q_phi = 1.0 / dphi_du

    # shell Jacobian factor (same as your deterministic code)
    jac = (mu*mu - nu*nu)

    # MC weights for the angular integral over the octant
    w = 4.0 * jac / (N * (q_nu * q_phi))

    # coordinates (same mapping you use)
    x = 0.5 * rAB * mu * nu
    rho = 0.5 * rAB * np.sqrt((mu*mu - 1.0) * (1.0 - nu*nu))
    y = rho * np.cos(phi)
    z = rho * np.sin(phi)

    return x, y, z, nu, phi, w

def gamma_phi_from_ds(ds, d0 = 0.4, gamma_max = 8.0):  # compute phi clustering parameter from ds = s1-s2 (big clustering for small ds)
    return 1.0 + (gamma_max - 1.0) / (1.0 + (ds / d0)**2)

def sample_s_shell_phi_bias_octant(rAB, s, nMu=12, Nphi=16, gamma_phi=3.0):
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
    wnu = 0.5*(b-a)*w0 * 2.0  # undo halfing of nu space in weight

    # biased phi on [0, pi/2)
    u = (np.arange(Nphi) + 0.5) / Nphi          # midpoint rule in u
    phi = (0.5*np.pi) * (u ** gamma_phi)
    wphi = (0.5*np.pi) * gamma_phi * (u ** (gamma_phi - 1.0)) * (1.0 / Nphi)
    wphi = wphi * 4.0  # octant mirroring factor

    nu_grid = np.repeat(nu, Nphi)
    phi_grid = np.tile(phi, nMu)

    jac = (mu*mu - nu_grid*nu_grid)
    w = np.repeat(wnu, Nphi) * np.tile(wphi, nMu) * jac

    x = 0.5 * rAB * mu * nu_grid
    rho = 0.5 * rAB * np.sqrt((mu*mu - 1.0) * (1.0 - nu_grid*nu_grid))
    y = rho * np.cos(phi_grid)
    z = rho * np.sin(phi_grid)

    return x,y,z,nu_grid,phi_grid,w



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
    # s_grid, ws = build_s_shells(rAB, Ks, s_max, gamma=gamma)  # :contentReference[oaicite:5]{index=5}
    s_grid, ws = build_s_shells(rAB, Ks)  # :contentReference[oaicite:5]{index=5}
    a = 0.5 * rAB
    pref = (a**3 / rAB)**2   # (a^3 dμ)^2 with dμ=ds/rAB  => overall rAB factors handled here

    total = 0.0
    for s, w_s in zip(s_grid, ws):
        s1, s2, w_split = split_s(s, rAB, Ku)  # :contentReference[oaicite:6]{index=6}
        for si, sj, w_ij in zip(s1, s2, w_split):
            x1,y1,z1,nu1,phi1,wsh1 = sample_s_shell(rAB, si, nMu=Nnu, Nphi=2, octant=False)  # :contentReference[oaicite:7]{index=7}
            x2,y2,z2,nu2,phi2,wsh2 = sample_s_shell(rAB, sj, nMu=Nnu, Nphi=32, octant=True, s1=s1)  # :contentReference[oaicite:8]{index=8}

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
        x1,y1,z1,nu1,phi1,w1 = sample_s_shell(rAB, s1, nMu=12, Nphi=2, octant=False)
        r1_2 = x1*x1 + y1*y1 + z1*z1

        for j, muj in enumerate(mu):
            s2 = muj * rAB
            x2,y2,z2,nu2,phi2,w2 = sample_s_shell(rAB, s2, nMu=12, Nphi=12, octant=True, s1=s1)
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

        # s_shells, sW = build_s_shells(rAB, Ks=30, s_max=50, gamma=3.0)
        # for ks, s in enumerate(s_shells):
        #     s1_vals, s2_vals, splitW = split_s(s, rAB, Ku=13)
        #
        #     for j, (s1, s2) in enumerate(zip(s1_vals, s2_vals)):
        #         x1, y1, _, mu1, _, w1 = sample_s_shell(rAB, s1, Nphi=2, nMu=12)  # x-y plane only
        #         x2, y2, z2, mu2, _, w2 = sample_s_shell(rAB, s2, octant=True, nMu=12, Nphi=12, s1=s1)
        #
        #         r1_2 = x1*x1 + y1*y1
        #         r2_2 = x2*x2 + y2*y2 + z2*z2
        #
        #         dx = x1[:,None]-x2[None,:]
        #         dy = y1[:,None]-y2[None,:]
        #         dz = -z2[None,:]
        #         r12_2 = dx*dx+dy*dy+dz*dz
        #
        #         base = np.exp(-alpha*(r1_2[:,None]+r2_2[None,:]) - beta*r12_2)
        #         block = np.sum((w1[:,None]*w2[None,:]) * (r12_2 * base))

        t, w = np.polynomial.laguerre.laggauss(22)
        p_map=1.2
        mu = 1.0 + t / p_map
        dmu = 1.0 / p_map
        a = 0.5*rAB
        pref = (a**3*dmu)**2

        total = 0.0
        for i, mui in enumerate(mu):
            s1 = mui*rAB
            x1,y1,z1,nu1,phi1,w1 = sample_s_shell(rAB, s1, nMu=12, Nphi=2, octant=False)
            r1_2 = x1*x1 + y1*y1 + z1*z1
            for j, muj in enumerate(mu):
                s2 = muj*rAB
                x2,y2,z2,nu2,phi2,w2 = sample_s_shell(rAB, s2, nMu=12, Nphi=24, octant=True, s1=s1)
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

def plot_r12_weight_cdf(
    rAB=1.4,
    nS=20,
    sMax=40.0,
    gamma=3.0,
    Ku=11,
    nMu=12,
    max_pairs=2_000_0000,
    seed=0,
):
    """
    Build the *actual* r12-weight distribution implied by your current sampling:
      build_s_shells -> split_s -> sample_s_shell(e1 in xy-plane) -> sample_s_shell(e2 octant)
    and plot:
      (1) CDF of total weight vs r12
      (2) weighted r12 histogram mass (log-y)

    Weight per pair is exactly:
        W = sW[ks] * splitW[j] * w1[i] * w2[j]
    which mirrors how main.py passes sW[ks]*splitW[j] into calc_AB and combines with w1,w2 there.
    """

    rng = np.random.default_rng(seed)

    s_shells, sW = build_s_shells(rAB, Ks=nS, s_max=sMax)

    r12_chunks = []
    w_chunks = []
    total_pairs = 0

    for ks, s in enumerate(s_shells):
        s1_vals, s2_vals, splitW = split_s(s, rAB, Ku=11, gamma = 3.0)

        outer_w = sW[ks]

        for j_split, (s1, s2) in enumerate(zip(s1_vals, s2_vals)):
            # electron 1: xy plane only (Nphi=2)
            x1, y1, z1, mu1, phi1, w1 = sample_s_shell(rAB, s1, Nphi=2, nMu=nMu, octant=False)

            # electron 2: octant (your biased-phi version is inside sample_s_shell now)
            x2, y2, z2, mu2, phi2, w2 = sample_s_shell(rAB, s2, Nphi=24, nMu=nMu, octant=True, s1=s1)

            W_outer = outer_w * splitW[j_split]

            dx = x1[:, None] - x2[None, :]
            dy = y1[:, None] - y2[None, :]
            dz = z1[:, None] - z2[None, :]
            r12 = np.sqrt(dx*dx + dy*dy + dz*dz)

            W = (W_outer * w1[:, None] * w2[None, :])

            flat_r12 = r12.ravel()
            flat_W = W.ravel()

            n = flat_r12.size
            if total_pairs + n > max_pairs:
                remaining = max_pairs - total_pairs
                if remaining <= 0:
                    break
                idx = rng.choice(n, size=remaining, replace=False)
                flat_r12 = flat_r12[idx]
                flat_W = flat_W[idx]
                n = remaining

            r12_chunks.append(flat_r12.astype(np.float64, copy=False))
            w_chunks.append(flat_W.astype(np.float64, copy=False))
            total_pairs += n

        if total_pairs >= max_pairs:
            break

    r12_all = np.concatenate(r12_chunks) if r12_chunks else np.array([], dtype=np.float64)
    w_all = np.concatenate(w_chunks) if w_chunks else np.array([], dtype=np.float64)

    mask = np.isfinite(r12_all) & np.isfinite(w_all) & (w_all > 0)
    r12_all = r12_all[mask]
    w_all = w_all[mask]

    if r12_all.size == 0:
        raise RuntimeError("No samples collected. Check your sampling parameters.")

    # CDF
    order = np.argsort(r12_all)
    r_sorted = r12_all[order]
    w_sorted = w_all[order]
    cdf = np.cumsum(w_sorted)
    cdf /= cdf[-1]

    plt.figure()
    plt.plot(r_sorted, cdf)
    plt.xlabel("r12")
    plt.ylabel("Cumulative weight fraction (<= r12)")
    plt.title(f"Weight CDF vs r12  (rAB={rAB}, pairs≈{r_sorted.size})")
    plt.grid(True)
    plt.show()

    # Weighted histogram mass (log-y)
    bins = np.linspace(0.0, np.percentile(r12_all, 99.9), 200)
    hist, edges = np.histogram(r12_all, bins=bins, weights=w_all)
    centers = 0.5*(edges[:-1] + edges[1:])
    hist = hist / np.sum(hist)

    plt.figure()
    plt.semilogy(centers, hist + 1e-300)
    plt.xlabel("r12")
    plt.ylabel("Weighted bin mass (log scale)")
    plt.title("Weighted r12 distribution (mass per bin, not /Δr)")
    plt.grid(True)
    plt.show()


    return r_sorted, cdf, centers, hist





# print("--")
# test_r12_gaussian()
# test_r12_moment()
# plot_r12_measure_diagnostics()
# plot_r12_weight_cdf(rAB=1.4, nS=40, sMax=50, gamma=3.0, Ku=10, nMu=24)
# test_split_swap_zero()
# test_split_s1_moment()