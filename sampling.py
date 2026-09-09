import numpy as np
from numpy.polynomial.legendre import leggauss

# TODO: sinc numerical methods
# TODO: manual exact calculation of leggauss nodes

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

def build_rAB_grid(
    KR: int,
    R_min: float,
    R_max: float,
    Re: float,
    gamma: float = 2.0,
):
    """
    Composite Gauss–Legendre quadrature for ∫_{R_min}^{R_max} f(R) 4π R^2 dR
    with clustering of nodes near Re from BOTH sides.

    Strategy:
      - Split [R_min, R_max] at Re
      - Left side:  R = Re - (Re-R_min) * (1-u)^gamma   (clusters near Re as u→1)
      - Right side: R = Re + (R_max-Re) * (u^gamma)     (clusters near Re as u→0)
      - u comes from Gauss–Legendre nodes mapped to [0,1]
      - Correct Jacobians included

    gamma=1 gives plain composite GL on the two sub-intervals (no extra clustering).
    """
    if KR < 2:
        raise ValueError("KR must be >= 2")
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1.0")
    if not (R_min < Re < R_max):
        raise ValueError("Re must lie strictly inside (R_min, R_max)")
    if R_max <= R_min:
        raise ValueError("R_max must be > R_min")

    # Allocate points proportional to interval lengths (asymmetric OK)
    frac_left = (Re - R_min) / (R_max - R_min)
    KR_left = int(np.clip(np.round(KR * frac_left), 1, KR - 1))
    KR_right = KR - KR_left

    # --- Left interval [R_min, Re], clustered toward Re ---
    xL, wL = leggauss(KR_left)            # [-1,1]
    uL = 0.5 * (xL + 1.0)                 # [0,1]
    wuL = 0.5 * wL

    aL = (Re - R_min)
    # R = Re - aL*(1-u)^gamma
    one_minus_u = (1.0 - uL)
    RL = Re - aL * (one_minus_u ** gamma)
    dRdu_L = aL * gamma * (one_minus_u ** (gamma - 1.0))

    wRL = wuL * dRdu_L * (4.0 * np.pi * RL * RL)

    # --- Right interval [Re, R_max], clustered toward Re ---
    xR, wR = leggauss(KR_right)
    uR = 0.5 * (xR + 1.0)
    wuR = 0.5 * wR

    aR = (R_max - Re)
    # R = Re + aR*u^gamma
    RR = Re + aR * (uR ** gamma)
    dRdu_R = aR * gamma * (uR ** (gamma - 1.0))

    wRR = wuR * dRdu_R * (4.0 * np.pi * RR * RR)

    # Combine (already sorted within each side; left ends at Re, right starts at Re)
    R = np.concatenate([RL, RR])
    wR = np.concatenate([wRL, wRR])

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

    x, w = leggauss(Ku)

    # Center clustering on [-1,1] (more s1~=s2)
    xg = np.sign(x) * np.abs(x) ** gamma

    if gamma == 1.0:
        dxg_dx = np.ones_like(x)
    else:
        dxg_dx = gamma * np.abs(x) ** (gamma - 1.0)

    u_center = 0.5 * (xg + 1.0)
    du_center_dx = 0.5 * dxg_dx

    # End clustering on [0,1], built from the same xg
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

    span = s - 2.0 * rAB
    s1 = rAB + u * span
    s2 = s - s1

    w_split = w * du_dx * span

    return s1, s2, w_split

def sample_s_shell(rAB, s, nMu=12, Nphi=32, octant=False, s1 = 1.0, gamma_phi = 8.0):
    """
    Deterministic quadrature on the prolate spheroidal shell rA+rB = s.
    Returns (x,y,z, w) arrays of length nMu*Nphi.

    R = full internuclear distance (same R used in mu=s/R)
    """

    mu = s / rAB

    nu0, w0 = leggauss(nMu)
    a, b = (0.0, 1.0) if octant else (-1.0, 1.0)
    nu = 0.5*(b-a)*nu0 + 0.5*(a+b)
    wnu = 0.5*(b-a)*w0 * (2.0 if octant else 1.0)  # (Immediately undo half-weighting, since octant will be mirrored back)

    # phi trapezoid grid on [0,2π)
    if octant:
        # phi, wphi = phi2_chebyshev_quadrant(Nphi)
        # wphi = np.tile(wphi, nMu)
        # # print(phi)

        phi0, w0 = leggauss(Nphi)
        a, b = (0.0, 0.5*np.pi)
        phi = 0.5*(b-a)*phi0 + 0.5*(a+b)
        wphi = 0.5*(b-a)*w0 * 4.0
        wphi = np.tile(wphi, nMu)

        # phi = 0.5 * np.pi / Nphi * np.arange(Nphi) + 0.25 * np.pi / Nphi
        # wphi = (0.5 * np.pi) / Nphi * 4.0  # 4: octant mirroring factor
        # print(phi)
    else:
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
