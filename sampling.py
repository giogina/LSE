
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
