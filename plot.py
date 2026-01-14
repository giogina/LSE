import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.tri as mtri
import matplotlib as mpl



def init_plot_chunks():
    plot_chunks = {}
    plot_chunks["mu2"] = []
    plot_chunks["x1"] = []
    plot_chunks["y1"] = []
    plot_chunks["B"] = []
    plot_chunks["A_1"] = []
    plot_chunks["A_alpha"] = []
    plot_chunks["A_alpha2"] = []
    plot_chunks["A_beta"] = []
    plot_chunks["A_beta2"] = []
    plot_chunks["A_alphabeta"] = []
    return plot_chunks

def update_plot_chunks(plot_chunks, mu2, B, P1, P2, x1, y1, phi2, plot_phi_target, A_1, A_alpha, A_alpha2, A_alphabeta, A_beta, A_beta2):
    mask_e2, phi_used = mask_closest_phi(phi2, plot_phi_target)
    mask_pair = np.tile(mask_e2, P1)
    x1_sel = np.repeat(x1, P2)[mask_pair]
    y1_sel = np.repeat(y1, P2)[mask_pair]
    mu2 = np.tile(mu2, P1)
    mu2_sel = mu2[mask_pair]

    B_plot = B[mask_pair, :]
    plot_chunks["mu2"].append(mu2_sel)
    plot_chunks["x1"].append(x1_sel)
    plot_chunks["y1"].append(y1_sel)
    plot_chunks["B"].append(B_plot)

    A_1_sel = A_1[mask_pair, :]
    A_alpha_sel = A_alpha[mask_pair, :]
    A_beta_sel = A_beta[mask_pair, :]
    A_alpha2_sel = A_alpha2[mask_pair, :]
    A_beta2_sel = A_beta2[mask_pair, :]
    A_ab_sel = A_alphabeta[mask_pair, :]
    assert B_plot.shape == A_alpha_sel.shape
    plot_chunks["A_1"].append(A_1_sel)
    plot_chunks["A_alpha"].append(A_alpha_sel)
    plot_chunks["A_beta"].append(A_beta_sel)
    plot_chunks["A_alpha2"].append(A_alpha2_sel)
    plot_chunks["A_alphabeta"].append(A_ab_sel)
    plot_chunks["A_beta2"].append(A_beta2_sel)

    return plot_chunks

def mask_closest_phi(phi, phi_target, rtol=1e-12, atol=1e-12):
    """
    Select all points whose phi equals the sampled phi value
    closest to phi_target (with 2π wrapping).
    Returns (mask, phi_selected).
    """
    # wrapped distance to target
    d = (phi - phi_target + np.pi) % (2*np.pi) - np.pi

    # index of closest sampled phi
    i0 = np.argmin(np.abs(d))
    phi_sel = phi[i0]

    # mask all points matching that sampled phi (with wrapping)
    d_sel = (phi - phi_sel + np.pi) % (2*np.pi) - np.pi
    mask = np.isclose(d_sel, 0.0, rtol=rtol, atol=atol)

    return mask, phi_sel


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.tri as mtri
from scipy.linalg import eig


def plot_mu2_with_alpha_beta(
    *,
    # plotting geometry/data (already assembled from your capture)
    plot_chunks,

    # layer dictionaries for assembling S/H (same as your outer code)
    S_layers,
    H_1_layers,
    H_alpha_layers,
    H_beta_layers,
    H_alpha_2_layers,
    H_beta_2_layers,
    H_alpha_beta_layers,

    # sizes + helpers
    matSize,
    diag_rescale_generalized,

    # plotting constants needed for exps and e2-dot
    plot_rAB_target,
    plot_s2_target,
    plot_phi_target,

    # alpha/beta slider grids (discrete; caching makes sense)
    alpha_values,
    beta_values,

    # numerics / plot options
    only_negative_E=True,
    eps=1e-16,
    mu2_rtol=1e-12,
    mu2_atol=1e-12,
    zlim_eloc=(-3, 0),     # view only
    subsample=None,        # per mu2 slice; e.g. 4000
    jitter_duplicates=0.0, # e.g. 1e-12 if triangulation complains
):
    """
    Interactive 3D plotting with sliders:
      - alpha index
      - beta index
      - mu2 index
      - solution i (over E<0 by default)

    Caches previously computed (alpha,beta) eigensystems and per-mu2 projections.

    Applies missing exponential factor:
      exps(x1,y1) = exp(-alpha*s_total(x1,y1) - beta*rAB)
    where s_total = s2_target + rA1 + rB1 (electron 1 in plane).
    """

    mu2_all = np.concatenate(plot_chunks["mu2"])
    x1_all = np.concatenate(plot_chunks["x1"])
    y1_all = np.concatenate(plot_chunks["y1"])
    B_plot_all = np.vstack(plot_chunks["B"])
    A_1_all = np.vstack(plot_chunks["A_1"])
    A_alpha_all = np.vstack(plot_chunks["A_alpha"])
    A_beta_all = np.vstack(plot_chunks["A_beta"])
    A_alpha2_all = np.vstack(plot_chunks["A_alpha2"])
    A_alphabeta_all = np.vstack(plot_chunks["A_alphabeta"])
    A_beta2_all = np.vstack(plot_chunks["A_beta2"])

    # ---------- helpers ----------
    def e2_xyz_from_mu(mu2, rAB, s2, phi2):
        mu_shell = s2 / rAB
        x2 = 0.5 * rAB * mu_shell * mu2
        rho = 0.5 * rAB * np.sqrt((mu_shell * mu_shell - 1.0) * (1.0 - mu2 * mu2))
        y2 = rho * np.cos(phi2)
        z2 = rho * np.sin(phi2)
        return x2, y2, z2

    def compute_s_total_for_e1(x1, y1, rAB, s2_target):
        # s_total = s2_target + rA1 + rB1, with nuclei at ±rAB/2 on x axis
        rA1 = np.sqrt((x1 + 0.5 * rAB) ** 2 + y1 ** 2)
        rB1 = np.sqrt((x1 - 0.5 * rAB) ** 2 + y1 ** 2)
        return s2_target + rA1 + rB1

    # ---------- normalize inputs ----------
    x1_all = np.asarray(x1_all).ravel()
    y1_all = np.asarray(y1_all).ravel()
    mu2_all = np.asarray(mu2_all).ravel()
    Npts = x1_all.size
    assert y1_all.size == Npts and mu2_all.size == Npts

    B_plot_all = np.asarray(B_plot_all); Nb = B_plot_all.shape[1]
    assert B_plot_all.shape[0] == Npts

    A_1_all = np.asarray(A_1_all);                 assert A_1_all.shape == (Npts, Nb)
    A_alpha_all = np.asarray(A_alpha_all);         assert A_alpha_all.shape == (Npts, Nb)
    A_beta_all = np.asarray(A_beta_all);           assert A_beta_all.shape == (Npts, Nb)
    A_alpha2_all = np.asarray(A_alpha2_all);       assert A_alpha2_all.shape == (Npts, Nb)
    A_alphabeta_all = np.asarray(A_alphabeta_all); assert A_alphabeta_all.shape == (Npts, Nb)
    A_beta2_all = np.asarray(A_beta2_all);         assert A_beta2_all.shape == (Npts, Nb)

    alpha_values = np.asarray(alpha_values, dtype=float)
    beta_values = np.asarray(beta_values, dtype=float)

    mu2_unique = np.unique(mu2_all)
    if mu2_unique.size == 0:
        raise ValueError("mu2_all has no values.")

    # Precompute s_total for each point once (used to build exps)
    s_total_all = compute_s_total_for_e1(x1_all, y1_all, plot_rAB_target, plot_s2_target)

    # Cache structure:
    # eig_cache[(ia,ib)] = {"alpha":a,"beta":b,"E":E,"C":C, "sol_idx": idx_array,
    #                       "mu_cache": {jmu: {"tri":..., "x":..., "y":..., "mu":..., "exps":..., "BC":..., "A1C":..., ...}}}
    eig_cache = {}

    def assemble_HS(alpha, beta):
        S = np.zeros((matSize, matSize), dtype=np.float64)
        H = np.zeros((matSize, matSize), dtype=np.float64)

        # S
        for (rAB, s), Sl in S_layers.items():
            S += Sl * np.exp(-2 * alpha * s - 2 * beta * rAB)

        # H base + polynomial alpha/beta parts
        for (rAB, s), Hl in H_1_layers.items():
            H += Hl * np.exp(-2 * alpha * s - 2 * beta * rAB)
        for (rAB, s), Hl in H_alpha_layers.items():
            H += Hl * alpha * np.exp(-2 * alpha * s - 2 * beta * rAB)
        for (rAB, s), Hl in H_alpha_2_layers.items():
            H += Hl * (alpha ** 2) * np.exp(-2 * alpha * s - 2 * beta * rAB)
        for (rAB, s), Hl in H_alpha_beta_layers.items():
            H += Hl * (alpha * beta) * np.exp(-2 * alpha * s - 2 * beta * rAB)
        for (rAB, s), Hl in H_beta_layers.items():
            H += Hl * beta * np.exp(-2 * alpha * s - 2 * beta * rAB)
        for (rAB, s), Hl in H_beta_2_layers.items():
            H += Hl * (beta ** 2) * np.exp(-2 * alpha * s - 2 * beta * rAB)

        H, S = diag_rescale_generalized(H, S)
        return H, S

    def get_cached_eigs(ia, ib):
        key = (int(ia), int(ib))
        if key in eig_cache:
            return eig_cache[key]

        alpha = float(alpha_values[ia])
        beta = float(beta_values[ib])

        H, S = assemble_HS(alpha, beta)

        # condition estimate
        eigS = np.linalg.eigvalsh(S)
        cond = float(eigS.max() / eigS.min())

        E, C = eig(H, S)
        idx = np.argsort(np.real(E))
        E = np.real(E[idx])
        C = np.real(C[:, idx])
        scale = C[0, :]
        scale[scale == 0.] = 1.0
        C /= scale

        # print solution
        i = 0
        ci = C[:, i]
        ci = ci / ci[0]
        eps = np.linalg.norm(H @ ci - E[i] * (S @ ci)) / (np.linalg.norm(H @ ci) + 1e-30)
        print(f"E[{i}] := {E[i]}: cond = {cond}, epsilon[{i}] := {eps}: "
              # f"C[{i}] := {[f' + ({float(x)}) * rAB^{h_idx[ii]}*r12^{k_idx[ii]}*s^{n_idx[ii]}*t^{m_idx[ii]}*mu1^{i_idx[ii]}*mu2^{j_idx[ii]}' for ii, x in enumerate(ci)]}")
              f"C[{i}] := " + ",".join(f" {float(x)}" for ii, x in enumerate(ci)))

        if only_negative_E:
            sol_idx = np.where(E < 0)[0]
            if sol_idx.size == 0:
                sol_idx = np.arange(E.size)
        else:
            sol_idx = np.arange(E.size)

        entry = {
            "alpha": alpha,
            "beta": beta,
            "H": H,
            "S": S,
            "condS": cond,
            "E": E,
            "C": C,              # (Nb, nsol)
            "sol_idx": sol_idx,  # indices into E/C
            "mu_cache": {}
        }
        eig_cache[key] = entry
        return entry

    def build_mu_slice(cache_entry, jmu):
        # cached per (alpha,beta) AND mu2 index:
        mu_cache = cache_entry["mu_cache"]
        jmu = int(jmu)
        if jmu in mu_cache:
            return mu_cache[jmu]

        mu_val = float(mu2_unique[jmu])

        mask = np.isclose(mu2_all, mu_val, rtol=mu2_rtol, atol=mu2_atol)
        if not np.any(mask):
            return None

        x = x1_all[mask]
        y = y1_all[mask]
        s_total = s_total_all[mask]  # for exps on this slice

        # matrices on this slice
        B  = B_plot_all[mask, :]
        A1 = A_1_all[mask, :]
        Aa = A_alpha_all[mask, :]
        Ab = A_beta_all[mask, :]
        Aa2 = A_alpha2_all[mask, :]
        Aab = A_alphabeta_all[mask, :]
        Ab2 = A_beta2_all[mask, :]

        # optional subsample
        if subsample is not None and x.size > subsample:
            idx = np.random.choice(x.size, size=subsample, replace=False)
            x, y, s_total = x[idx], y[idx], s_total[idx]
            B, A1, Aa, Ab, Aa2, Aab, Ab2 = B[idx,:], A1[idx,:], Aa[idx,:], Ab[idx,:], Aa2[idx,:], Aab[idx,:], Ab2[idx,:]

        if jitter_duplicates and jitter_duplicates > 0:
            x = x + jitter_duplicates * np.random.randn(x.size)
            y = y + jitter_duplicates * np.random.randn(y.size)

        tri = mtri.Triangulation(x, y)

        # Precompute projections for ALL selectable solutions for this (alpha,beta).
        C = cache_entry["C"]                      # (Nb, nsol)
        sol_idx = cache_entry["sol_idx"]
        Csel = C[:, sol_idx]                      # (Nb, nshown)

        BC  = (B  @ Csel).T
        A1C = (A1 @ Csel).T
        AaC = (Aa @ Csel).T
        AbC = (Ab @ Csel).T
        Aa2C = (Aa2 @ Csel).T
        AabC = (Aab @ Csel).T
        Ab2C = (Ab2 @ Csel).T

        mu_cache[jmu] = {
            "mu": mu_val,
            "x": x, "y": y, "tri": tri,
            "s_total": s_total,   # (npts_mu,)
            "BC": BC,
            "A1C": A1C,
            "AaC": AaC,
            "AbC": AbC,
            "Aa2C": Aa2C,
            "AabC": AabC,
            "Ab2C": Ab2C,
        }
        return mu_cache[jmu]

    # ---------- figure ----------
    fig = plt.figure(figsize=(14, 8))
    ax_phi = fig.add_subplot(1, 2, 1, projection="3d")
    ax_eloc = fig.add_subplot(1, 2, 2, projection="3d")
    fig.subplots_adjust(bottom=0.32)

    ax_alpha = fig.add_axes([0.15, 0.23, 0.7, 0.04])
    ax_beta  = fig.add_axes([0.15, 0.18, 0.7, 0.04])
    ax_mu2   = fig.add_axes([0.15, 0.13, 0.7, 0.04])
    ax_i     = fig.add_axes([0.15, 0.08, 0.7, 0.04])

    s_alpha = Slider(ax_alpha, "alpha idx", 0, len(alpha_values)-1, valinit=0, valstep=1)
    s_beta  = Slider(ax_beta,  "beta idx",  0, len(beta_values)-1,  valinit=0, valstep=1)
    s_mu2   = Slider(ax_mu2,   "mu2 idx",   0, mu2_unique.size-1,   valinit=0, valstep=1)

    # i slider max depends on alpha/beta; we’ll recreate its bounds dynamically
    s_i = Slider(ax_i, "i", 0, 1, valinit=0, valstep=1)

    def update_i_slider_max(n):
        # hack: Slider doesn't like changing valmax cleanly; rebuild the slider axis.
        nonlocal s_i
        ax_i.cla()
        s_i = Slider(ax_i, "i", 0, max(0, n-1), valinit=min(int(s_i.val), max(0, n-1)), valstep=1)
        s_i.on_changed(redraw)

    def redraw(_=None):
        ia = int(s_alpha.val)
        ib = int(s_beta.val)
        jmu = int(s_mu2.val)

        cache_entry = get_cached_eigs(ia, ib)
        alpha = cache_entry["alpha"]
        beta = cache_entry["beta"]
        E = cache_entry["E"]
        sol_idx = cache_entry["sol_idx"]

        # ensure i-slider matches available solution count
        if int(s_i.val) > sol_idx.size - 1 or int(s_i.valmax) != sol_idx.size - 1:
            update_i_slider_max(sol_idx.size)

        ii = int(s_i.val)
        i_real = int(sol_idx[ii])  # index into E/C arrays

        mu_slice = build_mu_slice(cache_entry, jmu)
        if mu_slice is None:
            ax_phi.clear(); ax_eloc.clear()
            ax_phi.set_title("No points for this mu2 slice")
            fig.canvas.draw_idle()
            return

        x = mu_slice["x"]
        y = mu_slice["y"]
        tri = mu_slice["tri"]
        s_total = mu_slice["s_total"]

        # pointwise exponential factor (your missing basis factor)
        exps = np.exp(-alpha * s_total - beta * plot_rAB_target)

        # projections for selected solution ii (already masked to this mu slice)
        psi0 = mu_slice["BC"][ii, :]
        A1c  = mu_slice["A1C"][ii, :]
        Aac  = mu_slice["AaC"][ii, :]
        Abc  = mu_slice["AbC"][ii, :]
        Aa2c = mu_slice["Aa2C"][ii, :]
        Aabc = mu_slice["AabC"][ii, :]
        Ab2c = mu_slice["Ab2C"][ii, :]

        # apply missing exp factor
        psi = exps * psi0

        Hpsi = exps * (
            A1c
            + alpha * Aac
            + beta  * Abc
            + (alpha**2) * Aa2c
            + (alpha*beta) * Aabc
            + (beta**2) * Ab2c
        )

        denom = np.where(np.abs(psi) < eps, np.nan, psi)
        Eloc = Hpsi / denom

        # quality metric: SSE of (Eloc - E) over available points
        E_i = float(E[i_real])
        diff = Eloc - E_i
        epsilon = float(np.nansum(diff * diff))

        # clear axes
        ax_phi.clear()
        ax_eloc.clear()

        trisurf_colored(ax_phi, tri, psi, cmap_name="viridis")
        if zlim_eloc is not None:
            trisurf_colored(ax_eloc, tri, Eloc, cmap_name="viridis", vmin=zlim_eloc[0], vmax=zlim_eloc[1])
            ax_eloc.set_zlim(zlim_eloc[0], zlim_eloc[1])
        else:
            trisurf_colored(ax_eloc, tri, Eloc, cmap_name="viridis")

        # mark electron 2 position (orange dot)
        mu_val = float(mu_slice["mu"])
        x2, y2, z2 = e2_xyz_from_mu(mu_val, plot_rAB_target, plot_s2_target, plot_phi_target)
        # put dot at bottom of z-range so it is always visible
        zmin1, zmax1 = ax_phi.get_zlim()
        # zmin2, zmax2 = ax_eloc.get_zlim()
        ax_phi.scatter([x2], [y2], [zmax1], c=["orange"], s=200, depthshade=False)
        # ax_eloc.scatter([x2], [y2], [zmax2], c=["orange"], s=200, depthshade=False)

        # titles with 10 digits energy
        ax_phi.set_title(
            f"ψ | alpha={alpha:.6f} beta={beta:.6f}  cond(S)={cache_entry['condS']:.3e}\n"
            f"i={i_real}  E={E[i_real]:.10f}  mu2={mu_val:.6g}"
        )
        # ax_eloc.set_title(f"Eloc | i={i_real}  E={E[i_real]:.10f}  mu2={mu_val:.6g}")
        ax_eloc.set_title(f"Eloc | i={i_real}  epsilon={epsilon:.10e}  mu2={mu_val:.6g}")

        ax_phi.set_xlabel("x1"); ax_phi.set_ylabel("y1"); ax_phi.set_zlabel("ψ")
        ax_eloc.set_xlabel("x1"); ax_eloc.set_ylabel("y1"); ax_eloc.set_zlabel("Eloc")

        if zlim_eloc is not None:
            ax_eloc.set_zlim(zlim_eloc[0], zlim_eloc[1])

        fig.canvas.draw_idle()

    # wire sliders
    s_alpha.on_changed(redraw)
    s_beta.on_changed(redraw)
    s_mu2.on_changed(redraw)
    s_i.on_changed(redraw)

    def on_key(event):
        # mu2 with up/down
        if event.key == "down":
            s_mu2.set_val(max(0, int(s_mu2.val) - 1))
        elif event.key == "up":
            s_mu2.set_val(min(mu2_unique.size - 1, int(s_mu2.val) + 1))

        # alpha with left/right
        elif event.key == "left":
            s_alpha.set_val(max(0, int(s_alpha.val) - 1))
        elif event.key == "right":
            s_alpha.set_val(min(len(alpha_values) - 1, int(s_alpha.val) + 1))

        # beta with pageup/pagedown
        elif event.key == "pageup":
            s_beta.set_val(min(len(beta_values) - 1, int(s_beta.val) + 1))
        elif event.key == "pagedown":
            s_beta.set_val(max(0, int(s_beta.val) - 1))

    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()
    plt.show(block=True)


def trisurf_colored(ax, tri, z, cmap_name="viridis", vmin=None, vmax=None):
    # one scalar per triangle
    z_tri = np.nanmean(z[tri.triangles], axis=1)

    if vmin is None:
        vmin = np.nanmin(z_tri)
    if vmax is None:
        vmax = np.nanmax(z_tri)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap(cmap_name)
    facecolors = cmap(norm(z_tri))

    # create surface WITHOUT facecolors
    surf = ax.plot_trisurf(
        tri,
        z,
        linewidth=0.0,
        antialiased=True,
        shade=False
    )

    # now override facecolors explicitly
    surf.set_facecolor(facecolors)

    return surf



def plot_phi_and_local_energy_mu2(
    x1_all, y1_all, mu2_all,
    B_plot_all,
    A_1_all,
    A_alpha_all,
    A_beta_all,
    A_alpha2_all,
    A_alphabeta_all,
    A_beta2_all,
    exps,
    C, E,
    *,
    alpha=0.0,
    beta=0.0,

    # --- (b) electron-2 marker ---
    rAB=None,
    s2=None,
    phi=None,

    # numerical
    eps=1e-12,
    mu2_rtol=1e-12,
    mu2_atol=1e-12,

    # view / speed
    zlim_eloc=None,          # e.g. (-3, 0) affects view only
    subsample=None,          # e.g. 3000 (applied per mu2 slice)
    jitter_duplicates=0.0,   # e.g. 1e-12 if triangulation complains

    # --- (c) precompute/cache ---
    precompute=True,
    only_negative_E=True,    # mimic your old plotting habit
):
    """
    Interactive 3D trisurf plots with sliders:
      - i: eigen-solution index (optionally only E<0)
      - mu2 idx: selects available mu2 values
      - xmax / ymax: symmetric view limits for both 3D axes

    Left:  ψ(x1,y1)
    Right: Eloc = Hψ/ψ

    Notes on electron-2 dot:
      The surface is in (x1,y1,z). Electron 2 lives in a *different* coordinate.
      We plot the dot at (x2,y2) in the same x/y frame only if your mapper returns
      coordinates in that same frame. Otherwise, pass a mapper that returns the
      appropriate projected (x,y).
    """

    # --- normalize inputs ---
    x1_all = np.asarray(x1_all).ravel()
    y1_all = np.asarray(y1_all).ravel()
    mu2_all = np.asarray(mu2_all).ravel()
    Npts = x1_all.size
    assert y1_all.size == Npts and mu2_all.size == Npts

    B_plot_all = np.asarray(B_plot_all) * exps
    Nb = B_plot_all.shape[1]
    assert B_plot_all.shape[0] == Npts

    A_1_all = np.asarray(A_1_all) * exps;                 assert A_1_all.shape == (Npts, Nb)
    A_alpha_all = np.asarray(A_alpha_all) * exps;         assert A_alpha_all.shape == (Npts, Nb)
    A_beta_all = np.asarray(A_beta_all) * exps;           assert A_beta_all.shape == (Npts, Nb)
    A_alpha2_all = np.asarray(A_alpha2_all) * exps;       assert A_alpha2_all.shape == (Npts, Nb)
    A_alphabeta_all = np.asarray(A_alphabeta_all) * exps; assert A_alphabeta_all.shape == (Npts, Nb)
    A_beta2_all = np.asarray(A_beta2_all) * exps;         assert A_beta2_all.shape == (Npts, Nb)

    E = np.asarray(E).ravel()

    # C can be (Nb,nsol) or (nsol,Nb)
    C = np.asarray(C)
    if C.shape[0] == Nb:
        C_full = C
    elif C.shape[1] == Nb:
        C_full = C.T
    else:
        raise ValueError(f"C shape {C.shape} incompatible with Nb={Nb}")

    nsol_all = C_full.shape[1]
    assert E.shape[0] == nsol_all, f"E has len {E.shape[0]} but C has {nsol_all} solutions"

    # choose solutions to show
    if only_negative_E:
        sol_idx = np.where(np.real(E) < 0)[0]
        if sol_idx.size == 0:
            sol_idx = np.arange(nsol_all)  # fall back
    else:
        sol_idx = np.arange(nsol_all)

    # unique mu2 values
    mu2_unique = np.unique(mu2_all)
    if mu2_unique.size == 0:
        raise ValueError("mu2_all has no values?")

    # --- helper: assemble Hpsi from layers ---
    def Hpsi_from_layers(A1c, Aac, Abc, Aa2c, Aabc, Ab2c):
        return (A1c
                + alpha * Aac
                + beta  * Abc
                + (alpha**2) * Aa2c
                + (alpha*beta) * Aabc
                + (beta**2) * Ab2c)

    # --- (c) cache per mu2 slice ---
    # Each entry stores geometry + triangulation + (optionally) precomputed projections.
    cache = []

    def build_slice(mu_val):
        mask = np.isclose(mu2_all, mu_val, rtol=mu2_rtol, atol=mu2_atol)
        if not np.any(mask):
            return None

        x = x1_all[mask]
        y = y1_all[mask]

        B  = B_plot_all[mask, :]
        A1 = A_1_all[mask, :]
        Aa = A_alpha_all[mask, :]
        Ab = A_beta_all[mask, :]
        Aa2 = A_alpha2_all[mask, :]
        Aab = A_alphabeta_all[mask, :]
        Ab2 = A_beta2_all[mask, :]

        # optional subsample for speed (apply consistently to all arrays)
        if subsample is not None and x.size > subsample:
            idx = np.random.choice(x.size, size=subsample, replace=False)
            x, y = x[idx], y[idx]
            B  = B[idx, :]
            A1 = A1[idx, :]
            Aa = Aa[idx, :]
            Ab = Ab[idx, :]
            Aa2 = Aa2[idx, :]
            Aab = Aab[idx, :]
            Ab2 = Ab2[idx, :]

        if jitter_duplicates and jitter_duplicates > 0:
            x = x + jitter_duplicates * np.random.randn(x.size)
            y = y + jitter_duplicates * np.random.randn(y.size)

        tri = mtri.Triangulation(x, y)

        entry = dict(mu=mu_val, x=x, y=y, tri=tri,
                     B=B, A1=A1, Aa=Aa, Ab=Ab, Aa2=Aa2, Aab=Aab, Ab2=Ab2)

        if precompute:
            # Precompute projections for all displayed solutions:
            # psi_all:  (nshown, npts)
            # Hpsi_all: (nshown, npts)  (already combined with alpha/beta)
            Csel = np.real(C_full[:, sol_idx])  # (Nb, nshown)

            psi_all = (B @ Csel).T

            A1c  = (A1 @ Csel).T
            Aac  = (Aa @ Csel).T
            Abc  = (Ab @ Csel).T
            Aa2c = (Aa2 @ Csel).T
            Aabc = (Aab @ Csel).T
            Ab2c = (Ab2 @ Csel).T

            Hpsi_all = Hpsi_from_layers(A1c, Aac, Abc, Aa2c, Aabc, Ab2c)

            entry["psi_all"] = psi_all
            entry["Hpsi_all"] = Hpsi_all

        return entry

    # build cache
    if precompute:
        for mu in mu2_unique:
            ent = build_slice(mu)
            cache.append(ent)
    else:
        cache = [None] * mu2_unique.size

    # --- figure / axes (must be 3D axes) ---
    fig = plt.figure(figsize=(14, 7))
    ax_phi = fig.add_subplot(1, 2, 1, projection="3d")
    ax_eloc = fig.add_subplot(1, 2, 2, projection="3d")
    fig.subplots_adjust(bottom=0.30)

    # sliders
    ax_jmu  = fig.add_axes([0.15, 0.20, 0.7, 0.04])
    ax_i    = fig.add_axes([0.15, 0.14, 0.7, 0.04])
    ax_xmax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    ax_ymax = fig.add_axes([0.15, 0.04, 0.7, 0.03])

    jmu_slider = Slider(ax_jmu, "mu2 idx", 0, mu2_unique.size - 1, valinit=0, valstep=1)
    i_slider   = Slider(ax_i,   "i",       0, sol_idx.size - 1, valinit=0, valstep=1)

    # initial view limits from data
    xabs = np.nanmax(np.abs(x1_all)) if x1_all.size else 1.0
    yabs = np.nanmax(np.abs(y1_all)) if y1_all.size else 1.0
    xmax_slider = Slider(ax_xmax, "xmax", 0.1 * xabs, xabs, valinit=xabs, valstep=(xabs / 200))
    ymax_slider = Slider(ax_ymax, "ymax", 0.1 * yabs, yabs, valinit=yabs, valstep=(yabs / 200))

    def apply_xy_limits():
        xmax = float(xmax_slider.val)
        ymax = float(ymax_slider.val)
        for ax in (ax_phi, ax_eloc):
            ax.set_xlim(-xmax, xmax)
            ax.set_ylim(-ymax, ymax)

    def redraw(_=None):
        jmu = int(jmu_slider.val)
        ii  = int(i_slider.val)
        i_real = int(sol_idx[ii])

        mu_val = mu2_unique[jmu]

        # get/build slice
        ent = cache[jmu]
        if ent is None:
            ent = build_slice(mu_val)
            cache[jmu] = ent
        if ent is None:
            ax_phi.clear(); ax_eloc.clear()
            ax_phi.set_title(f"No points for mu2={mu_val:.6g}")
            fig.canvas.draw_idle()
            return

        tri = ent["tri"]
        x = ent["x"]; y = ent["y"]

        # compute psi/Eloc either from cache or on-demand
        if precompute and ("psi_all" in ent):
            psi = ent["psi_all"][ii, :]
            Hpsi = ent["Hpsi_all"][ii, :]
        else:
            c = np.real(C_full[:, i_real])
            B  = ent["B"]
            A1 = ent["A1"]; Aa = ent["Aa"]; Ab = ent["Ab"]
            Aa2 = ent["Aa2"]; Aab = ent["Aab"]; Ab2 = ent["Ab2"]

            psi = np.real(B @ c)

            A1c  = np.real(A1 @ c)
            Aac  = np.real(Aa @ c)
            Abc  = np.real(Ab @ c)
            Aa2c = np.real(Aa2 @ c)
            Aabc = np.real(Aab @ c)
            Ab2c = np.real(Ab2 @ c)

            Hpsi = Hpsi_from_layers(A1c, Aac, Abc, Aa2c, Aabc, Ab2c)

        denom = np.where(np.abs(psi) < eps, np.nan, psi)
        Eloc = Hpsi / denom

        # --- (d) crop the surface to the requested window ---
        xmax = float(xmax_slider.val)
        ymax = float(ymax_slider.val)

        win = (np.abs(x) <= xmax) & (np.abs(y) <= ymax)
        if np.count_nonzero(win) < 10:
            # too few points -> avoid triangulation crash
            ax_phi.clear();
            ax_eloc.clear()
            ax_phi.set_title("Too few points in window")
            fig.canvas.draw_idle()
            return

        x = x[win]
        y = y[win]
        psi = psi[win]
        Eloc = Eloc[win]
        tri = mtri.Triangulation(x, y)

        # keep the axes limits consistent with the crop
        for ax in (ax_phi, ax_eloc):
            ax.set_xlim(-xmax, xmax)
            ax.set_ylim(-ymax, ymax)

        # clear + plot surfaces
        ax_phi.clear()
        ax_eloc.clear()

        trisurf_colored(ax_phi, tri, psi, cmap_name="viridis")

        # Eloc surface colored by Eloc (optionally lock scale to zlim_eloc)
        if zlim_eloc is not None:
            trisurf_colored(ax_eloc, tri, Eloc, cmap_name="viridis", vmin=zlim_eloc[0], vmax=zlim_eloc[1])
            ax_eloc.set_zlim(zlim_eloc[0], zlim_eloc[1])
        else:
            trisurf_colored(ax_eloc, tri, Eloc, cmap_name="viridis")


        ax_phi.set_title(f"ψ(x1,y1) | i={i_real}, E={float(np.real(E[i_real])):.6g}, mu2={mu_val:.6g}")
        ax_phi.set_xlabel("x1"); ax_phi.set_ylabel("y1"); ax_phi.set_zlabel("ψ")

        ax_eloc.set_title(f"Eloc=Hψ/ψ | i={i_real}, alpha={alpha:.6g}, beta={beta:.6g}, mu2={mu_val:.6g}")
        ax_eloc.set_xlabel("x1"); ax_eloc.set_ylabel("y1"); ax_eloc.set_zlabel("Eloc")
        if zlim_eloc is not None:
            ax_eloc.set_zlim(zlim_eloc[0], zlim_eloc[1])

        # (d) apply xy limits (view only)
        apply_xy_limits()

        # electron-2 marker (orange) if mapping provided

        if (rAB is None) or (s2 is None) or (phi is None):
            # silently skip if not enough info
            pass
        else:
            mu = s2 / rAB

            x2 = 0.5 * rAB * mu * mu_val
            rho = 0.5 * rAB * np.sqrt((mu * mu - 1.0) * (1.0 - mu_val * mu_val))
            y2 = rho * np.cos(phi)
            z2 = rho * np.sin(phi)
            # print(x2, y2, z2)

            # Draw e2 position

            # ψ plot: bottom z
            zmin1, zmax1 = ax_phi.get_zlim()
            ax_phi.scatter([x2], [y2], [z2], c=["orange"], s=400, depthshade=False)
            # Eloc plot: bottom z
            zmin2, zmax2 = ax_eloc.get_zlim()
            ax_eloc.scatter([x2], [y2], [z2], c=["orange"], s=40, depthshade=False)


        fig.canvas.draw_idle()

    # slider callbacks
    jmu_slider.on_changed(redraw)
    i_slider.on_changed(redraw)
    xmax_slider.on_changed(redraw)
    ymax_slider.on_changed(redraw)

    # hotkeys
    def on_key(event):
        if event.key == "left":
            i_slider.set_val(max(0, int(i_slider.val) - 1))
        elif event.key == "right":
            i_slider.set_val(min(sol_idx.size - 1, int(i_slider.val) + 1))
        elif event.key == "down":
            jmu_slider.set_val(max(0, int(jmu_slider.val) - 1))
        elif event.key == "up":
            jmu_slider.set_val(min(mu2_unique.size - 1, int(jmu_slider.val) + 1))

    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()
    plt.show()



def plot_wavefn_and_local_energy(
    A_test, B_test, C, E,
    x2_vals, y2_vals, z2_vals,
    eps=1e-12,
    clip_percentiles=None,  # (1, 99)
):
    """
    Two 3D plots + two sliders (kz, i), with precomputation.
      - Left:  φ(x2,y2) = ψ on xy slice
      - Right: local energy Hψ/ψ on the same slice

    Only solutions with E[i] < 0 are precomputed and selectable.

    Shapes:
      A_test, B_test: (Npts, Nb)
      C: (nsol, Nb) or (Nb, nsol)   (auto-detected)
      E: (nsol,)
    """

    nx, ny, nz = len(x2_vals), len(y2_vals), len(z2_vals)
    Npts = nx * ny * nz

    assert A_test.shape[0] == Npts and B_test.shape[0] == Npts, "A/B rows must equal nx*ny*nz"
    Nb = A_test.shape[1]
    assert B_test.shape[1] == Nb, "A/B must have same number of basis columns"

    E = np.asarray(E).ravel()

    # Allow C to be either (nsol, Nb) or (Nb, nsol)
    if C.shape[1] == Nb:
        C_full = np.asarray(C)
    elif C.shape[0] == Nb:
        C_full = np.asarray(C).T
    else:
        raise ValueError(f"C shape {C.shape} incompatible with Nb={Nb}")

    nsol_full = C_full.shape[0]
    assert E.shape[0] == nsol_full, f"E has len {E.shape[0]} but C has {nsol_full} solutions"

    # Filter to E<0 only
    valid = np.where(E < 0)[0]
    if valid.size == 0:
        raise ValueError("No solutions with E[i] < 0 to display.")

    C_use = C_full[valid]
    E_use = E[valid]
    nsel = valid.size

    # Coordinates for the slice plot
    X, Y = np.meshgrid(x2_vals, y2_vals, indexing="ij")

    # ---- Precompute ψ and Hψ for all selectable i ----
    # Each i: psi_i_flat = A_test @ C_use[i]
    # Stack: psi_flat_all = (A_test @ C_use.T).T
    psi_flat_all  = (A_test @ C_use.T).T   # (nsel, Npts)
    Hpsi_flat_all = (B_test @ C_use.T).T   # (nsel, Npts)

    psi_all  = psi_flat_all.reshape(nsel, nx, ny, nz)
    Hpsi_all = Hpsi_flat_all.reshape(nsel, nx, ny, nz)

    # ---- Build figure with two 3D axes ----
    fig = plt.figure(figsize=(14, 7))
    ax_phi = fig.add_subplot(1, 2, 1, projection="3d")
    ax_eloc = fig.add_subplot(1, 2, 2, projection="3d")
    ax_eloc.set_zlim(-3.0, 0.0)
    fig.subplots_adjust(bottom=0.22)

    # Slider axes
    ax_kz = fig.add_axes([0.15, 0.11, 0.7, 0.04])
    ax_i  = fig.add_axes([0.15, 0.05, 0.7, 0.04])

    kz_slider = Slider(ax_kz, "kz", 0, nz - 1, valinit=0, valstep=1)
    i_slider  = Slider(ax_i,  "i (E<0)", 0, nsel - 1, valinit=0, valstep=1)

    surf_phi = None
    surf_eloc = None

    def compute_eloc(psi_xy, Hpsi_xy):
        Eloc = Hpsi_xy / np.where(np.abs(psi_xy) < eps, np.nan, psi_xy)
        if clip_percentiles is not None:
            lo, hi = np.nanpercentile(Eloc, clip_percentiles)
            Eloc = np.clip(Eloc, lo, hi)
        # else:
        #     Eloc = np.clip(Eloc, -3.0, 0.0)
        return Eloc

    def redraw(_=None):
        nonlocal surf_phi, surf_eloc
        kz = int(kz_slider.val)
        j  = int(i_slider.val)          # index in filtered list
        i0 = int(valid[j])              # original eigen-solution index
        Ei = float(E_use[j])

        psi_xy  = psi_all[j, :, :, kz]
        Hpsi_xy = Hpsi_all[j, :, :, kz]
        Eloc_xy = compute_eloc(psi_xy, Hpsi_xy)

        # Clear old surfaces
        if surf_phi is not None:
            surf_phi.remove()
        if surf_eloc is not None:
            surf_eloc.remove()

        # Left: φ = ψ
        surf_phi = ax_phi.plot_surface(X, Y, psi_xy, rstride=1, cstride=1, linewidth=0, antialiased=True)
        ax_phi.set_xlabel("x2")
        ax_phi.set_ylabel("y2")
        ax_phi.set_zlabel("ψ")
        ax_phi.set_title(f"ψ slice | i={j} (orig {i0}), E={Ei:.6g}, z2={z2_vals[kz]:.6g} (kz={kz})")

        # Right: local energy
        surf_eloc = ax_eloc.plot_surface(X, Y, Eloc_xy, rstride=1, cstride=1,
                                         linewidth=0, antialiased=True)
        ax_eloc.set_xlabel("x2")
        ax_eloc.set_ylabel("y2")
        ax_eloc.set_zlabel("Hψ/ψ")
        ax_eloc.set_title(f"Local energy | i={j} (orig {i0}), E={Ei:.6g}, z2={z2_vals[kz]:.6g} (kz={kz})")

        fig.canvas.draw_idle()

    kz_slider.on_changed(redraw)
    i_slider.on_changed(redraw)

    # Optional hotkeys: arrows for kz; pageup/pagedown for i
    def on_key(event):
        if event.key == "down":
            kz_slider.set_val(max(0, int(kz_slider.val) - 1))
        elif event.key == "up":
            kz_slider.set_val(min(nz - 1, int(kz_slider.val) + 1))
        elif event.key == "left":
            i_slider.set_val(max(0, int(i_slider.val) - 1))
        elif event.key == "right":
            i_slider.set_val(min(nsel - 1, int(i_slider.val) + 1))

    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()
    plt.show()