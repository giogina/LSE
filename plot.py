import gc
import os

import numpy as np
import matplotlib as mpl
# mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle
from glob import glob

from calc import calc_F_ee, calc_F_ne, calc_AB
from solver import solve_HS_from_layers, solve_HS, assemble_HS


def clustered_linspace(vmin, vmax, n, strength=2.5):
    # maps uniform u in [-1,1] to clustered x in [vmin,vmax]
    u = np.linspace(-1.0, 1.0, int(n))
    w = np.sinh(strength * u) / np.sinh(strength)  # still in [-1,1], denser near 0
    return 0.5*(vmin+vmax) + 0.5*(vmax-vmin)*w

def plot_Psi_Eloc_by_alpha_beta(
    *,

    # plot-domain definition (x1,y1 grid)
    x1_min=-4.0,
    x1_max=4.0,
    y1_min=-4.0,
    y1_max=4.0,
    nx1=140,
    ny1=140,

    # basis / physics inputs
    meta = None,

    # layer dictionaries for assembling S/H
    SH_layers=None,

    # fixed geometry needed for exp(-beta*rAB) and distance construction
    plot_rAB_target=None,

    # alpha/beta slider grids
    alpha_values=None,
    beta_values=None,

    # numerics / plot options
    only_negative_E=True,
    eps=1e-16,
    zlim_eloc=None,

    # color clarity: use discrete colormap levels
    psi_levels=128,
    eloc_levels=128,
):
    """
    Interactive 3D plotting on a regular (x1,y1) grid, with sliders:
      - alpha idx
      - beta idx
      - x2, y2, z2 (electron-2 position, each in [0,3])
      - solution i (over E<0 by default)

    Uses a colored surface on a regular mesh (no triangulation).
    """

    # Extract basis meta data
    coords = meta["coords"]
    basis_idx = meta["basis_idx"]
    delta = meta["delta"]
    M1M = meta["M1M"]
    M_inv = meta["M_inv"]
    Fij = meta["Fij"]
    Fji = meta["Fji"]
    X = meta["X"]

    # ---------------------------
    # Build x1,y1 grid for plotting (electron 1 in xy plane)
    # ---------------------------
    rAB = float(plot_rAB_target)
    x1_vals = clustered_linspace(x1_min, x1_max, nx1, strength=3.0)
    y1_vals = clustered_linspace(y1_min, y1_max, ny1, strength=3.5)

    X1, Y1 = np.meshgrid(x1_vals, y1_vals, indexing="xy")  # both (ny1, nx1)
    x1_flat = X1.ravel()
    y1_flat = Y1.ravel()
    P1 = x1_flat.size

    # derive rA1,rB1 -> s1,mu1 (arrays length P1)
    rA1 = np.sqrt((x1_flat + 0.5 * rAB) ** 2 + y1_flat ** 2)
    rB1 = np.sqrt((x1_flat - 0.5 * rAB) ** 2 + y1_flat ** 2)
    s1 = rA1 + rB1
    mu1 = (rA1 - rB1) / rAB

    # weights for plotting (no quadrature weighting)
    w1 = np.ones(P1, dtype=np.float64)
    shell_weight = 1.0

    alpha_values = np.asarray(alpha_values, dtype=float)
    beta_values = np.asarray(beta_values, dtype=float)

    # ---------------------------
    # Colored grid surface helper (DISCRETE color steps)
    # ---------------------------
    def surface_grid_colored_discrete(ax, X, Y, Z, cmap_name="viridis", vmin=None, vmax=None, nlevels=128):
        """
        Plot Z on a regular grid using plot_surface with per-cell facecolors.
        Uses a discrete BoundaryNorm for clearer color separation.
        """
        Z = np.asarray(Z, dtype=float)
        if vmin is None:
            vmin = float(np.nanmin(Z))
        if vmax is None:
            vmax = float(np.nanmax(Z))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = float(np.nanmin(Z[np.isfinite(Z)])) if np.any(np.isfinite(Z)) else 0.0
            vmax = vmin + 1.0

        # discrete bins
        bounds = np.linspace(vmin, vmax, int(nlevels) + 1)
        norm = mpl.colors.BoundaryNorm(bounds, ncolors=plt.get_cmap(cmap_name).N, clip=True)
        cmap = plt.get_cmap(cmap_name)

        fc = cmap(norm(Z))  # (ny,nx,4)
        nanmask = ~np.isfinite(Z)
        fc[nanmask, 3] = 0.0

        surf = ax.plot_surface(
            X, Y, Z,
            facecolors=fc,
            rstride=1, cstride=1,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )
        return surf, vmin, vmax

    # ---------------------------
    # Cache structure
    # ---------------------------
    eig_cache = {}   # (ia,ib) -> {"alpha","beta","E","C","sol_idx",...}
    geom_cache = {}  # (x2,y2,z2 rounded) -> {"B","A*","s_total","s2","x2","y2","z2"}

    def surface_grid_colored(ax, X, Y, Z, cmap_name, vmin=None, vmax=None, alpha=0.8):
        Z = np.asarray(Z)

        if vmin is None:
            vmin = np.nanmin(Z)
        if vmax is None:
            vmax = np.nanmax(Z)

        cmap = plt.get_cmap(cmap_name)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

        fc = cmap(norm(Z))
        fc[..., 3] *= alpha  # transparency
        fc[~np.isfinite(Z), 3] = 0.0  # NaNs transparent

        ax.plot_surface(
            X, Y, Z,
            facecolors=fc,
            rstride=1, cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
        )

    def get_cached_eigs(ia, ib):
        key = (int(ia), int(ib))
        if key in eig_cache:
            return eig_cache[key]

        alpha = float(alpha_values[ia])
        beta = float(beta_values[ib])

        E, C, cond = solve_HS_from_layers(SH_layers, alpha, beta, basis_idx, 1e-15, coords)
        idx = np.argsort(np.real(E))
        E = np.real(E[idx])
        C = np.real(C[:, idx])
        print(E[0])

        if only_negative_E:
            sol_idx = np.where(E < 0)[0]
            if sol_idx.size == 0:
                sol_idx = np.arange(E.size)
        else:
            sol_idx = np.arange(E.size)

        entry = {
            "alpha": alpha,
            "beta": beta,
            "condS": cond,
            "E": E,
            "C": C,
            "sol_idx": sol_idx,
        }
        eig_cache[key] = entry
        return entry

    def get_cached_geom(x2v, y2v, z2v):
        # round key so cache hits while dragging sliders
        key = (round(float(x2v), 3), round(float(y2v), 3), round(float(z2v), 3))
        if key in geom_cache:
            return geom_cache[key]

        x2 = np.array([key[0]], dtype=np.float64)
        y2 = np.array([key[1]], dtype=np.float64)
        z2 = np.array([key[2]], dtype=np.float64)

        # derive s2/mu2 as arrays (shape (1,))
        rA2 = np.sqrt((x2 + 0.5 * rAB) ** 2 + y2 ** 2 + z2 ** 2)
        rB2 = np.sqrt((x2 - 0.5 * rAB) ** 2 + y2 ** 2 + z2 ** 2)
        s2 = rA2 + rB2
        mu2 = (rA2 - rB2) / rAB

        # pointwise s_total over the electron-1 grid (shape (P1,))
        s = s1 + s2
        w2 = np.ones_like(x2)

        B, A1, Aa, Ab, Aa2, Aab, c_beta2, P = calc_AB(
            x1_flat, y1_flat,
            x2, y2, z2,
            rAB,
            s, s1, s2,  # s: (P1,), s1:(P1,), s2:(1,)
            mu1, mu2,  # mu1:(P1,), mu2:(1,)
            w1, w2,
            shell_weight,
            coords, basis_idx, delta, M1M, M_inv, Fij, Fji, X
        )

        Bee, Fee = calc_F_ee(x1_flat, y1_flat, rAB, coords, basis_idx, delta, X)
        Bne, Fne_1, Fne_alpha = calc_F_ne(x1_flat, y1_flat, rAB, coords, basis_idx, X)

        if B.shape[0] != P1:
            raise ValueError(
                f"calc_AB returned {B.shape[0]} rows, expected {P1}. Check s/mu broadcasting inside calc_AB.")

        Ab2 = c_beta2 * B

        entry = {
            "x2": key[0], "y2": key[1], "z2": key[2],
            "s2": float(s2[0]),
            "s_total": s,  # length P1
            "B": B,
            "A1": A1,
            "Aa": Aa,
            "Ab": Ab,
            "Aa2": Aa2,
            "Aab": Aab,
            "Ab2": Ab2,
            "Bee": Bee,
            "Fee": Fee,
            "Bne": Bne,
            "Fne_1": Fne_1,
            "Fne_alpha": Fne_alpha,
        }
        geom_cache[key] = entry
        return entry

    # ---------------------------
    # Figure + sliders
    # ---------------------------
    fig = plt.figure(figsize=(18, 8))
    ax_phi = fig.add_subplot(1, 3, 1, projection="3d")
    ax_eloc = fig.add_subplot(1, 3, 2, projection="3d")
    ax_cusp = fig.add_subplot(1, 3, 3, projection="3d")
    fig.subplots_adjust(bottom=0.34)

    ax_eloc.view_init(elev=0, azim=30)
    ax_cusp.view_init(elev=0, azim=90)

    # long sliders
    ax_alpha = fig.add_axes([0.15, 0.26, 0.7, 0.035])
    ax_beta  = fig.add_axes([0.15, 0.21, 0.7, 0.035])
    ax_i     = fig.add_axes([0.15, 0.16, 0.7, 0.035])

    # short side-by-side sliders for x2,y2,z2
    ax_x2 = fig.add_axes([0.15, 0.10, 0.22, 0.035])
    ax_y2 = fig.add_axes([0.41, 0.10, 0.22, 0.035])
    ax_z2 = fig.add_axes([0.67, 0.10, 0.22, 0.035])

    s_alpha = Slider(ax_alpha, "alpha idx", 0, len(alpha_values)-1, valinit=120, valstep=1)
    s_beta  = Slider(ax_beta,  "beta idx",  0, len(beta_values)-1,  valinit=0, valstep=1)

    # i slider max depends on alpha/beta; we rebuild bounds dynamically
    s_i = Slider(ax_i, "i", 0, 1, valinit=0, valstep=1)

    # x2,y2,z2 sliders in [0,3]
    s_x2 = Slider(ax_x2, "x2", 0.0, 3.0, valinit=0.4)
    s_y2 = Slider(ax_y2, "y2", 0.0, 3.0, valinit=0.4)
    s_z2 = Slider(ax_z2, "z2", 0.0, 3.0, valinit=0.4)

    def update_i_slider_max(n):
        nonlocal s_i
        ax_i.cla()
        s_i = Slider(ax_i, "i", 0, max(0, n-1), valinit=min(int(s_i.val), max(0, n-1)), valstep=1)
        s_i.on_changed(redraw)

    def redraw(_=None):
        ia = int(s_alpha.val)
        ib = int(s_beta.val)

        cache_entry = get_cached_eigs(ia, ib)
        alpha = cache_entry["alpha"]
        beta = cache_entry["beta"]
        E = cache_entry["E"]
        C = cache_entry["C"]
        sol_idx = cache_entry["sol_idx"]

        if int(s_i.val) > sol_idx.size - 1 or int(getattr(s_i, "valmax", 0)) != sol_idx.size - 1:
            update_i_slider_max(sol_idx.size)

        # geometry (x2,y2,z2)
        geom = get_cached_geom(s_x2.val, s_y2.val, s_z2.val)

        # projections for all shown solutions
        B   = geom["B"];   A1  = geom["A1"];  Aa = geom["Aa"];  Ab = geom["Ab"]
        Aa2 = geom["Aa2"]; Aab = geom["Aab"]; Ab2 = geom["Ab2"]
        s_total = geom["s_total"]  # length P1

        ii = int(s_i.val)
        i_real = int(sol_idx[ii])
        c = C[:, i_real]  # single eigenvector
        # print('+'.join([f"({float(cc / c[0])}) * r12^{basis_idx[i][1]} " for i, cc in enumerate(c) if (basis_idx[i][0]==0 and basis_idx[i][2]==0 and basis_idx[i][3]==0 and basis_idx[i][4]==0 and basis_idx[i][5]==0)]))
        print('+'.join([f"({float(cc / c[0])}) * rAB^{basis_idx[i][0]} " for i, cc in enumerate(c) if (basis_idx[i][1]==0 and basis_idx[i][2]==0 and basis_idx[i][3]==0 and basis_idx[i][4]==0 and basis_idx[i][5]==0)]))

        psi0 = B @ c
        A1c = A1 @ c
        Aac = Aa @ c
        Abc = Ab @ c
        Aa2c = Aa2 @ c
        Aabc = Aab @ c
        Ab2c = Ab2 @ c

        # pointwise exp factor
        if coords.endswith("_morse"):
            exps = np.exp(-alpha * s_total - beta * (rAB-1.4011)**2)
        else:
            exps = np.exp(-alpha * s_total - beta * rAB)

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

        # SSE quality metric
        E_i = float(E[i_real])
        diff = Eloc - E_i
        epsilon = float(np.nansum(diff * diff))

        psi = psi / (np.nanmax(np.abs(psi)) + 1e-300)
        if np.max(psi) < np.abs(np.min(psi)):
            psi *= -1.

        # reshape to grid (ny1,nx1)
        psi_grid  = psi.reshape(Y1.shape)
        Eloc_grid = Eloc.reshape(Y1.shape)

        # Cusp surfaces (if implemented)
        Bee = geom["Bee"]
        if len(np.shape(Bee)) > 0:
            Fee = geom["Fee"]
            Bee_c = Bee @ c
            Fee_c = Fee @ c
            denom_cusp = np.where(np.abs(Bee_c) < eps, np.nan, Bee_c)
            cusp_ee = Fee_c / denom_cusp - 0.5
            cusp_ee_grid = cusp_ee.reshape(Y1.shape)
            Bne = geom["Bne"]
            Fne_1 = geom["Fne_1"]
            Fne_alpha = geom["Fne_alpha"]
            Fne = Fne_1 + Fne_alpha * alpha
            Bne_c = Bne @ c
            Fne_c = Fne @ c
            denom_cusp = np.where(np.abs(Bne_c) < eps, np.nan, Bne_c)
            cusp_ne = Fne_c / denom_cusp + 1.0
            cusp_ne_grid = cusp_ne.reshape(Y1.shape)
            ax_cusp.clear()

            xmin, xmax = -3., 3.
            ymin, ymax = 0.0, 1.

            Zee = cusp_ee_grid.copy()
            Zne = cusp_ne_grid.copy()

            mask = (X1 < xmin) | (X1 > xmax) | (Y1 < ymin) | (Y1 > ymax)

            Zee[mask] = np.nan
            Zne[mask] = np.nan

            # electron–electron cusp (yellow/orange palette)
            surface_grid_colored(
                ax_cusp, X1, Y1, Zee,
                cmap_name="plasma",
                vmin=-2.0, vmax=2.0,
                alpha=0.5
            )

            # electron–nucleus cusp (blue/green palette)
            surface_grid_colored(
                ax_cusp, X1, Y1, Zne,
                cmap_name="viridis",
                vmin=-2.0, vmax=2.0,
                alpha=0.5
            )

            # ax_cusp.set_zlim(-2.0, 2.0)

        ax_phi.clear()
        ax_eloc.clear()

        surface_grid_colored_discrete(ax_phi, X1, Y1, psi_grid, cmap_name="viridis", nlevels=int(psi_levels))

        if zlim_eloc is not None:
            surface_grid_colored_discrete(ax_eloc, X1, Y1, Eloc_grid, cmap_name="viridis",
                                          vmin=float(zlim_eloc[0]), vmax=float(zlim_eloc[1]),
                                          nlevels=int(eloc_levels))
            ax_eloc.set_zlim(float(zlim_eloc[0]), float(zlim_eloc[1]))
        else:
            surface_grid_colored_discrete(ax_eloc, X1, Y1, Eloc_grid, cmap_name="viridis", nlevels=int(eloc_levels))

        # mark electron 2 position (orange dot), place at top z so it stays visible
        zmax1 = ax_phi.get_zlim()[1]
        zmax2 = ax_eloc.get_zlim()[1]
        ax_phi.scatter([geom["x2"]], [geom["y2"]], [zmax1], c=["orange"], s=160, depthshade=False)
        ax_eloc.scatter([geom["x2"]], [geom["y2"]], [zmax2], c=["orange"], s=160, depthshade=False)
        ax_cusp.set_xlim(-3.0, 3.0)
        ax_cusp.set_ylim(0.0, 3.0)
        ax_cusp.set_zlim(-0.03, 0.03)

        ax_phi.set_title(
            f"ψ | alpha={alpha:.6f} beta={beta:.6f}\n"  #  cond(S)={cache_entry['condS']:.3e}
            f"i={i_real}, E={E[i_real]:.10f}\n"
            f"Electron 2 fixed at: x2,y2,z2=({geom['x2']:.3f},{geom['y2']:.3f},{geom['z2']:.3f})"
        )
        ax_eloc.set_title(
            f"Local energy | epsilon={epsilon:.10e}"
        )
        ax_cusp.set_title(
            f"Cusp functions (should be 0)\n"
            f"Red: e-e, Green: e-n"
        )

        ax_phi.set_xlabel("x1"); ax_phi.set_ylabel("y1"); ax_phi.set_zlabel("ψ")
        ax_eloc.set_xlabel("x1"); ax_eloc.set_ylabel("y1"); ax_eloc.set_zlabel("Eloc")

        fig.canvas.draw_idle()

    # wire sliders
    s_alpha.on_changed(redraw)
    s_beta.on_changed(redraw)
    s_i.on_changed(redraw)
    s_x2.on_changed(redraw)
    s_y2.on_changed(redraw)
    s_z2.on_changed(redraw)

    def on_key(event):
        if event.key == "left":
            s_alpha.set_val(max(0, int(s_alpha.val) - 1))
        elif event.key == "right":
            s_alpha.set_val(min(len(alpha_values) - 1, int(s_alpha.val) + 1))
        elif event.key == "pageup":
            s_beta.set_val(min(len(beta_values) - 1, int(s_beta.val) + 1))
        elif event.key == "pagedown":
            s_beta.set_val(max(0, int(s_beta.val) - 1))

    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()
    plt.show(block=True)

def plot_nonBO_from_files(
    file,
    alpha,
    beta,
    *,
    # plot-domain definition (x1,y1 grid)
    x1_min=-4.0,
    x1_max=4.0,
    y1_min=-4.0,
    y1_max=4.0,
    nx1=140,
    ny1=140,

    # fixed geometry needed for exp(-beta*rAB) and distance construction
    plot_rAB_target=None,

    # numerics / plot options
    only_negative_E=True,
    eps=1e-16,
    zlim_eloc=None,

    # color clarity: use discrete colormap levels
    psi_levels=128,
    eloc_levels=128,

    # initial slider values
    x2_init=0.4,
    y2_init=0.4,
    z2_init=0.4,

    rcond = 1e-17,
    savepic = None # file name for saving the plot instead
):
    """
    Stream-read SHlayers*.pkl files (supports '*' glob), assemble total SH_layers,
    solve at fixed (alpha,beta), and plot like plot_Psi_Eloc_by_alpha_beta
    but WITHOUT alpha/beta sliders.
    """

    print("Start file stream...")
    # ---------------------------
    # Stream-read and accumulate components
    # ---------------------------
    if "*" in file:
        files = sorted(glob(file))
        if not files:
            raise FileNotFoundError(f"No files matched glob: {file}")
    else:
        files = [file]

    meta = None
    S = None
    H = None

    for ff in files:
        print(ff)
        with open(ff, "rb") as f:
            layers_new = pickle.load(f)

        if meta is None:
            if "meta" not in layers_new:
                raise KeyError(f"'meta' not found in {ff}")
            meta = layers_new["meta"]
            S = np.zeros_like(next(iter(layers_new["S"].values())), dtype=np.float64)
            H = np.zeros_like(S, dtype=np.float64)

        H, S = assemble_HS(layers_new, alpha, beta, H=H, S=S, coords=meta["coords"])  # Assemble matrices in-place

        del layers_new
        gc.collect()


    # ------------------
    #  basis meta data
    # ------------------
    coords = meta["coords"]
    basis_idx = meta["basis_idx"]
    delta = meta["delta"]
    M1M = meta["M1M"]
    M_inv = meta["M_inv"]
    Fij = meta["Fij"]
    Fji = meta["Fji"]
    X = meta["X"]

    # ---------------------------
    # Build x1,y1 grid for plotting (electron 1 in xy plane)
    # ---------------------------
    if plot_rAB_target is None:
        raise ValueError("plot_rAB_target must be provided (fixed rAB for plotting).")
    rAB = float(plot_rAB_target)

    # use the same clustered grid helper already in this module
    x1_vals = clustered_linspace(x1_min, x1_max, nx1, strength=3.0)
    y1_vals = clustered_linspace(y1_min, y1_max, ny1, strength=3.5)

    X1, Y1 = np.meshgrid(x1_vals, y1_vals, indexing="xy")
    x1_flat = X1.ravel()
    y1_flat = Y1.ravel()
    P1 = x1_flat.size

    rA1 = np.sqrt((x1_flat + 0.5 * rAB) ** 2 + y1_flat ** 2)
    rB1 = np.sqrt((x1_flat - 0.5 * rAB) ** 2 + y1_flat ** 2)
    s1 = rA1 + rB1
    mu1 = (rA1 - rB1) / rAB

    w1 = np.ones(P1, dtype=np.float64)
    shell_weight = 1.0

    # ------------------
    #  plotting helpers
    # ------------------
    def surface_grid_colored_discrete(ax, Xg, Yg, Zg, cmap_name="viridis", vmin=None, vmax=None, nlevels=128):
        Zg = np.asarray(Zg, dtype=float)
        if vmin is None:
            vmin = float(np.nanmin(Zg))
        if vmax is None:
            vmax = float(np.nanmax(Zg))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            finite = Zg[np.isfinite(Zg)]
            vmin = float(np.nanmin(finite)) if finite.size else 0.0
            vmax = vmin + 1.0

        bounds = np.linspace(vmin, vmax, int(nlevels) + 1)
        norm = mpl.colors.BoundaryNorm(bounds, ncolors=plt.get_cmap(cmap_name).N, clip=True)
        cmap = plt.get_cmap(cmap_name)

        fc = cmap(norm(Zg))
        nanmask = ~np.isfinite(Zg)
        fc[nanmask, 3] = 0.0

        ax.plot_surface(
            Xg, Yg, Zg,
            facecolors=fc,
            rstride=1, cstride=1,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )

    def surface_grid_colored(ax, Xg, Yg, Zg, cmap_name, vmin=None, vmax=None, alpha=0.8):
        Zg = np.asarray(Zg, dtype=float)
        if vmin is None:
            vmin = float(np.nanmin(Zg))
        if vmax is None:
            vmax = float(np.nanmax(Zg))

        cmap = plt.get_cmap(cmap_name)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        fc = cmap(norm(Zg))
        fc[..., 3] *= alpha
        fc[~np.isfinite(Zg), 3] = 0.0

        ax.plot_surface(
            Xg, Yg, Zg,
            facecolors=fc,
            rstride=1, cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
        )

    # -------
    #  Solve
    # -------

    E, C, cond = solve_HS(H, S, rcond)
    idx = np.argsort(np.real(E))
    E = np.real(E[idx])
    C = np.real(C[:, idx])

    if only_negative_E:
        sol_idx = np.where(E < 0)[0]
        if sol_idx.size == 0:
            sol_idx = np.arange(E.size)
    else:
        sol_idx = np.arange(E.size)

    # ---------------------------
    # Geometry cache (x2,y2,z2 -> B/A* etc)
    # ---------------------------
    geom_cache = {}

    def get_cached_geom(x2v, y2v, z2v):
        key = (round(float(x2v), 3), round(float(y2v), 3), round(float(z2v), 3))
        if key in geom_cache:
            return geom_cache[key]

        x2 = np.array([key[0]], dtype=np.float64)
        y2 = np.array([key[1]], dtype=np.float64)
        z2 = np.array([key[2]], dtype=np.float64)

        rA2 = np.sqrt((x2 + 0.5 * rAB) ** 2 + y2 ** 2 + z2 ** 2)
        rB2 = np.sqrt((x2 - 0.5 * rAB) ** 2 + y2 ** 2 + z2 ** 2)
        s2 = rA2 + rB2
        mu2 = (rA2 - rB2) / rAB

        s_total = s1 + s2
        w2 = np.ones_like(x2)

        B, A1, Aa, Ab, Aa2, Aab, c_beta2, P = calc_AB(
            x1_flat, y1_flat,
            x2, y2, z2,
            rAB,
            s_total, s1, s2,
            mu1, mu2,
            w1, w2,
            shell_weight,
            coords, basis_idx, delta, M1M, M_inv, Fij, Fji, X
        )

        Bee, Fee = calc_F_ee(x1_flat, y1_flat, rAB, coords, basis_idx, delta, X)
        Bne, Fne_1, Fne_alpha = calc_F_ne(x1_flat, y1_flat, rAB, coords, basis_idx, X)

        if B.shape[0] != P1:
            raise ValueError(f"calc_AB returned {B.shape[0]} rows, expected {P1}.")

        Ab2 = c_beta2 * B

        entry = {
            "x2": key[0], "y2": key[1], "z2": key[2],
            "s_total": s_total,
            "B": B, "A1": A1, "Aa": Aa, "Ab": Ab, "Aa2": Aa2, "Aab": Aab, "Ab2": Ab2,
            "Bee": Bee, "Fee": Fee,
            "Bne": Bne, "Fne_1": Fne_1, "Fne_alpha": Fne_alpha,
        }
        geom_cache[key] = entry
        return entry

    # ---------------------------
    # Figure + sliders (i, x2,y2,z2 only)
    # ---------------------------
    fig = plt.figure(figsize=(18, 8))
    ax_phi = fig.add_subplot(1, 3, 1, projection="3d")
    ax_eloc = fig.add_subplot(1, 3, 2, projection="3d")
    ax_cusp = fig.add_subplot(1, 3, 3, projection="3d")
    fig.subplots_adjust(bottom=0.26)

    ax_eloc.view_init(elev=0, azim=30)
    ax_cusp.view_init(elev=0, azim=90)

    # ax_i  = fig.add_axes([0.15, 0.18, 0.7, 0.035])
    ax_x2 = fig.add_axes([0.15, 0.10, 0.22, 0.035])
    ax_y2 = fig.add_axes([0.41, 0.10, 0.22, 0.035])
    ax_z2 = fig.add_axes([0.67, 0.10, 0.22, 0.035])

    # s_i  = Slider(ax_i, "i", 0, max(0, sol_idx.size - 1), valinit=0, valstep=1)
    s_x2 = Slider(ax_x2, "x2", 0.0, 3.0, valinit=float(x2_init))
    s_y2 = Slider(ax_y2, "y2", 0.0, 3.0, valinit=float(y2_init))
    s_z2 = Slider(ax_z2, "z2", 0.0, 3.0, valinit=float(z2_init))

    def redraw(_=None):
        # ii = int(s_i.val)
        ii = 0
        i_real = int(sol_idx[ii])
        c = C[:, i_real]
        print("("+'+'.join([f"({float(cc / c[0])}) * rAB^{basis_idx[i][0]} " for i, cc in enumerate(c) if (basis_idx[i][1]==0 and basis_idx[i][2]==0 and basis_idx[i][3]==0 and basis_idx[i][4]==0 and basis_idx[i][5]==0)])+f")*exp(-{beta}*(rAB-1.4011)^2)")
        print("("+'+'.join([f"({float(cc / c[0])}) * rAB^{basis_idx[i][0]} " for i, cc in enumerate(c) if (basis_idx[i][1]==1 and basis_idx[i][2]==0 and basis_idx[i][3]==0 and basis_idx[i][4]==0 and basis_idx[i][5]==0)])+f")*exp(-{beta}*(rAB-1.4011)^2)")
        print("("+'+'.join([f"({float(cc / c[0])}) * rAB^{basis_idx[i][0]} " for i, cc in enumerate(c) if (basis_idx[i][1]==0 and basis_idx[i][2]==1 and basis_idx[i][3]==0 and basis_idx[i][4]==0 and basis_idx[i][5]==0)])+f")*exp(-{beta}*(rAB-1.4011)^2)")
        print("("+'+'.join([f"({float(cc / c[0])}) * rAB^{basis_idx[i][0]} " for i, cc in enumerate(c) if (basis_idx[i][1]==0 and basis_idx[i][2]==0 and basis_idx[i][3]==0 and basis_idx[i][4]==1 and basis_idx[i][5]==1)])+f")*exp(-{beta}*(rAB-1.4011)^2)")

        geom = get_cached_geom(s_x2.val, s_y2.val, s_z2.val)

        B   = geom["B"];   A1  = geom["A1"];  Aa = geom["Aa"];  Ab = geom["Ab"]
        Aa2 = geom["Aa2"]; Aab = geom["Aab"]; Ab2 = geom["Ab2"]
        s_total = geom["s_total"]

        psi0 = B @ c
        A1c = A1 @ c
        Aac = Aa @ c
        Abc = Ab @ c
        Aa2c = Aa2 @ c
        Aabc = Aab @ c
        Ab2c = Ab2 @ c

        if coords.endswith("_morse"):
            exps = np.exp(-alpha * s_total - beta * (rAB - 1.4011) ** 2)
        else:
            exps = np.exp(-alpha * s_total - beta * rAB)

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

        E_i = float(E[i_real])
        diff = Eloc - E_i
        epsilon = float(np.nansum(diff * diff))

        psi = psi / (np.nanmax(np.abs(psi)) + 1e-300)
        if np.nanmax(psi) < np.abs(np.nanmin(psi)):
            psi *= -1.0

        psi_grid  = psi.reshape(Y1.shape)
        Eloc_grid = Eloc.reshape(Y1.shape)

        # cusps
        ax_cusp.clear()
        Bee = geom["Bee"]
        if len(np.shape(Bee)) > 0:
            Fee = geom["Fee"]
            Bee_c = Bee @ c
            Fee_c = Fee @ c
            denom_cusp = np.where(np.abs(Bee_c) < eps, np.nan, Bee_c)
            cusp_ee = Fee_c / denom_cusp - 0.5
            cusp_ee_grid = cusp_ee.reshape(Y1.shape)

            Bne = geom["Bne"]
            Fne_1 = geom["Fne_1"]
            Fne_alpha = geom["Fne_alpha"]
            Fne = Fne_1 + Fne_alpha * alpha
            Bne_c = Bne @ c
            Fne_c = Fne @ c
            denom_cusp = np.where(np.abs(Bne_c) < eps, np.nan, Bne_c)
            cusp_ne = Fne_c / denom_cusp + 1.0
            cusp_ne_grid = cusp_ne.reshape(Y1.shape)

            Zee = cusp_ee_grid.copy()
            Zne = cusp_ne_grid.copy()

            # same windowing logic as before
            xmin, xmax = -3.0, 3.0
            ymin, ymax = 0.0, 1.0
            mask = (X1 < xmin) | (X1 > xmax) | (Y1 < ymin) | (Y1 > ymax)
            Zee[mask] = np.nan
            Zne[mask] = np.nan

            surface_grid_colored(ax_cusp, X1, Y1, Zee, cmap_name="plasma", vmin=-2.0, vmax=2.0, alpha=0.5)
            surface_grid_colored(ax_cusp, X1, Y1, Zne, cmap_name="viridis", vmin=-2.0, vmax=2.0, alpha=0.5)

            ax_cusp.set_xlim(-3.0, 3.0)
            ax_cusp.set_ylim(0.0, 3.0)
            ax_cusp.set_zlim(-0.05, 0.05)

        ax_phi.clear()
        ax_eloc.clear()

        surface_grid_colored_discrete(ax_phi, X1, Y1, psi_grid, cmap_name="viridis", nlevels=int(psi_levels))

        if zlim_eloc is not None:
            surface_grid_colored_discrete(
                ax_eloc, X1, Y1, Eloc_grid, cmap_name="viridis",
                vmin=float(zlim_eloc[0]), vmax=float(zlim_eloc[1]),
                nlevels=int(eloc_levels)
            )
            ax_eloc.set_zlim(float(zlim_eloc[0]), float(zlim_eloc[1]))
        else:
            surface_grid_colored_discrete(ax_eloc, X1, Y1, Eloc_grid, cmap_name="viridis", nlevels=int(eloc_levels))

        zmax1 = ax_phi.get_zlim()[1]
        zmax2 = ax_eloc.get_zlim()[1]
        ax_phi.scatter([geom["x2"]], [geom["y2"]], [zmax1], c=["orange"], s=160, depthshade=False)
        ax_eloc.scatter([geom["x2"]], [geom["y2"]], [zmax2], c=["orange"], s=160, depthshade=False)

        ax_phi.set_title(
            f"ψ | alpha={alpha:.6f} beta={beta:.6f}\n"  # cond(S)={cache_entry['condS']:.3e}
            f"E={E[i_real]:.10f}\n"
            f"Electron 2 fixed at: x2,y2,z2=({geom['x2']:.3f},{geom['y2']:.3f},{geom['z2']:.3f})"
        )
        ax_eloc.set_title(
            f"Local energy | epsilon={epsilon:.10e}"
        )
        ax_cusp.set_title(
            f"Cusp functions (should be 0)\n"
            f"Red: e-e, Green: e-n"
        )

        ax_phi.set_xlabel("x1"); ax_phi.set_ylabel("y1"); ax_phi.set_zlabel("ψ")
        ax_eloc.set_xlabel("x1"); ax_eloc.set_ylabel("y1"); ax_eloc.set_zlabel("Eloc")

        fig.canvas.draw_idle()

    # s_i.on_changed(redraw)
    s_x2.on_changed(redraw)
    s_y2.on_changed(redraw)
    s_z2.on_changed(redraw)

    redraw()

    if savepic == None:
        plt.show(block=True)
    else:
        fig.savefig(os.path.expanduser(savepic.replace("Eeee", f"E{E[0]}")), dpi=150, bbox_inches="tight")
        plt.close(fig)




