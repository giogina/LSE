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
from solver import solve_HS_from_layers, solve_HS
from layers import assemble_HS, assemble_HS_multi_alpha


def clustered_linspace(vmin, vmax, n, strength=2.5):
    # maps uniform u in [-1,1] to clustered x in [vmin,vmax]
    u = np.linspace(-1.0, 1.0, int(n))
    w = np.sinh(strength * u) / np.sinh(strength)  # still in [-1,1], denser near 0
    return 0.5*(vmin+vmax) + 0.5*(vmax-vmin)*w

def eval_psi_eloc_on_points(
    *,
    rAB_vals, x1_vals, y1_vals,
    x2v, y2v, z2v,
    c_full, frankenBasis, meta,
    eps,
    psi_scale
):
    """
    Evaluate psi and Eloc on a 2D grid defined by (rAB_vals, x1_vals) with y1 fixed.
    Returns: (RAB_grid, X1_grid, psi_grid, eloc_grid)

    Shapes:
      RAB_grid, X1_grid, psi_grid, eloc_grid are (nrAB, nx1)
    """
    rAB_vals = np.asarray(rAB_vals, dtype=np.float64)
    x1_vals  = np.asarray(x1_vals,  dtype=np.float64)
    y1_vals  = np.asarray(y1_vals,  dtype=np.float64)
    if y1_vals.size == 1:
        y1_vals = np.full_like(x1_vals, float(y1_vals[0]), dtype=np.float64)
    if y1_vals.shape != x1_vals.shape:
        raise ValueError("y1_vals must be same shape as x1_vals (or scalar/len=1).")

    nx = x1_vals.size
    nr = rAB_vals.size

    coords    = meta["coords"]
    basis_idx = meta["basis_idx"]
    delta     = meta["delta"]
    M1M       = meta["M1M"]
    M_inv     = meta["M_inv"]
    Fij       = meta["Fij"]
    Fji       = meta["Fji"]
    Xmeta     = meta["X"]

    # electron 2 fixed (lab coords), but distances depend on rAB
    x2 = np.array([float(x2v)], dtype=np.float64)
    y2 = np.array([float(y2v)], dtype=np.float64)
    z2 = np.array([float(z2v)], dtype=np.float64)
    w2 = np.ones_like(x2)
    shell_weight = 1.0

    # output grids (nr, nx)
    psi_grid  = np.full((nr, nx), np.nan, dtype=np.float64)
    eloc_grid = np.full((nr, nx), np.nan, dtype=np.float64)

    N = frankenBasis.N
    nblocks = len(frankenBasis.blocks)
    assert c_full.shape[0] == nblocks * N

    # loop rAB -> build 1D line operators -> project -> accumulate blocks
    for ir, rAB_local in enumerate(rAB_vals):
        # electron 1 line points
        x1_flat = x1_vals
        y1_flat = y1_vals

        too_close = (np.abs(x1_flat - (-0.5 * rAB_local)) < 1e-3) | (np.abs(x1_flat - (0.5 * rAB_local)) < 1e-3)

        rA1 = np.sqrt((x1_flat + 0.5 * rAB_local) ** 2 + y1_flat ** 2)
        rB1 = np.sqrt((x1_flat - 0.5 * rAB_local) ** 2 + y1_flat ** 2)
        s1  = rA1 + rB1
        mu1 = (rA1 - rB1) / rAB_local
        w1  = np.ones(nx, dtype=np.float64)

        # electron 2 distances for this rAB
        rA2 = np.sqrt((x2 + 0.5 * rAB_local) ** 2 + y2 ** 2 + z2 ** 2)
        rB2 = np.sqrt((x2 - 0.5 * rAB_local) ** 2 + y2 ** 2 + z2 ** 2)
        s2  = rA2 + rB2
        mu2 = (rA2 - rB2) / rAB_local

        s_total = s1 + s2  # (nx,)
        B, A1, Aa, Ab, Aa2, Aab, c_beta2, _P = calc_AB(
            x1_flat, y1_flat,
            x2, y2, z2,
            float(rAB_local),
            s_total, s1, s2,
            mu1, mu2,
            w1, w2,
            shell_weight,
            coords, basis_idx, delta, M1M, M_inv, Fij, Fji, Xmeta
        )
        Ab2 = c_beta2 * B

        # accumulators along x1
        psi  = np.zeros(nx, dtype=np.float64)
        Hpsi = np.zeros(nx, dtype=np.float64)

        for _k, blk, sl in frankenBasis.iter_blocks():
            alpha_k = float(blk["alpha"])
            beta_k  = float(blk["beta"])
            Rm_k    = blk.get("Rm", None)
            ck = c_full[sl]

            psi0_k = B   @ ck
            A1c_k  = A1  @ ck
            Aac_k  = Aa  @ ck
            Abc_k  = Ab  @ ck
            Aa2c_k = Aa2 @ ck
            Aabc_k = Aab @ ck
            Ab2c_k = Ab2 @ ck

            if coords.endswith("_morse"):
                exps_k = np.exp(-alpha_k * s_total - beta_k * (float(Rm_k) - float(rAB_local)) ** 2)
                rm = (float(rAB_local) - float(Rm_k))
                Hpsi += exps_k * (
                    A1c_k
                    + alpha_k * Aac_k
                    + beta_k * rm * Abc_k
                    + (alpha_k ** 2) * Aa2c_k
                    + (alpha_k * beta_k * rm) * Aabc_k
                    + ((beta_k * rm) ** 2) * Ab2c_k
                    + (2.0 * meta["M_inv"] * beta_k) * psi0_k
                )
            else:
                exps_k = np.exp(-alpha_k * s_total - beta_k * float(rAB_local))
                Hpsi += exps_k * (
                    A1c_k
                    + alpha_k * Aac_k
                    + beta_k * Abc_k
                    + (alpha_k ** 2) * Aa2c_k
                    + (alpha_k * beta_k) * Aabc_k
                    + (beta_k ** 2) * Ab2c_k
                )

            psi += exps_k * psi0_k

        denom = np.where(np.abs(psi) < eps, np.nan, psi)
        Eloc  = Hpsi / denom

        Eloc[too_close] = np.nan

        # fixed scaling: keeps true rAB dependence
        psi_grid[ir, :] = psi_scale * psi
        eloc_grid[ir, :] = Eloc

    RAB_grid, X1_grid = np.meshgrid(rAB_vals, x1_vals, indexing="ij")  # (nr,nx)
    return RAB_grid, X1_grid, psi_grid, eloc_grid


def plot_Psi_Eloc_multi_params_from_files(
    file,
    *,
    # fixed geometry needed for exp(-beta*rAB) and distance construction
    plot_rAB_target = 1.4,

    # plot-domain definition (x1,y1 grid)
    xy_max = 4.0,
    nx1=40,

    # rAB-x1 surface plot domain
    rAB_surf_min=None,  # if None: plot_rAB_target - 0.7
    rAB_surf_max=None,  # if None: plot_rAB_target + 0.7
    y1_line_fixed=0.0,  # y1 fixed for the rAB-x1 surface

    # numerics / plot options
    only_negative_E=True,
    eps=1e-12,
    zlim_eloc=None,
    psi_levels=128,
    eloc_levels=128,
    rcond=1e-17,

    # initial electron-2 position
    x2_init=0.4,
    y2_init=0.4,
    z2_init=0.4,

    # initial parameter text
    alpha_text_init="0.2, 0.5, 0.8, 1.1",  # 0.2,
    # beta_rm_text_init="7.6 1.2; 7.8 1.3; 8.0 1.4",
    # alpha_text_init="0.8",
    beta_rm_text_init="8.0 1.4",
):
    """
    Like plot_Psi_Eloc_by_alpha_beta, but:
      - NO alpha/beta sliders.
      - You enter sets of alphas and (beta,Rm) pairs in text boxes.
      - Click Apply -> rebuild H,S via assemble_HS_multi_alpha, solve, and keep plotting.

    Text formats:
      alphas:
        - list: "0.6,0.8,1.0"
        - range: "0.6:1.2:7"  meaning linspace(min,max,n)
      beta Rm pairs:
        - "0.5 1.4011; 1.0 1.4100"
        - or one pair per line.
    """
    import gc
    import pickle
    from glob import glob

    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, TextBox, Button

    from calc import calc_F_ee, calc_F_ne, calc_AB
    from solver import solve_HS
    from layers import assemble_HS_multi_alpha

    nrAB=nx1  # rAB resolution for the surface
    nx1_line=nx1  # x1 resolution for the surface (y1 fixed to 0)
    ny1=nx1
    # ---------------------------
    # helpers
    # ---------------------------

    def parse_list_or_range(s: str):
        s = (s or "").strip()
        if not s:
            return np.array([], float)

        # range form: "a:b:n"
        if ":" in s and "," not in s:
            a, b, n = [x.strip() for x in s.split(":")]
            return np.linspace(float(a), float(b), int(n), dtype=float)

        # list form: "a,b,c"
        return np.array([float(x) for x in s.split(",") if x.strip() != ""], dtype=float)

    def parse_beta_rm_pairs(s: str):
        s = (s or "").strip()
        if not s:
            return np.empty((0, 2), float)
        parts = []
        for line in s.replace(";", "\n").splitlines():
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) != 2:
                raise ValueError(f"Bad beta/Rm line: '{line}'. Expected: '<beta> <Rm>'")
            b, rm = toks
            parts.append((float(b), float(rm)))
        return np.array(parts, dtype=float)

    def surface_grid_colored_discrete(ax, X, Y, Z, cmap_name="viridis", vmin=None, vmax=None, nlevels=128):
        Z = np.asarray(Z, dtype=float)
        if vmin is None:
            vmin = float(np.nanmin(Z))
        if vmax is None:
            vmax = float(np.nanmax(Z))
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
            vmin = float(np.nanmin(Z[np.isfinite(Z)])) if np.any(np.isfinite(Z)) else 0.0
            vmax = vmin + 1.0

        bounds = np.linspace(vmin, vmax, int(nlevels) + 1)
        norm = mpl.colors.BoundaryNorm(bounds, ncolors=plt.get_cmap(cmap_name).N, clip=True)
        cmap = plt.get_cmap(cmap_name)

        fc = cmap(norm(Z))
        nanmask = ~np.isfinite(Z)
        fc[nanmask, 3] = 0.0

        ax.plot_surface(
            X, Y, Z,
            facecolors=fc,
            rstride=1, cstride=1,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )

    def surface_grid_colored(ax, X, Y, Z, cmap_name, vmin=None, vmax=None, alpha=0.6):
        Z = np.asarray(Z, dtype=float)
        if vmin is None:
            vmin = float(np.nanmin(Z))
        if vmax is None:
            vmax = float(np.nanmax(Z))
        cmap = plt.get_cmap(cmap_name)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        fc = cmap(norm(Z))
        fc[..., 3] *= float(alpha)
        fc[~np.isfinite(Z), 3] = 0.0

        ax.plot_surface(
            X, Y, Z,
            facecolors=fc,
            rstride=1, cstride=1,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )

    # ---------------------------
    # load files list
    # ---------------------------
    if "*" in file:
        files = sorted(glob(file))
        if not files:
            raise FileNotFoundError(f"No files matched glob: {file}")
    else:
        files = [file]

    # ---------------------------
    # fixed plot grid (electron 1)
    # ---------------------------
    rAB = float(plot_rAB_target)

    x_extra = np.array([+(rAB / 2 + 0.01), -(rAB / 2 + 0.01)], dtype=float)  # include points close to the nuclei
    y_extra = np.array([0.0], dtype=float)
    x1_vals = clustered_linspace(-xy_max, xy_max, nx1, strength=3.0)
    y1_vals = clustered_linspace(-xy_max, xy_max, ny1, strength=3.5)
    x1_vals = np.unique(np.sort(np.concatenate([x1_vals, x_extra])))
    y1_vals = np.unique(np.sort(np.concatenate([y1_vals, y_extra])))

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

    # ---------------------------
    # state (rebuilt on Apply)
    # ---------------------------
    meta = None
    H = None
    S = None
    frankenBasis = None
    E = None
    C = None
    sol_idx = None

    geom_cache = {}  # keyed by rounded (x2,y2,z2)

    def reset_geom_cache():
        geom_cache.clear()

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

        coords = meta["coords"]
        basis_idx = meta["basis_idx"]
        delta = meta["delta"]
        M1M = meta["M1M"]
        M_inv = meta["M_inv"]
        Fij = meta["Fij"]
        Fji = meta["Fji"]
        X = meta["X"]

        B, A1, Aa, Ab, Aa2, Aab, c_beta2, _P = calc_AB(
            x1_flat, y1_flat,
            x2, y2, z2,
            rAB,
            s_total, s1, s2,
            mu1, mu2,
            w1, w2,
            shell_weight,
            coords, basis_idx, delta, M1M, M_inv, Fij, Fji, X
        )
        #
        # Bee, Fee = calc_F_ee(x1_flat, y1_flat, rAB, coords, basis_idx, delta, X)
        # Bne, Fne_1, Fne_alpha = calc_F_ne(x1_flat, y1_flat, rAB, coords, basis_idx, X)

        Ab2 = c_beta2 * B

        entry = {
            "x2": key[0], "y2": key[1], "z2": key[2],
            "s_total": s_total,  # length P1
            "B": B,
            "A1": A1,
            "Aa": Aa,
            "Ab": Ab,
            "Aa2": Aa2,
            "Aab": Aab,
            "Ab2": Ab2,
            # "Bee": Bee,
            # "Fee": Fee,
            # "Bne": Bne,
            # "Fne_1": Fne_1,
            # "Fne_alpha": Fne_alpha,
        }
        geom_cache[key] = entry
        return entry

    def rebuild_and_solve(alphas, betas, Rms):
        nonlocal meta, H, S, frankenBasis, E, C, sol_idx

        meta = None
        H = None
        S = None
        frankenBasis = None

        print("Rebuilding H/S from files...")
        for ff in files:
            print(" ", ff)
            with open(ff, "rb") as f:
                layers_new = pickle.load(f)

            if meta is None:
                if "meta" not in layers_new:
                    raise KeyError(f"'meta' not found in {ff}")
                meta = layers_new["meta"]

            H, S, frankenBasis = assemble_HS_multi_alpha(
                layers_new,
                alphas=alphas,
                betas=betas,
                Rms=Rms,
                H=H,
                S=S,
                coords=meta["coords"]
            )

            del layers_new
            gc.collect()

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

        print("E0:", E[sol_idx[0]] if sol_idx.size else E[0])
        reset_geom_cache()

    # ---------------------------
    # figure + widgets
    # ---------------------------
    fig = plt.figure(figsize=(18, 12))
    ax_phi = fig.add_subplot(2, 3, 1, projection="3d")
    ax_eloc = fig.add_subplot(2, 3, 2, projection="3d")
    ax_cusp = fig.add_subplot(2, 3, 3, projection="3d")

    ax_phi_R = fig.add_subplot(2, 3, 4, projection="3d")  # psi(rAB,x1)
    ax_eloc_R = fig.add_subplot(2, 3, 5, projection="3d")  # eloc(rAB,x1)

    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        bottom=0.1,  # was 0.08; slightly tighter
        top=0.95,
        wspace=0.15,
        hspace=0.15
    )
    for ax in (ax_phi, ax_eloc, ax_cusp, ax_phi_R, ax_eloc_R):
        ax.margins(x=0, y=0, z=0)
        ax.set_proj_type('ortho')

    ax_eloc.view_init(elev=0, azim=30)
    ax_cusp.view_init(elev=0, azim=90)
    ax_phi_R.view_init(elev=0, azim=-90)

    h_txt = 0.032  # Row heights
    h_sl = 0.020
    y_sl = 0.010
    y_txt = y_sl + h_sl + 0.010

    x0 = 0.06
    gap = 0.012
    w_btn = 0.12
    w_txt_total = 0.98 - x0 - gap - w_btn  # total width available for both text fields
    w_txt = (w_txt_total - gap) * 0.4

    ax_alpha_txt = fig.add_axes([x0, y_txt, w_txt, h_txt])
    ax_pairs_txt = fig.add_axes([x0 + w_txt + gap, y_txt, w_txt, h_txt])
    ax_apply = fig.add_axes([x0 + 2 * w_txt + 2 * gap, y_txt, w_btn, h_txt])

    t_alpha = TextBox(ax_alpha_txt, "alphas", initial=alpha_text_init)
    t_pairs = TextBox(ax_pairs_txt, "beta Rm", initial=beta_rm_text_init)
    b_apply = Button(ax_apply, "Apply")

    w_sl = (0.98 - x0 - 2 * gap) / 3.0   # Sliders row (x2/y2/z2)
    ax_x2 = fig.add_axes([x0, y_sl, w_sl, h_sl])
    ax_y2 = fig.add_axes([x0 + w_sl + gap, y_sl, w_sl, h_sl])
    ax_z2 = fig.add_axes([x0 + 2 * (w_sl + gap), y_sl, w_sl, h_sl])

    s_x2 = Slider(ax_x2, "x2", 0.0, 3.0, valinit=float(x2_init))
    s_y2 = Slider(ax_y2, "y2", 0.0, 3.0, valinit=float(y2_init))
    s_z2 = Slider(ax_z2, "z2", 0.0, 3.0, valinit=float(z2_init))


    def redraw(_=None):
        if meta is None or E is None or C is None or sol_idx is None or frankenBasis is None:
            return

        i_real = 0
        c_full = C[:, i_real]

        geom = get_cached_geom(s_x2.val, s_y2.val, s_z2.val)

        B   = geom["B"];   A1  = geom["A1"];  Aa = geom["Aa"];  Ab = geom["Ab"]
        Aa2 = geom["Aa2"]; Aab = geom["Aab"]; Ab2 = geom["Ab2"]
        s_total = geom["s_total"]

        # pointwise accumulators
        psi  = np.zeros(B.shape[0], dtype=np.float64)
        Hpsi = np.zeros(B.shape[0], dtype=np.float64)

        coords = meta["coords"]
        rAB_local = rAB

        # cusp accumulators (full-wavefunction cusp)
        Bee, Fee = calc_F_ee(x1_flat, y1_flat, rAB, coords, meta["basis_idx"], meta["delta"], frankenBasis, meta["X"])
        Bne, Fne, _ = calc_F_ne(x1_flat, y1_flat, rAB, coords, meta["basis_idx"], frankenBasis, meta["X"])

        N = frankenBasis.N
        nblocks = len(frankenBasis.blocks)
        assert c_full.shape[0] == nblocks * N
        assert B.shape[1] == N

        for _k, blk, sl in frankenBasis.iter_blocks():
            alpha_k = float(blk["alpha"])
            beta_k  = float(blk["beta"])
            Rm_k    = blk.get("Rm", None)
            ck = c_full[sl]

            psi0_k  = B @ ck
            A1c_k   = A1 @ ck
            Aac_k   = Aa @ ck
            Abc_k   = Ab @ ck
            Aa2c_k  = Aa2 @ ck
            Aabc_k  = Aab @ ck
            Ab2c_k  = Ab2 @ ck

            if coords.endswith("_morse"):
                exps_k = np.exp(-alpha_k * s_total - beta_k * (float(Rm_k) - rAB_local) ** 2)
            else:
                exps_k = np.exp(-alpha_k * s_total - beta_k * rAB_local)

            print(f"alpha = {blk['alpha']}, (beta, Rm) = ({blk['beta']}, {blk['Rm']}) contribution {np.linalg.norm(exps_k * psi0_k)}")
            rm = (rAB_local - float(Rm_k))
            psi  += exps_k * psi0_k
            Hpsi += exps_k * (
                A1c_k
                + alpha_k * Aac_k
                + beta_k * rm * Abc_k
                + (alpha_k ** 2) * Aa2c_k
                + (alpha_k * beta_k * rm) * Aabc_k
                + ((beta_k * rm) ** 2) * Ab2c_k  # todo: for nonBO, implement correct assembly of blocks (including extra rm factors and additive _beta_1 compontent!)
                + (2.0 * meta["M_inv"] * beta_k) * psi0_k
            )

            # # cusp: accumulate full numerator/denominator consistently per block
            # if np.ndim(Bee) > 0:
            #     psi_ee += exps_k * (Bee @ ck)
            #     F_ee   += exps_k * (Fee @ ck)
            #
            #     Fne_k = Fne_1 + Fne_alpha * alpha_k
            #     psi_ne += exps_k * (Bne @ ck)
            #     F_ne   += exps_k * (Fne_k @ ck)

        psi_ee = Bee @ c_full
        F_ee = Fee @ c_full
        psi_ne = Bne @ c_full
        F_ne = Fne @ c_full

        denom = np.where(np.abs(psi) < eps, np.nan, psi)
        Eloc = Hpsi / denom

        # SSE metric vs eigenvalue
        E_i = float(E[i_real])
        diff = Eloc - E_i
        epsilon = float(np.nansum(diff * diff))

        # normalize psi for display
        psi_disp = psi / (np.nanmax(np.abs(psi)) + 1e-300)
        if np.nanmax(psi_disp) < np.abs(np.nanmin(psi_disp)):
            psi_disp *= -1.0

        psi_grid  = psi_disp.reshape(Y1.shape)
        Eloc_grid = Eloc.reshape(Y1.shape)

        ax_phi.clear()
        ax_eloc.clear()
        ax_cusp.clear()

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

        # cusp surfaces if available
        if np.ndim(Bee) > 0:
            denom_ee = np.where(np.abs(psi_ee) < eps, np.nan, psi_ee)
            denom_ne = np.where(np.abs(psi_ne) < eps, np.nan, psi_ne)

            cusp_ee = (F_ee / denom_ee) - 0.5
            cusp_ne = (F_ne / denom_ne) + 1.0

            Zee = cusp_ee.reshape(Y1.shape)
            Zne = cusp_ne.reshape(Y1.shape)

            xmin, xmax = -3.0, 3.0
            ymin, ymax = 0.0, 1.0
            mask = (X1 < xmin) | (X1 > xmax) | (Y1 < ymin) | (Y1 > ymax)
            Zee = Zee.copy();  Zne = Zne.copy()
            Zee[mask] = np.nan
            Zne[mask] = np.nan

            surface_grid_colored(ax_cusp, X1, Y1, Zee, cmap_name="plasma",  vmin=-2.0, vmax=2.0, alpha=0.5)
            surface_grid_colored(ax_cusp, X1, Y1, Zne, cmap_name="viridis", vmin=-2.0, vmax=2.0, alpha=0.5)

            ax_cusp.set_xlim(-3.0, 3.0)
            ax_cusp.set_ylim(0.0, 3.0)
            ax_cusp.set_zlim(-0.05, 0.05)
            ax_cusp.set_title("Cusp functions (should be 0)\n(red: e-e, green: e-n)")

        # mark electron 2 position
        zmax1 = ax_phi.get_zlim()[1]
        zmax2 = ax_eloc.get_zlim()[1]
        ax_phi.scatter([geom["x2"]], [geom["y2"]], [zmax1], c=["orange"], s=160, depthshade=False)
        ax_eloc.scatter([geom["x2"]], [geom["y2"]], [zmax2], c=["orange"], s=160, depthshade=False)

        ax_phi.set_title(
            f"ψ | i={i_real}, E={E[i_real]:.10f}"
            # f"Electron 2 at: ({geom['x2']:.3f},{geom['y2']:.3f},{geom['z2']:.3f})"
        )
        ax_eloc.set_title(f"Local energy | epsilon={epsilon:.10e}")

        ax_phi.set_xlabel("x1");  ax_phi.set_ylabel("y1");  ax_phi.set_zlabel("ψ")
        ax_eloc.set_xlabel("x1"); ax_eloc.set_ylabel("y1"); ax_eloc.set_zlabel("Eloc")


        ax_phi_R.clear()
        ax_eloc_R.clear()

        if rAB_surf_min is None:
            rmin = float(plot_rAB_target) - 0.7
        else:
            rmin = float(rAB_surf_min)

        if rAB_surf_max is None:
            rmax = float(plot_rAB_target) + 0.7
        else:
            rmax = float(rAB_surf_max)

        rAB_vals = np.linspace(rmin, rmax, int(nrAB), dtype=np.float64)

        # x1 line grid (reuse your clustered_linspace for nicer focus near 0)
        x1_line_vals = clustered_linspace(-0.7-xy_max, -0.7+xy_max, int(nx1_line), strength=3.0) # shift by -0.7 such that clustering happens at x=-0.7
        y1_line_vals = np.array([float(y1_line_fixed)], dtype=np.float64)

        _, _, psi_ref_grid, _ = eval_psi_eloc_on_points(rAB_vals=np.array([1.4]), x1_vals=np.array([-0.70011]), y1_vals=np.array([0.0]), x2v=s_x2.val, y2v=s_y2.val, z2v=s_z2.val, c_full=c_full, frankenBasis=frankenBasis, meta=meta, eps=eps, psi_scale=1.0)
        psi_scale = 1.0 / float(psi_ref_grid[0, 0]) if abs(float(psi_ref_grid[0, 0])) > 0 else 1.0

        RABg, X1g, psi_R, eloc_R = eval_psi_eloc_on_points(
            rAB_vals=rAB_vals, x1_vals=x1_line_vals, y1_vals=y1_line_vals,
            x2v=s_x2.val, y2v=s_y2.val, z2v=s_z2.val,
            c_full=c_full, frankenBasis=frankenBasis, meta=meta, eps=eps, psi_scale=psi_scale)

        mask_r = (RABg < 1.0) | (RABg > 2.0)
        eloc_R = np.where(mask_r, np.nan, eloc_R)
        ax_eloc_R.set_xlim(1.0, 2.0)

        surface_grid_colored_discrete(ax_phi_R,  RABg, X1g, psi_R,  cmap_name="viridis", nlevels=int(psi_levels))

        if zlim_eloc is not None:
            surface_grid_colored_discrete(
                ax_eloc_R, RABg, X1g, eloc_R,
                cmap_name="viridis",
                vmin=float(zlim_eloc[0]), vmax=float(zlim_eloc[1]),
                nlevels=int(eloc_levels)
            )
            ax_eloc_R.set_zlim(float(zlim_eloc[0]), float(zlim_eloc[1]))
        else:
            surface_grid_colored_discrete(ax_eloc_R, RABg, X1g, eloc_R, cmap_name="viridis", nlevels=int(eloc_levels))

        ax_phi_R.set_xlabel("rAB")
        ax_phi_R.set_ylabel("x1 (y1=0)")
        ax_phi_R.set_zlabel("ψ")
        ax_phi_R.set_title("ψ(rAB, x1) with y1=0")

        ax_eloc_R.set_xlabel("rAB")
        ax_eloc_R.set_ylabel("x1 (y1=0)")
        ax_eloc_R.set_zlabel("Eloc")
        ax_eloc_R.set_title("Eloc(rAB, x1) with y1=0")

        fig.canvas.draw_idle()

    def on_apply(_evt):
        alphas = parse_list_or_range(t_alpha.text)
        pairs = parse_beta_rm_pairs(t_pairs.text)
        if alphas.size == 0:
            raise ValueError("No alphas provided.")
        if pairs.shape[0] == 0:
            raise ValueError("No (beta,Rm) pairs provided.")

        betas = pairs[:, 0]
        Rms   = pairs[:, 1]

        rebuild_and_solve(alphas, betas, Rms)
        # update_i_slider_max(sol_idx.size)
        redraw()

    b_apply.on_clicked(on_apply)

    # redraw on geometry changes
    # s_i.on_changed(redraw)
    s_x2.on_changed(redraw)
    s_y2.on_changed(redraw)
    s_z2.on_changed(redraw)

    # initial build + show
    on_apply(None)
    plt.show(block=True)


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

        (E, C, cond), frankenBasis = solve_HS_from_layers(SH_layers, alpha, beta, basis_idx, 1e-15, coords)
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
            "frankenBasis": frankenBasis
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
        frankenBasis = cache_entry["frankenBasis"]

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
        # print('+'.join([f"({float(cc / c[0])}) * rAB^{basis_idx[i][0]} " for i, cc in enumerate(c) if (basis_idx[i][1]==0 and basis_idx[i][2]==0 and basis_idx[i][3]==0 and basis_idx[i][4]==0 and basis_idx[i][5]==0)]))

        N = frankenBasis.N
        nblocks = len(frankenBasis.blocks)

        # sanity
        assert c.shape[0] == nblocks * N
        assert B.shape[1] == N

        # accumulators over points
        psi = np.zeros(B.shape[0], dtype=np.float64)
        Hpsi = np.zeros(B.shape[0], dtype=np.float64)

        for k, blk, sl in frankenBasis.iter_blocks():
            alpha_k = blk["alpha"]
            beta_k = blk["beta"]
            Rm_k = blk.get("Rm", None)
            ck = c[sl]  # tile coefficients for this block, length N

            # coefficient projections (pointwise vectors, length npts)
            psi0_k = B @ ck
            A1c_k = A1 @ ck
            Aac_k = Aa @ ck
            Abc_k = Ab @ ck
            Aa2c_k = Aa2 @ ck
            Aabc_k = Aab @ ck
            Ab2c_k = Ab2 @ ck

            # pointwise exp factor for this block
            if coords.endswith("_morse"):
                exps_k = np.exp(-alpha_k * s_total - beta_k * (Rm_k - rAB) ** 2)
            else:
                exps_k = np.exp(-alpha_k * s_total - beta_k * rAB)

            psi += exps_k * psi0_k
            Hpsi += exps_k * (
                    A1c_k
                    + alpha_k * Aac_k
                    + beta_k * Abc_k
                    + (alpha_k ** 2) * Aa2c_k
                    + (alpha_k * beta_k) * Aabc_k
                    + (beta_k ** 2) * Ab2c_k
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
    alphas,
    betas,
    Rms,
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
    frankenBasis = None

    for ff in files:
        print(ff)
        with open(ff, "rb") as f:
            layers_new = pickle.load(f)

        if meta is None:
            if "meta" not in layers_new:
                raise KeyError(f"'meta' not found in {ff}")
            meta = layers_new["meta"]

        # H, S = assemble_HS(layers_new, alpha, beta, H=H, S=S, coords=meta["coords"])  # Assemble matrices in-place
        H, S, frankenBasis = assemble_HS_multi_alpha(layers_new, alphas=alphas, betas = betas, Rms=Rms, H=H, S=S, coords=meta["coords"])

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

    print(E[0])
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
        # print("("+'+'.join([f"({float(cc / c[0])}) * rAB^{basis_idx[i][0]} " for i, cc in enumerate(c) if (basis_idx[i][1]==0 and basis_idx[i][2]==0 and basis_idx[i][3]==0 and basis_idx[i][4]==0 and basis_idx[i][5]==0)])+f")*exp(-{beta}*(rAB-1.4011)^2)")
        # print("("+'+'.join([f"({float(cc / c[0])}) * rAB^{basis_idx[i][0]} " for i, cc in enumerate(c) if (basis_idx[i][1]==1 and basis_idx[i][2]==0 and basis_idx[i][3]==0 and basis_idx[i][4]==0 and basis_idx[i][5]==0)])+f")*exp(-{beta}*(rAB-1.4011)^2)")
        # print("("+'+'.join([f"({float(cc / c[0])}) * rAB^{basis_idx[i][0]} " for i, cc in enumerate(c) if (basis_idx[i][1]==0 and basis_idx[i][2]==1 and basis_idx[i][3]==0 and basis_idx[i][4]==0 and basis_idx[i][5]==0)])+f")*exp(-{beta}*(rAB-1.4011)^2)")
        # print("("+'+'.join([f"({float(cc / c[0])}) * rAB^{basis_idx[i][0]} " for i, cc in enumerate(c) if (basis_idx[i][1]==0 and basis_idx[i][2]==0 and basis_idx[i][3]==0 and basis_idx[i][4]==1 and basis_idx[i][5]==1)])+f")*exp(-{beta}*(rAB-1.4011)^2)")

        geom = get_cached_geom(s_x2.val, s_y2.val, s_z2.val)

        B   = geom["B"];   A1  = geom["A1"];  Aa = geom["Aa"];  Ab = geom["Ab"]
        Aa2 = geom["Aa2"]; Aab = geom["Aab"]; Ab2 = geom["Ab2"]
        s_total = geom["s_total"]

        N = frankenBasis.N
        nblocks = len(frankenBasis.blocks)

        # sanity
        assert c.shape[0] == nblocks * N
        assert B.shape[1] == N

        # accumulators over points
        psi = np.zeros(B.shape[0], dtype=np.float64)
        Hpsi = np.zeros(B.shape[0], dtype=np.float64)

        for k, blk, sl in frankenBasis.iter_blocks():
            alpha_k = blk["alpha"]
            beta_k = blk["beta"]
            Rm_k = blk.get("Rm", None)
            ck = c[sl]  # tile coefficients for this block, length N

            # coefficient projections (pointwise vectors, length npts)
            psi0_k = B @ ck
            A1c_k = A1 @ ck
            Aac_k = Aa @ ck
            Abc_k = Ab @ ck
            Aa2c_k = Aa2 @ ck
            Aabc_k = Aab @ ck
            Ab2c_k = Ab2 @ ck

            # pointwise exp factor for this block
            if coords.endswith("_morse"):
                exps_k = np.exp(-alpha_k * s_total - beta_k * (Rm_k - rAB) ** 2)
            else:
                exps_k = np.exp(-alpha_k * s_total - beta_k * rAB)

            psi += exps_k * psi0_k
            Hpsi += exps_k * (
                    A1c_k
                    + alpha_k * Aac_k
                    + beta_k * Abc_k
                    + (alpha_k ** 2) * Aa2c_k
                    + (alpha_k * beta_k) * Aabc_k
                    + (beta_k ** 2) * Ab2c_k
            )

        # psi0 = B @ c
        # A1c = A1 @ c
        # Aac = Aa @ c
        # Abc = Ab @ c
        # Aa2c = Aa2 @ c
        # Aabc = Aab @ c
        # Ab2c = Ab2 @ c
        #
        # if coords.endswith("_morse"):
        #     exps = np.exp(-alpha * s_total - beta * (rAB - 1.4011) ** 2)
        # else:
        #     exps = np.exp(-alpha * s_total - beta * rAB)
        #
        # psi = exps * psi0
        # Hpsi = exps * (
        #     A1c
        #     + alpha * Aac
        #     + beta  * Abc
        #     + (alpha**2) * Aa2c
        #     + (alpha*beta) * Aabc
        #     + (beta**2) * Ab2c
        # )

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
            ax_cusp.set_zlim(-0.35, 0.35)

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
            f"ψ | alpha={alphas} beta={betas} Rm={Rms}\n"  # cond(S)={cache_entry['condS']:.3e}
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




