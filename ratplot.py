import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.widgets import Slider
import pickle
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def save_psi_ad_bundle_pickle(
    base_filename: str,
    *,
    C,
    energy: float,
    n_max: int,
    k_max: int,
    h_max: int,
    alpha: float,
    label: str | None = None,
    extra: dict | None = None,
):
    """
    Save solution + plotting parameters into a pickle file.

    base_filename: e.g. "he_solution"
    label: optional tag appended to filename (e.g. "run3")
           -> he_solution_run3.pkl
    """

    filename = base_filename
    if label:
        filename = f"{base_filename}_{label}"
    filename = Path(filename).with_suffix(".pkl")

    bundle = {
        "energy": float(energy),
        "C": np.asarray(C, dtype=np.float64),
        "n_max": int(n_max),
        "k_max": int(k_max),
        "h_max": int(h_max),
        "alpha": float(alpha),
        "extra": extra if extra is not None else {},
    }

    with open(filename, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved bundle to {filename}")

def load_psi_ad_bundle_pickle(filename: str) -> dict:
    with open(filename, "rb") as f:
        bundle = pickle.load(f)

    # minimal sanity cleanup
    bundle["C"] = np.asarray(bundle["C"], dtype=np.float64)

    return bundle

def plot_from_psi_ad_bundle_pickle(
    filename: str,
    *,
    a_points: int = 80,
    d_points: int = 80,
    r_init: float = 1.0,
    r_min: float = 0.0,
    r_max: float = 10.0,
    use_abs: bool = False,
):
    """
    Loads pickle bundle and calls ratplot(...).
    """

    b = load_psi_ad_bundle_pickle(filename)

    print(b["energy"])
    fig, ax, slider = ratplot(
        b["C"]/b["C"][0],
        n_max=b["n_max"],
        k_max=b["k_max"],
        h_max=b["h_max"],
        alpha=b["alpha"],
        a_points=a_points,
        d_points=d_points,
        r_init=r_init,
        r_min=r_min,
        r_max=r_max,
        use_abs=use_abs,
    )

    ax.set_title(f"ψ(a,d;r)   E = {b['energy']:.12f}")
    fig.canvas.draw_idle()

    return fig, ax, slider, b


def _hl_index_list(h_max: int):
    """Return arrays h_list, l_list in the exact order used by R_matrix()."""
    h_list = []
    l_list = []
    for h in range(h_max + 1):
        for l in range(h // 2 + 1):
            h_list.append(h)
            l_list.append(l)
    return np.array(h_list, dtype=np.int64), np.array(l_list, dtype=np.int64)

def _to_maple_coeff(c: float, sig: int = 17) -> str:
    # Maple is fine with scientific notation like 1.23e-4
    return f"({c:.{sig}g})"

def print_solution_maple(
    C,
    *,
    n_max: int,
    k_max: int,
    h_max: int,
    alpha: float,
    coeff_sig: int = 17,
    drop_rel: float = 0.0,
    name: str = "psi",
):
    """
    Print the full solution in Maple syntax:

      psi := (a,d,r) -> exp(-alpha*r*(sqrt(2+2*a)+sqrt(2-2*a))) * (sum ...);

    Uses the basis ordering:
      hl: h=0..h_max, l=0..floor(h/2)
      then n=0..n_max, k=0..k_max
    """

    Nu, Kd = n_max + 1, k_max + 1
    h_list, l_list = _hl_index_list(h_max)
    Nrhl = len(h_list)

    C = np.asarray(C, dtype=np.float64).ravel()
    matsize_expected = Nrhl * Nu * Kd
    if C.size != matsize_expected:
        raise ValueError(f"C has length {C.size}, expected {matsize_expected}.")

    # Threshold for dropping tiny coefficients (keeps print sane)
    maxabs = float(np.max(np.abs(C))) if C.size else 0.0
    thresh = drop_rel * max(1.0, maxabs)

    # Build sum terms
    terms = []
    # Iterate in the exact order implied by C reshape
    idx = 0
    for hl in range(Nrhl):
        h = int(h_list[hl])
        l = int(l_list[hl])
        for n in range(Nu):
            for k in range(Kd):
                c = float(C[idx]); idx += 1
                if abs(c) <= thresh:
                    continue

                parts = [_to_maple_coeff(c, coeff_sig)]

                if h != 0:
                    parts.append(f"r^{h}")
                if l != 0:
                    parts.append(f"ln(r)^{l}")
                if n != 0:
                    parts.append(f"a^{2*n}")
                if k != 0:
                    parts.append(f"d^{k}")

                terms.append("*".join(parts))

    if not terms:
        body = "0"
    else:
        # Wrap lines for readability
        # Maple doesn't care about whitespace/newlines.
        chunks = []
        line = ""
        for t in terms:
            add = (" + " if line else "") + t
            if len(line) + len(add) > 120:
                chunks.append(line)
                line = t
            else:
                line += add
        if line:
            chunks.append(line)
        body = ("\n  + ".join(chunks)) if len(chunks) > 1 else chunks[0]

    print(f"\n# Maple syntax ({name})")
    print(f"{name} := exp(-{alpha}*r*(sqrt(2+2*a)+sqrt(2-2*a))/2) * (")
    print(f"  {body}")
    print("):")

def _qs_of_a(a):
    """Float64 version of qs = sqrt(2+2a) + sqrt(2-2a), for a in [0,1]."""
    a = np.asarray(a, dtype=np.float64)
    q1 = np.sqrt(2.0 + 2.0 * a)
    q2 = np.sqrt(2.0 - 2.0 * a)
    return q1 + q2

def _format_poly(var: str, coeffs, powers, *, max_terms=12, sig=6):
    """
    Pretty-ish polynomial string: sum_i coeffs[i] * var^(powers[i]).
    Shows up to max_terms nonzero-ish terms.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    powers = np.asarray(powers, dtype=np.int64)

    # filter tiny
    thresh = 1e-14 * max(1.0, np.max(np.abs(coeffs)))
    nz = np.where(np.abs(coeffs) > thresh)[0]

    if nz.size == 0:
        return "0"

    # keep largest terms by magnitude, but preserve increasing power order for readability
    if nz.size > max_terms:
        keep = nz[np.argsort(-np.abs(coeffs[nz]))[:max_terms]]
        keep = np.sort(keep)
    else:
        keep = nz

    parts = []
    for i in keep:
        c = coeffs[i]
        p = int(powers[i])
        cs = f"{c:.{sig}g}"
        if p == 0:
            parts.append(f"{cs}")
        elif p == 1:
            parts.append(f"{cs}*{var}")
        else:
            parts.append(f"{cs}*{var}^{p}")

    return " + ".join(parts)

def plot_psi_ar_at_d1(
    *,
    C_hlnk,          # (Nrhl, Nu, Kd)
    h_list, l_list,  # (Nrhl,), (Nrhl,)
    n_max: int,
    k_max: int,
    alpha: float,
    r_min: float,
    r_max: float,
    a_points: int = 241,
    r_points: int = 140,
    eps_r: float = 1e-12,
    use_abs: bool = False,
):
    """
    3D plot of psi at fixed d=1: surface over (a, r).

    At d=1: polynomial in d collapses to sum_k coeff[n,k] (since 1^k = 1).
    Psi(a,r) = exp(-alpha*r*qs(a)) * sum_n a^(2n) * sum_k coeff_nk(r)[n,k].
    """
    Nu = n_max + 1
    Kd = k_max + 1

    # grids
    a_grid = np.linspace(-1.0, 1.0, a_points, dtype=np.float64)
    r_grid = np.linspace(r_min, r_max, r_points, dtype=np.float64)

    # a^(2n) table: (a_points, Nu)
    A2N = np.empty((a_points, Nu), dtype=np.float64)
    A2N[:, 0] = 1.0
    a2 = a_grid * a_grid
    for n in range(1, Nu):
        A2N[:, n] = A2N[:, n - 1] * a2

    # qs(a): (a_points,)
    QS = _qs_of_a(a_grid).astype(np.float64)

    # output surface
    A = np.repeat(a_grid[None, :], r_points, axis=0)     # (r_points, a_points)
    R = np.repeat(r_grid[:, None], a_points, axis=1)     # (r_points, a_points)
    Z = np.empty((r_points, a_points), dtype=np.float64)

    for i, rv in enumerate(r_grid):
        r_eff = max(float(rv), eps_r)
        log_r = np.log(r_eff)

        # r^h * log(r)^l across hl
        RHL = (r_eff ** h_list) * (log_r ** l_list)                  # (Nrhl,)

        # coeff_nk(r) = sum_hl RHL[hl] * C_hlnk[hl,n,k]
        coeff_nk = np.tensordot(RHL, C_hlnk, axes=(0, 0))            # (Nu, Kd)

        # d=1 -> sum over k
        coeff_n = coeff_nk.sum(axis=1)                               # (Nu,)

        # combine n with a^(2n)
        psi_a = A2N @ coeff_n                                        # (a_points,)

        # exponential prefactor depends on a
        # psi_a *= np.exp(-alpha * r_eff * QS) # todo: exp test

        if use_abs:
            psi_a = np.abs(psi_a)

        Z[i, :] = psi_a

    # plot
    fig4 = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax4 = fig4.add_subplot(111, projection="3d")
    ax4.plot_surface(A, R, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)

    ax4.set_xlabel("a")
    ax4.set_ylabel("r")
    ax4.set_zlabel("|psi|" if use_abs else "psi")
    ax4.set_title("ψ(a,r) at fixed d=1")
    ax4.set_proj_type("ortho")
    ax4.view_init(elev=0, azim=90)

    return fig4, ax4

def plot_psi_rd_at_a0(
    *,
    C_hlnk,          # (Nrhl, Nu, Kd)
    h_list, l_list,  # (Nrhl,), (Nrhl,)
    n_max: int,
    k_max: int,
    alpha: float,
    r_min: float = 0.0,
    r_max: float = 10.0,
    r_points: int = 120,
    d_points: int = 120,
    eps_r: float = 1e-12,
    use_abs: bool = False,
):
    """
    3D plot of psi at fixed a=0: surface over (r, d).

    At a=0: a^(2n) is 1 for n=0 and 0 for n>0 -> only n=0 contributes.
    """
    Nu = n_max + 1
    Kd = k_max + 1

    # d-range for a=0 using same bounds function as ratplot()
    a0 = 0.0
    s = np.sqrt(1 - a0**2)              # =1
    dmin = np.sqrt(1 + s)
    dmax = np.sqrt(1 - s)
    d_grid = np.linspace(dmin, dmax, d_points, dtype=np.float64)

    # r grid
    r_grid = np.linspace(r_min, r_max, r_points, dtype=np.float64)

    # qs(a) for a=0 (shape (1,) -> scalar)
    qs0 = float(_qs_of_a(np.array([a0], dtype=np.float64))[0])

    # Precompute d powers table: (Kd, d_points)
    d_pows = np.ones((Kd, d_points), dtype=np.float64)
    for kk in range(1, Kd):
        d_pows[kk, :] = d_pows[kk - 1, :] * d_grid

    # Build surface arrays
    R = np.repeat(r_grid[:, None], d_points, axis=1)   # (r_points, d_points)
    D = np.repeat(d_grid[None, :], r_points, axis=0)   # (r_points, d_points)
    Z = np.empty_like(R, dtype=np.float64)

    for i, rv in enumerate(r_grid):
        r_eff = max(float(rv), eps_r)
        log_r = np.log(r_eff)

        # r^h * log(r)^l across hl
        RHL = (r_eff ** h_list) * (log_r ** l_list)                  # (Nrhl,)

        # coeff_nk = sum_hl RHL[hl] * C_hlnk[hl,n,k]
        coeff_nk = np.tensordot(RHL, C_hlnk, axes=(0, 0))            # (Nu, Kd)

        # a=0 -> only n=0 survives
        coeff_k = coeff_nk[0, :]                                     # (Kd,)

        # polynomial in d
        psi_poly = coeff_k @ d_pows                                  # (d_points,)

        # exponential prefactor
        # psi_row = np.exp(-alpha * r_eff * qs0) * psi_poly            # (d_points,)
        psi_row = psi_poly            # todo: (d_points,) # no-exp test
        if use_abs:
            psi_row = np.abs(psi_row)

        Z[i, :] = psi_row

    # Plot
    fig3 = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.plot_surface(R, D, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)

    ax3.set_xlabel("r")
    ax3.set_ylabel("d")
    ax3.set_zlabel("|psi|" if use_abs else "psi")
    ax3.set_title("ψ(r,d) at fixed a=0")
    ax3.set_proj_type("ortho")
    ax3.view_init(elev=0, azim=90)

    return fig3, ax3

def plot_r_power_components_at_r1(
    *,
    C_hlnk,          # (Nrhl, Nu, Kd)
    h_list, l_list,  # (Nrhl,), (Nrhl,)
    A, D,            # (a_points, d_points) surfaces
    A2N,             # (a_points, Nu)
    QS,              # (a_points,)
    alpha: float,
    h_max: int,
    use_abs: bool = False,
    r0: float = 1.0,
    wire_stride: int = 2,
):
    """
    Plot the contribution of each r^h (at r=r0) in a single 3D plot.

    With r0=1: ln(r0)=0, so only l=0 contributes; and r0^h=1.
    Therefore the 'coeff(psi, r, h)' surface is built from the (h, l=0) block only.
    """
    a_points, d_points = A.shape

    # common exponential factor at r=r0
    expo = np.exp(-alpha * float(r0) * QS)[:, None]  # (a_points, 1)

    fig2 = plt.figure(figsize=(10, 7))
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.set_xlabel("a")
    ax2.set_ylabel("d")
    ax2.set_zlabel("|psi_h|" if use_abs else "psi_h")
    ax2.set_title(f"ψ_h(a,d) pieces for each r^h at r={r0:g} (ln(r)=0 ⇒ l=0 only)")
    ax2.set_proj_type('ortho')
    ax2.view_init(elev=0, azim=90)
    fig2.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95)

    cmap = pyplot.get_cmap("viridis", h_max + 1)
    norm = mcolors.Normalize(vmin=0, vmax=h_max)

    # For each h, find the unique hl index where (h_list==h and l_list==0)
    for h in range(h_max + 1):
        mask = (h_list == h) & (l_list == 0)
        idxs = np.where(mask)[0]
        if idxs.size != 1:
            # If something odd happens in ordering, just skip safely
            continue
        hl0 = int(idxs[0])

        coeff_nk = C_hlnk[hl0, :, :]                      # (Nu, Kd)
        coeff_k_by_a = A2N @ coeff_nk                     # (a_points, Kd)

        # Evaluate polynomial in d on the surface grid
        psi_poly = np.zeros((a_points, d_points), dtype=np.float64)
        Dpow = np.ones((a_points, d_points), dtype=np.float64)
        for kk in range(coeff_nk.shape[1]):
            psi_poly += coeff_k_by_a[:, kk][:, None] * Dpow
            Dpow *= D

        Z = expo * psi_poly
        # Z = psi_poly
        if use_abs:
            Z = np.abs(Z)

        ax2.plot_wireframe(
            A, D, Z,
            rstride=wire_stride,
            cstride=wire_stride,
            linewidth=0.9,
            color=cmap(norm(h)),
            label=f"h={h}",
        )

    # 3D legends are finicky but usually work OK for wireframes
    try:
        ax2.legend(loc="upper left")
    except Exception:
        pass

    return fig2, ax2


def ratplot(
    C,
    *,
    n_max: int,
    k_max: int,
    h_max: int,
    alpha: float,
    a_points: int = 80,
    d_points: int = 80,
    r_init: float = 1.0,
    r_min: float = 0.0,
    r_max: float = 10.0,
    eps_r: float = 1e-12,
    use_abs: bool = False,
):
    """
    Plot psi(a,d;r) over a∈[0,1] with d-bounds depending on a, and a slider for r.

    Parameters
    ----------
    C : (matsize,) array
        Eigenvector
    n_max, k_max, h_max, alpha : as in the file.
    a_points, d_points : grid resolution for plotting.
    r_init : initial r for the plot.
    r_min, r_max : slider range for r.
    eps_r : used when r==0 to avoid log(0) and r^h issues.
    use_abs : if True, plot |psi| instead of psi (useful if sign changes make surface hard to read).
    """

    Nu = n_max + 1
    Kd = k_max + 1

    # Build (h,l) list in the exact same order as in R_matrix()
    h_list, l_list = _hl_index_list(h_max)
    Nrhl = len(h_list)

    print_solution_maple(
        C,
        n_max=n_max,
        k_max=k_max,
        h_max=h_max,
        alpha=alpha,
        name="psi",
    )

    matsize_expected = Nrhl * Nu * Kd
    C = np.asarray(C, dtype=np.float64).ravel()
    if C.size != matsize_expected:
        raise ValueError(f"C has length {C.size}, expected {matsize_expected} = Nrhl*Nu*Kd.")

    # Reshape to C[hl, n, k] using the ordering inferred from kron structure:
    C_hlnk = C.reshape(Nrhl, Nu, Kd)

    # a grid (plot grid, not quadrature)
    a_grid = np.linspace(-0.99, 0.99, a_points, dtype=np.float64)

    def d_bounds_from_a(a):
        s = np.sqrt(1-a**2)  # s = sqrt(1 - a^2)
        dmin = np.sqrt(1+s)  # sqrt(1 - s)
        dmax = np.sqrt(1-s)  # sqrt(1 + s)
        return dmin, dmax

    dmin = np.empty_like(a_grid)
    dmax = np.empty_like(a_grid)
    for i, av in enumerate(a_grid):
        dmin[i], dmax[i] = d_bounds_from_a(av)

    # Build a rectangular surface grid with per-row d ranges
    A = np.repeat(a_grid[:, None], d_points, axis=1)
    D = np.empty_like(A)
    for i in range(a_points):
        D[i, :] = np.linspace(dmin[i], dmax[i], d_points, dtype=np.float64)

    n = np.arange(Nu, dtype=np.int64)
    k = np.arange(Kd, dtype=np.int64)

    # Precompute a^(2n) table: shape (a_points, Nu)
    A2N = (a_grid[:, None] ** (2 * n[None, :])).astype(np.float64)

    # Precompute qs(a) for the exponential
    QS = _qs_of_a(a_grid)

    def compute_psi_for_r(r_value: float):
        r_eff = max(float(r_value), eps_r)
        log_r = np.log(r_eff)

        # r^h * log(r)^l for each hl
        r_h = (r_eff ** h_list).astype(np.float64)                 # (Nrhl,)
        log_l = (log_r ** l_list).astype(np.float64)               # (Nrhl,)
        RHL = r_h * log_l                                          # (Nrhl,)

        # coeff_nk[n,k] = sum_hl C[hl,n,k] * RHL[hl]
        coeff_nk = np.tensordot(RHL, C_hlnk, axes=(0, 0))          # (Nu, Kd)

        # --- sub-polynomials (POLYNOMIAL PART ONLY, no exp factor) ---
        # P(0,d;r) = sum_k coeff_nk[0,k] d^k
        d_poly_coeffs = coeff_nk[0, :].copy()                      # (Kd,)

        # P(a,1;r) = sum_n (sum_k coeff_nk[n,k]) a^(2n)
        a_poly_coeffs = coeff_nk.sum(axis=1).copy()                # (Nu,)

        # Evaluate surface P(a,d;r) on the plot grid
        coeff_k_by_a = A2N @ coeff_nk                               # (a_points, Kd)

        psi_poly = np.zeros((a_points, d_points), dtype=np.float64)
        Dpow = np.ones((a_points, d_points), dtype=np.float64)
        for kk in range(Kd):
            psi_poly += coeff_k_by_a[:, kk][:, None] * Dpow
            Dpow *= D

        # expo = np.exp(-alpha * r_eff * QS)[:, None]
        # psi = expo * psi_poly # todo: test
        psi = psi_poly
        if use_abs:
            psi = np.abs(psi)

        return psi, d_poly_coeffs, a_poly_coeffs


    # --- Matplotlib interactive plot ---
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.2)

    Z, d_poly_coeffs, a_poly_coeffs = compute_psi_for_r(r_init)

    k_powers = np.arange(Kd, dtype=np.int64)
    n_powers = 2 * np.arange(Nu, dtype=np.int64)  # powers of a are 0,2,4,...
    print(_format_poly("d", d_poly_coeffs, k_powers, max_terms=10, sig=6))
    print(_format_poly("a", a_poly_coeffs, n_powers, max_terms=10, sig=6))

    surf = ax.plot_surface(A, D, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)

    ax.set_xlabel("a")
    ax.set_ylabel("d")
    ax.set_zlabel("|psi|" if use_abs else "psi")
    ax.set_title("ψ(a,d;r) with allowed d-range per a")
    ax.set_proj_type('ortho')
    ax.view_init(elev=0, azim=90)

    # Slider
    slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.04])
    r_slider = Slider(slider_ax, "r", r_min, r_max, valinit=r_init)

    def _update(val):
        nonlocal surf
        r_val = r_slider.val
        Znew, d_poly_coeffs, a_poly_coeffs = compute_psi_for_r(r_val)

        k_powers = np.arange(Kd, dtype=np.int64)
        n_powers = 2 * np.arange(Nu, dtype=np.int64)  # powers of a are 0,2,4,...
        print(_format_poly("d", d_poly_coeffs, k_powers, max_terms=10, sig=6))
        print(_format_poly("a", a_poly_coeffs, n_powers, max_terms=10, sig=6))

        # redraw surface (matplotlib 3d doesn't support efficient z-update)
        surf.remove()
        surf = ax.plot_surface(A, D, Znew, rstride=1, cstride=1, linewidth=0, antialiased=True)
        fig.canvas.draw_idle()

    r_slider.on_changed(_update)


    # --- ADD inside ratplot(), after QS / C_hlnk exist ---
    fig2, ax2 = plot_r_power_components_at_r1(
        C_hlnk=C_hlnk,
        h_list=h_list,
        l_list=l_list,
        A=A,
        D=D,
        A2N=A2N,
        QS=QS,
        alpha=alpha,
        h_max=h_max,
        use_abs=use_abs,
        r0=1.0,
        wire_stride=2,
    )

    fig3, ax3 = plot_psi_rd_at_a0(
        C_hlnk=C_hlnk,
        h_list=h_list,
        l_list=l_list,
        n_max=n_max,
        k_max=k_max,
        alpha=alpha,
        r_min=0.0,
        r_max=2.0,
        r_points=60,
        d_points=d_points,
        eps_r=eps_r,
        use_abs=use_abs,
    )

    fig4, ax4 = plot_psi_ar_at_d1(
        C_hlnk=C_hlnk,
        h_list=h_list,
        l_list=l_list,
        n_max=n_max,
        k_max=k_max,
        alpha=alpha,
        r_min=0.0,
        r_max=2.0,
        a_points=40,
        r_points=40,
        eps_r=eps_r,
        use_abs=use_abs,
    )

    plt.show()

    return fig, ax, r_slider


if __name__ == "__main__":
    # plot_from_psi_ad_bundle_pickle("he_solution_he_p8_s100.pkl", a_points=40, d_points=40)
    # plot_from_psi_ad_bundle_pickle("he_solution_he_p10_r80_a80_gamma4.pkl", a_points=40, d_points=40)
    plot_from_psi_ad_bundle_pickle("he_solution_he_p10_r100_a100_h3_k3_n3_gamma4.pkl", a_points=40, d_points=40)