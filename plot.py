import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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
    ax_phi  = fig.add_subplot(1, 2, 1, projection="3d")
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
        surf_phi = ax_phi.plot_surface(X, Y, psi_xy, rstride=1, cstride=1,
                                       linewidth=0, antialiased=True)
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