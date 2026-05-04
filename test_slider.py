import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from AFMReader.jpk import load_jpk

# ── Config ────────────────────────────────────────────────────────────────────
FOLDER  = "/Users/lmonari/Documents/afm/PART2/Box-4/2026_03_02_hepes_spin_filtration_flatten"
CHANNEL = "height_trace"


# ── Utilities ─────────────────────────────────────────────────────────────────
def nanmin_safe(x):
    return float(np.nanmin(x))

def nanmax_safe(x):
    return float(np.nanmax(x))


# ── Processing functions ──────────────────────────────────────────────────────
def row_align_poly(img, mask=None, degree=3):
    """
    Polynomial row alignment.
    If mask is provided, fit only on masked (True) pixels.
    """
    aligned = img.copy().astype(float)
    x = np.arange(aligned.shape[1])

    for i in range(aligned.shape[0]):
        row = aligned[i]

        if mask is not None:
            m = mask[i]
            valid = m & np.isfinite(row)

            if np.count_nonzero(valid) > degree + 1:
                coeffs = np.polyfit(x[valid], row[valid], deg=degree)
            else:
                valid_all = np.isfinite(row)
                coeffs = np.polyfit(x[valid_all], row[valid_all], deg=degree)
        else:
            valid_all = np.isfinite(row)
            coeffs = np.polyfit(x[valid_all], row[valid_all], deg=degree)

        aligned[i] -= np.polyval(coeffs, x)

    return aligned


def remove_scars(img, threshold_factor=3.0):
    """
    Horizontal scar correction.
    A scar is detected when the median-normalised row differs from its
    neighbours by more than threshold_factor * global MAD.
    Scarred rows are replaced by interpolation from nearest non-scarred rows.
    """
    corrected = img.copy().astype(float)

    row_medians = np.nanmedian(corrected, axis=1)
    diff = np.abs(np.diff(row_medians))
    mad = np.nanmedian(diff)

    if not np.isfinite(mad) or mad == 0:
        return corrected

    threshold = threshold_factor * mad
    scar_rows = np.where(diff > threshold)[0] + 1

    scar_set = set(scar_rows.tolist())

    for r in scar_rows:
        above = r - 1
        below = r + 1

        while above in scar_set and above > 0:
            above -= 1
        while below in scar_set and below < img.shape[0] - 1:
            below += 1

        if above >= 0 and below < img.shape[0]:
            corrected[r] = 0.5 * (corrected[above] + corrected[below])
        elif above >= 0:
            corrected[r] = corrected[above]
        elif below < img.shape[0]:
            corrected[r] = corrected[below]

    n_scars = len(scar_rows)
    if n_scars:
        print(f"  Scar correction: {n_scars} row(s) corrected")

    return corrected


def clip_only(img, zmin, zmax):
    """
    Clip to [zmin, zmax] without renormalizing.
    This keeps the displayed scale consistent with the slider values.
    """
    return np.clip(img, zmin, zmax)


def coarse_for_mask(image_nm):
    """
    Coarse alignment used only to build the background mask.
    """
    coarse = row_align_poly(image_nm, mask=None, degree=1)
    coarse -= nanmin_safe(coarse)
    return coarse


def full_pipeline(image_nm, mask_threshold_nm=2.0, poly_degree=3):
    """
    Full processing pipeline:
    1. Coarse row alignment (no mask)
    2. Build mask: pixels below mask_threshold are 'background'
    3. Scar correction on coarse-aligned image
    4. Masked polynomial row alignment (ignore features above threshold)
    5. Final floor shift to 0
    """
    # Step 1 — coarse alignment to get approximate heights for masking
    coarse = coarse_for_mask(image_nm)

    # Step 2 — background mask: True where height < threshold
    mask = coarse < mask_threshold_nm

    # Step 3 — scar correction on coarse image
    scar_corrected = remove_scars(coarse)

    # Step 4 — masked polynomial alignment (fit only background pixels)
    aligned = row_align_poly(scar_corrected, mask=mask, degree=poly_degree)

    # Step 5 — final floor shift
    aligned -= nanmin_safe(aligned)

    return aligned, mask, coarse


def process_file(filepath):
    image, px_to_nm = load_jpk(file_path=filepath, channel=CHANNEL)

    z_range = np.nanmax(image) - np.nanmin(image)
    image_nm = image * 1e9 if z_range < 1e-3 else image.copy().astype(float)

    scan_size_um = image_nm.shape[1] * (px_to_nm * 1e6)
    return image_nm, scan_size_um


# ── Per-file interactive viewer ───────────────────────────────────────────────
jpk_files = sorted([f for f in os.listdir(FOLDER) if f.endswith(".jpk")])
print(f"Found {len(jpk_files)} JPK files")

for filename in jpk_files:
    print(f"\nOpening: {filename}")
    filepath = os.path.join(FOLDER, filename)
    image_nm, scan_size_um = process_file(filepath)

    extent = [0, scan_size_um, 0, scan_size_um]
    save_path = os.path.splitext(filepath)[0] + ".png"

    # Initial pipeline run
    image_aligned, mask, coarse = full_pipeline(image_nm, mask_threshold_nm=0.0)

    # IMPORTANT: separate domains for mask slider and z sliders
    mask_max = nanmax_safe(coarse)

    # Initial settings
    INIT_MASK_NM = mask_max # 2.0
    INIT_ZMIN    = 0.0
    INIT_ZMAX    = mask_max # 4.0

    # aligned_max = nanmax_safe(image_aligned)
    # if aligned_max >= 50.0:
    #     print(f"Warning: aligned heights have large range ({aligned_max:.2f} nm), capped at 50 nm for sliders")
    #     aligned_max = 50.0
    # print(f"Initial mask threshold: {INIT_MASK_NM:.2f} nm (max coarse height: {mask_max:.2f} nm)")
    # print(f"Initial Z max: {INIT_ZMAX:.2f} nm (max aligned height: {aligned_max:.2f} nm)")

    init_zmax = min(INIT_ZMAX, mask_max) if mask_max > 0 else INIT_ZMAX
    display_data = clip_only(image_aligned, INIT_ZMIN, init_zmax)

    # ── Layout: 3 panels (raw | mask | aligned) ───────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(bottom=0.28)
    fig.suptitle(filename, fontsize=10)

    # Panel 1 — Raw
    im_raw = axes[0].imshow(
        image_nm,
        cmap="afmhot",
        origin="upper",
        extent=extent,
        interpolation="nearest",
        vmin=nanmin_safe(image_nm),
        vmax=nanmax_safe(image_nm),
    )
    axes[0].set_title("Raw height (nm)")
    axes[0].set_xlabel("X (µm)")
    axes[0].set_ylabel("Y (µm)")
    fig.colorbar(im_raw, ax=axes[0], fraction=0.046, pad=0.04).set_label("Height (nm)")

    # Panel 2 — Mask (background = white, features = black)
    im_mask = axes[1].imshow(
        mask.astype(float),
        cmap="gray",
        origin="upper",
        extent=extent,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(f"Background mask (< {INIT_MASK_NM:.1f} nm)")
    axes[1].set_xlabel("X (µm)")
    axes[1].set_ylabel("Y (µm)")

    # Panel 3 — Aligned
    im_aligned = axes[2].imshow(
        display_data,
        cmap="afmhot",
        origin="upper",
        extent=extent,
        interpolation="nearest",
        vmin=INIT_ZMIN,
        vmax=init_zmax,
    )
    axes[2].set_title("Processed height (nm)")
    axes[2].set_xlabel("X (µm)")
    axes[2].set_ylabel("Y (µm)")
    cbar_aligned = fig.colorbar(im_aligned, ax=axes[2], fraction=0.046, pad=0.04)
    cbar_aligned.set_label("Height (nm)")

    # ── Sliders ───────────────────────────────────────────────────────────────
    ax_smask = fig.add_axes([0.15, 0.18, 0.65, 0.03])
    ax_smin  = fig.add_axes([0.15, 0.12, 0.65, 0.03])
    ax_smax  = fig.add_axes([0.15, 0.07, 0.65, 0.03])

    slider_mask = Slider(
        ax_smask,
        "Mask threshold (nm)",
        0.0,
        max(mask_max, INIT_MASK_NM),
        valinit=INIT_MASK_NM,
        valstep=0.1,
        valfmt="%.2f",
    )

    slider_min = Slider(
        ax_smin,
        "Z min (nm)",
        0.0,
        max(mask_max, INIT_ZMAX),
        valinit=INIT_ZMIN,
        valstep=0.05,
        valfmt="%.2f",
    )

    slider_max = Slider(
        ax_smax,
        "Z max (nm)",
        0.0,
        max(mask_max, INIT_ZMAX),
        valinit=init_zmax,
        valstep=0.05,
        valfmt="%.2f",
    )

    # ── Save button ───────────────────────────────────────────────────────────
    ax_btn = fig.add_axes([0.42, 0.01, 0.15, 0.04])
    btn_save = Button(ax_btn, "Save PNG")

    # Shared state
    state = {
        "aligned": image_aligned,
        "coarse": coarse,
        "mask": mask,
    }

    def update_mask(
        val,
        state=state,
        fig=fig,
        im_mask=im_mask,
        im_aligned=im_aligned,
        cbar=cbar_aligned,
        axes=axes,
        image_nm=image_nm,
        smask=slider_mask,
        smin=slider_min,
        smax=slider_max,
    ):
        mask_thresh = smask.val
        zmin = smin.val
        zmax = smax.val

        if zmin >= zmax:
            return

        # Re-run pipeline with the new mask threshold
        new_aligned, new_mask, new_coarse = full_pipeline(
            image_nm,
            mask_threshold_nm=mask_thresh,
        )

        state["aligned"] = new_aligned
        state["coarse"] = new_coarse
        state["mask"] = new_mask

        # Update mask panel
        im_mask.set_data(new_mask.astype(float))
        axes[1].set_title(f"Background mask (< {mask_thresh:.1f} nm)")

        # Update aligned panel
        im_aligned.set_data(clip_only(new_aligned, zmin, zmax))
        im_aligned.set_clim(vmin=zmin, vmax=zmax)
        cbar.update_normal(im_aligned)

        fig.canvas.draw_idle()

    def update_display(
        val,
        state=state,
        fig=fig,
        im_aligned=im_aligned,
        cbar=cbar_aligned,
        smin=slider_min,
        smax=slider_max,
    ):
        zmin = smin.val
        zmax = smax.val

        if zmin >= zmax:
            return

        im_aligned.set_data(clip_only(state["aligned"], zmin, zmax))
        im_aligned.set_clim(vmin=zmin, vmax=zmax)
        cbar.update_normal(im_aligned)

        fig.canvas.draw_idle()

    def save(event, fig=fig, path=save_path):
        fig.savefig(path, dpi=900, bbox_inches="tight")
        print(f"Saved: {path}")

    slider_mask.on_changed(update_mask)
    slider_min.on_changed(update_display)
    slider_max.on_changed(update_display)
    btn_save.on_clicked(save)

    plt.show()