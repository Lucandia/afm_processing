import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from AFMReader.jpk import load_jpk

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
CHANNEL     = "height_trace"
POLY_DEGREE = 3   # degree for masked polynomial row alignment

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════════
def nanmin_safe(x: np.ndarray) -> float:
    return float(np.nanmin(x))

def nanmax_safe(x: np.ndarray) -> float:
    return float(np.nanmax(x))

# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def row_align_poly(img: np.ndarray, mask: np.ndarray = None, degree: int = 3) -> np.ndarray:
    """
    Polynomial row-by-row alignment.

    For each scan line, fits a polynomial of the given degree and subtracts it.
    If a boolean mask is provided (True = background pixels), the polynomial is
    fitted only on background pixels — this prevents tall features from biasing
    the baseline correction.

    Parameters
    ----------
    img    : 2-D height array (nm)
    mask   : bool array same shape as img; True = background pixel used for fitting
    degree : polynomial degree (1 = linear tilt, 2 = bow, 3 = higher-order warp)
    """
    aligned = img.copy().astype(float)
    x = np.arange(aligned.shape[1])

    for i in range(aligned.shape[0]):
        row = aligned[i]
        if mask is not None:
            m = mask[i] & np.isfinite(row)
            # Fall back to all finite pixels if not enough background points
            fit_x = x[m] if m.sum() > degree + 1 else x[np.isfinite(row)]
            fit_y = row[m] if m.sum() > degree + 1 else row[np.isfinite(row)]
        else:
            fit_x = x[np.isfinite(row)]
            fit_y = row[np.isfinite(row)]

        coeffs = np.polyfit(fit_x, fit_y, deg=degree)
        aligned[i] -= np.polyval(coeffs, x)

    return aligned


def remove_scars(img: np.ndarray, threshold_factor: float = 3.0) -> np.ndarray:
    """
    Horizontal scar correction.

    Detects scan lines whose median height deviates from their neighbours by
    more than `threshold_factor` × MAD of inter-row differences. Scarred rows
    are replaced by linear interpolation between the nearest clean neighbours.

    Parameters
    ----------
    img              : 2-D height array (nm)
    threshold_factor : sensitivity — lower = more aggressive correction
    """
    corrected    = img.copy().astype(float)
    row_medians  = np.nanmedian(corrected, axis=1)
    diff         = np.abs(np.diff(row_medians))
    mad          = np.nanmedian(diff)

    if not np.isfinite(mad) or mad == 0:
        return corrected

    scar_rows = np.where(diff > threshold_factor * mad)[0] + 1
    scar_set  = set(scar_rows.tolist())

    for r in scar_rows:
        # Walk outward from the scarred row to find clean neighbours
        above, below = r - 1, r + 1
        while above in scar_set and above > 0:            above -= 1
        while below in scar_set and below < img.shape[0]: below += 1

        if above >= 0 and below < img.shape[0]:
            corrected[r] = 0.5 * (corrected[above] + corrected[below])
        elif above >= 0:
            corrected[r] = corrected[above]
        elif below < img.shape[0]:
            corrected[r] = corrected[below]

    return corrected


# def full_pipeline(image_nm: np.ndarray,
#                   mask_threshold_nm: float = 2.0,
#                   poly_degree: int = POLY_DEGREE):
#     """
#     Full AFM processing pipeline:

#       1. Coarse linear (degree-1) row alignment — no mask, just removes gross tilt
#       2. Background mask — pixels below `mask_threshold_nm` are labelled background
#       3. Scar correction — scarred rows replaced by neighbour interpolation
#       4. Masked polynomial alignment — polynomial fitted only on background pixels
#       5. Floor shift — minimum set to 0 nm

#     Returns
#     -------
#     aligned : processed height array (nm), floor at 0
#     mask    : bool array, True = background (used for display in panel 2)
#     coarse  : coarse-aligned array (nm), used to set slider ranges
#     """
#     # Step 1 — coarse alignment (no mask, degree 1)
#     coarse  = row_align_poly(image_nm, mask=None, degree=1)
#     coarse -= nanmin_safe(coarse)

#     # Step 2 — background mask
#     mask = coarse < mask_threshold_nm

#     # Step 3 — scar correction on coarse image
#     scar_corrected = remove_scars(coarse)

#     # Step 4 — masked polynomial alignment
#     aligned  = row_align_poly(scar_corrected, mask=mask, degree=poly_degree)
#     aligned -= nanmin_safe(aligned)

#     return aligned, mask, coarse

def full_pipeline(image_nm: np.ndarray,
                  mask_threshold_nm: float = 2.0,
                  poly_degree: int = POLY_DEGREE,
                  iterations: int = 1):
    """
    Full AFM processing pipeline (repeated `iterations` times).

    Each iteration:
      0. Pre-step: light scar removal on raw/previous-iteration data
      1. Coarse linear row alignment (no mask)
      2. Background mask (pixels below mask_threshold_nm)
      3. Scar correction
      4. Masked polynomial alignment
      5. Floor shift to 0

    Repeating improves results because each pass refines the background mask
    using a cleaner baseline from the previous iteration.
    """
    aligned = image_nm.copy()

    for i in range(iterations):
        # Pre-step: light scar removal on raw/previous-iteration data
        aligned = remove_scars(aligned, threshold_factor=5.0)   # conservative pass

        # Step 1 — coarse alignment
        coarse  = row_align_poly(aligned, mask=None, degree=1)
        coarse -= nanmin_safe(coarse)

        # Step 2 — background mask
        mask = coarse < mask_threshold_nm

        # Step 3 — two scar correction steps
        scar_corrected = remove_scars(coarse)

        # Step 4 — masked polynomial alignment
        aligned  = row_align_poly(scar_corrected, mask=mask, degree=poly_degree)
        aligned -= nanmin_safe(aligned)
        aligned = remove_scars(aligned)

    return aligned, mask, coarse


# ═══════════════════════════════════════════════════════════════════════════════
# FILE I/O  (cached so re-runs on slider change are instant)
# ═══════════════════════════════════════════════════════════════════════════════

# @st.cache_data(show_spinner="Loading file…")
def load_file(filepath: str):
    """
    Load a JPK AFM image and convert height to nanometres.

    JPK stores height in SI units (metres). We detect this by checking whether
    the full Z range is below 1 mm; if so we multiply by 1e9 to convert to nm.

    Returns
    -------
    image_nm     : height array in nm
    scan_size_um : physical scan width in µm (used for axis labels)
    """
    image, px_to_nm = load_jpk(file_path=filepath, channel=CHANNEL)
    z_range  = np.nanmax(image) - np.nanmin(image)
    image_nm = image * 1e9 if z_range < 1e-3 else image.copy().astype(float)
    scan_size_um = image_nm.shape[1] * (px_to_nm * 1e6)
    return image_nm, scan_size_um


# @st.cache_data(show_spinner="Processing…")
def run_pipeline(filepath: str, mask_threshold_nm: float,
                 poly_degree: int = POLY_DEGREE,
                 iterations: int = 1):           # ← add this
    image_nm, scan_size_um = load_file(filepath)
    aligned, mask, coarse  = full_pipeline(image_nm,
                                           mask_threshold_nm=mask_threshold_nm,
                                           poly_degree=poly_degree,
                                           iterations=iterations)    # ← pass through
    return image_nm, aligned, mask, scan_size_um, nanmax_safe(coarse)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def get_display_extent(scan_size_um: float):
    """
    Choose axis unit based on scan size:
    - If scan size >= 1 µm → use µm
    - If scan size < 1 µm  → convert to nm to avoid scientific notation
    Returns (extent, unit_label)
    """
    if scan_size_um >= 1.0e9:
        return [0, scan_size_um/1.0e9, 0, scan_size_um/1.0e9], "µm"
    else:
        scan_size_nm = scan_size_um
        return [0, scan_size_nm/1.0e6, 0, scan_size_nm/1.0e6], "nm"

def make_figure(image_nm, aligned, mask, extent,
                zmin, zmax, mask_thresh, filename, set_colorbar_at_0,
                xy_unit="µm") -> plt.Figure:
    """
    Build a 3-panel matplotlib figure:
      Panel 1 — Raw image         (full data range, afmhot)
      Panel 2 — Background mask   (white = background, black = features)
      Panel 3 — Processed image   (clipped to [zmin, zmax], afmhot)

    The `set_colorbar_at_0` toggle only relabels colorbar ticks to start at 0
    — it never modifies image data, vmin, or vmax.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(filename, fontsize=10)

    # ── Panel 1: Raw ──────────────────────────────────────────────────────────
    raw_vmin = nanmin_safe(image_nm)
    raw_vmax = nanmax_safe(image_nm)

    im_raw = axes[0].imshow(
        image_nm, cmap="afmhot", origin="upper", extent=extent,
        interpolation="nearest", vmin=raw_vmin, vmax=raw_vmax
    )
    axes[0].set_title("Raw height (nm)")
    axes[0].set_xlabel(f"X ({xy_unit})")
    axes[0].set_ylabel(f"Y ({xy_unit})")
    cbar_raw = fig.colorbar(im_raw, ax=axes[0], fraction=0.046, pad=0.04)
    cbar_raw.set_label("Height (nm)")
    cbar_raw.formatter.set_scientific(False)
    cbar_raw.formatter.set_useOffset(False)
    cbar_raw.update_ticks()

    if set_colorbar_at_0:
        ticks = np.linspace(raw_vmin, raw_vmax, 5)
        cbar_raw.set_ticks(ticks)
        cbar_raw.set_ticklabels([f"{v - raw_vmin:.2f}" for v in ticks])

    # ── Panel 2: Background mask ──────────────────────────────────────────────
    axes[1].imshow(
        mask.astype(float), cmap="gray", origin="upper", extent=extent,
        interpolation="nearest", vmin=0, vmax=1
    )
    axes[1].set_title(f"Background mask (< {mask_thresh:.2f} nm)\n"
                      f"white = background · black = features")
    axes[1].set_xlabel(f"X ({xy_unit})")
    axes[1].set_ylabel(f"Y ({xy_unit})")

    # ── Panel 3: Processed ────────────────────────────────────────────────────
    im_al = axes[2].imshow(
        np.clip(aligned, zmin, zmax), cmap="afmhot", origin="upper", extent=extent,
        interpolation="nearest", vmin=zmin, vmax=zmax,   # never modified
    )
    axes[2].set_title("Processed height (nm)")
    axes[2].set_xlabel(f"X ({xy_unit})")
    axes[2].set_ylabel(f"Y ({xy_unit})")
    cbar_al = fig.colorbar(im_al, ax=axes[2], fraction=0.046, pad=0.04)
    cbar_al.set_label("Height (nm)")
    cbar_al.formatter.set_scientific(False)
    cbar_al.formatter.set_useOffset(False)
    cbar_al.update_ticks()

    if set_colorbar_at_0:
        ticks = np.linspace(zmin, zmax, 5)
        cbar_al.set_ticks(ticks)
        cbar_al.set_ticklabels([f"{v - zmin:.2f}" for v in ticks])

    plt.tight_layout()
    return fig


def make_processed_only_fig(aligned, extent, zmin, zmax,
                        set_colorbar_at_0=False, 
                        xy_unit="µm"):
    """
    Save only the processed image panel — no title, no scientific notation,
    clean colorbar, high resolution.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    im = ax.imshow(
        np.clip(aligned, zmin, zmax), cmap="afmhot", origin="upper",
        extent=extent, interpolation="nearest", vmin=zmin, vmax=zmax
    )
    ax.set_xlabel(f"X ({xy_unit})")
    ax.set_ylabel(f"Y ({xy_unit})")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Height (nm)")

    # Disable scientific notation on colorbar
    cbar.formatter.set_scientific(False)
    cbar.formatter.set_useOffset(False)
    cbar.update_ticks()

    if set_colorbar_at_0:
        n_ticks = 5
        abs_ticks   = np.linspace(zmin, zmax, n_ticks)
        label_ticks = np.linspace(0, zmax - zmin, n_ticks)
        cbar.set_ticks(abs_ticks)
        cbar.set_ticklabels([f"{v:.2f}" for v in label_ticks])

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="AFM Viewer", layout="wide",
                   initial_sidebar_state="expanded")
st.title("🔬 AFM JPK Image Viewer")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 File selection")
    folder = st.text_input("Folder path", value="",
                           placeholder="/path/to/jpk/files")

    if not folder:
        st.info("Enter a folder path to start.")
        st.stop()

    if not os.path.isdir(folder):
        st.error("Path does not exist.")
        st.stop()

    # jpk files are files that end with .jpk or in the case of FastScan
    # jpk files inside subfolders that start with "imaging-"
    jpk_files = []
    all_f = list(os.listdir(folder))
    all_f.sort()
    for f in all_f:
        if f.endswith(".jpk"):
            jpk_files.append(f)
        elif os.path.isdir(os.path.join(folder, f)) and f.startswith("imaging-"):
            for subf in os.listdir(os.path.join(folder, f)):
                if subf.endswith(".jpk"):
                    jpk_files.append(os.path.join(f, subf))

    if not jpk_files:
        st.warning("No .jpk files found in this folder.")
        st.stop()

    file_to_ind = {f: i for i, f in enumerate(jpk_files)}
    # selected_file = st.select_slider("Select file", options=jpk_files,
    #                                  value=jpk_files[0])
    selected_ind = st.select_slider("Select file", options=range(len(jpk_files)),
                                   value=0)
    selected_file = jpk_files[selected_ind]
    st.write(f"Selected file: `{selected_file}`")
    filepath  = os.path.join(folder, selected_file)
    # regardless of subfolder, save it in the main folder with the same name but .png extension
    save_path = os.path.splitext(
                    os.path.join(folder, 
                                 os.path.basename(filepath)
                                 )
                )[0] + ".png"

    # ── Pre-load to establish slider ranges ───────────────────────────────────
    # Run pipeline with mask=0 (no masking) to get the coarse height max,
    # which is used as the upper bound for all sliders.
    with st.spinner("Loading…"):
        _, _, _, _, coarse_max = run_pipeline(filepath, mask_threshold_nm=0.0)

    slider_max_val = round(float(coarse_max), 2) + 1 # add 1 nm margin

    # st.divider()
    st.header("⚙️ Processing")

    iterations = 1
    with st.container(horizontal=True,
                      horizontal_alignment="distribute",
                      vertical_alignment="center"):
        iters_on = st.toggle("Iterative processing", value=False,
                            help="Repeat scar correction + background removal + "
                                "line alignment. Each pass refines the mask using the cleaner baseline "
                                "from the previous iteration. 2–3 iterations is usually sufficient.")

        if iters_on:
            iterations = st.number_input(
                "Processing iterations",
                min_value=1, max_value=10,
                value=1, step=1,
                help="How many times to repeat scar correction + background removal + "
                    "line alignment. Each pass refines the mask using the cleaner baseline "
                    "from the previous iteration. 2–3 iterations is usually sufficient."
            )

    mask_thresh = st.slider(
        "Mask threshold (nm)",
        min_value=0.0, max_value=slider_max_val,
        value=round(slider_max_val / 2, 2), step=0.05,
        help="Pixels below this height are treated as background "
             "for polynomial fitting. Features above are ignored during alignment."
    )

    # st.divider()
    st.header("🎨 Display range")

    # Double (range) slider for Z min and Z max in one widget
    with st.container(horizontal=True,
                      horizontal_alignment="distribute",
                      vertical_alignment="center"):
        use_fixed_range = st.toggle("Use fixed Z range", value=False,
                                    help="Use a fixed Z range for all images.")
        if use_fixed_range:
            window_size = st.number_input(
                "Z range colorbar size (nm)",
                min_value=1.0, max_value=slider_max_val,
                value=3.0, step=0.01,
                help="Set the maximum value for the Z range slider. Adjust if the default does not suit your data."
            )
    if use_fixed_range:
        zmin = st.slider(
            "Baseline Z range (nm)",
            min_value=0.0, max_value=slider_max_val,
            value=0.0,   # default: full range (max = mask_max)
            step=0.05,
            help="Clip the processed image colorbar to this height range."
        )
        zmax = zmin + window_size
    else:
        zmin, zmax = st.slider(
                "Z range (nm)",
                min_value=0.0, max_value=slider_max_val,
                value=(0.0, slider_max_val),   # default: full range (max = mask_max)
                step=0.05,
                help="Clip the processed image colorbar to this height range."
            )

    set_colorbar_at_0 = st.checkbox("Set Z min to 0", value=True,
                                   help="Force the minimum of the Z range to be 0 nm. Uncheck to allow custom Z min.")
    
# ── Validation ────────────────────────────────────────────────────────────────
if zmin >= zmax:
    st.error("Z min must be less than Z max. Adjust the Z range slider.")
    st.stop()

# ── Run pipeline & render ─────────────────────────────────────────────────────
v_num = st.number_input("Save image version number", min_value=0, value=0, step=1)
save_path = os.path.splitext(save_path)[0] + f"_v{v_num}.png"
if os.path.isfile(save_path):
    st.warning(f"Found existing PNG → `{save_path}`")
    with st.container(horizontal=True, 
                      horizontal_alignment="distribute",
                      vertical_alignment="center"):
        redo = st.toggle("Reprocess JPK file", value=False, help="Uncheck to load the existing PNG instead of re-processing the JPK.")
    if not redo:
        st.info(f"Displaying saved PNG → `{save_path}`")
        img = plt.imread(save_path)
        st.image(img)
        st.stop()

image_nm, aligned, mask, scan_size_um, _ = run_pipeline(
    filepath, mask_threshold_nm=mask_thresh, iterations=iterations
)
extent, xy_unit = get_display_extent(scan_size_um)

processed = None
if st.toggle("Show Processed image only", value=False):
    fig = make_processed_only_fig(aligned, extent, zmin, zmax,
                                set_colorbar_at_0=set_colorbar_at_0,
                                xy_unit=xy_unit)
    processed = fig
    st.pyplot(fig)
else:
    fig = make_figure(image_nm, aligned, mask, extent,
                    zmin, zmax, mask_thresh, selected_file,
                    set_colorbar_at_0=set_colorbar_at_0, xy_unit=xy_unit)
    st.pyplot(fig)
    plt.close(fig)

# ── Save ──────────────────────────────────────────────────────────────────────
# if st.button("💾 Save PNG", use_container_width=True):
#     fig2 = make_figure(image_nm, aligned, mask, extent,
#                        zmin, zmax, mask_thresh, selected_file)
#     fig2.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close(fig2)
#     st.success(f"Saved → `{save_path}`")
if st.button("💾 Save Processed PNG", use_container_width=True):
    if processed is None:
        processed = make_processed_only_fig(aligned, extent, zmin, zmax,
                                    set_colorbar_at_0=set_colorbar_at_0,
                                    xy_unit=xy_unit)
    processed.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(processed)
    st.success(f"Saved → `{save_path}`")